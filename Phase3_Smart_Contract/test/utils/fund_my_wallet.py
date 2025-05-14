#!/usr/bin/env python3
"""
Wallet Funding Script for Hardhat Fork Testing
Version: 3.0 (Final Working Version)
"""

import os
import sys
import logging
import requests
from time import sleep
from web3 import Web3, HTTPProvider
from web3.exceptions import ContractLogicError, TransactionNotFound
from typing import Dict, Any, Optional

# --- Constants ---
DEFAULT_RPC_URL = "http://127.0.0.1:8545"
MAX_RETRIES = 5
RETRY_DELAY = 3
GAS_BUFFER = 1.2
MIN_ETH_BALANCE = 0.1

# Token Configuration
TOKEN_CONFIG = {
    "WETH": {
        "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "whale": "0x2f0b23f53734252bda2277357e97e1517d6b042a",
        "amount": 20,
        "decimals": 18
    },
    "USDC": {
        "address": "0xA0b86991c6218b36c1d19D4A2e9Eb0cE3606eB48",
        "whale": "0x55fe002aeff02f77364de339a1292923a15844b8",
        "amount": 50000,
        "decimals": 6
    }
}

# ERC20 ABI
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('funding.log')
    ]
)
logger = logging.getLogger('wallet_funder')

class WalletFunder:
    def __init__(self):
        """Initialize with Web3 connection and deployer address."""
        self.rpc_url = os.getenv('MAINNET_FORK_RPC_URL', DEFAULT_RPC_URL)
        logger.info(f"Initializing Web3 connection to {self.rpc_url}")
        self.w3 = self._init_web3()
        self.deployer_address = self._get_deployer_address()
        logger.info(f"Using deployer address: {self.deployer_address}")

    def _init_web3(self) -> Web3:
        """Initialize and verify Web3 connection."""
        for attempt in range(MAX_RETRIES):
            try:
                w3 = Web3(HTTPProvider(self.rpc_url, request_kwargs={'timeout': 60}))
                
                # Verify connection
                if w3.is_connected():
                    logger.info(f"Connected to chain ID: {w3.eth.chain_id}")
                    return w3
                raise ConnectionError("Web3 not connected")
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logger.critical("Failed to connect to Web3 provider")
                    sys.exit(1)
                sleep(RETRY_DELAY)
        raise ConnectionError("Web3 initialization failed")

    def _get_deployer_address(self) -> str:
        """Get and validate deployer address."""
        address = os.getenv('DEPLOYER_ADDRESS')
        if not address:
            logger.critical("DEPLOYER_ADDRESS not set")
            sys.exit(1)
        
        try:
            return self.w3.to_checksum_address(address)
        except ValueError as e:
            logger.critical(f"Invalid address format: {str(e)}")
            sys.exit(1)

    def make_rpc_request(self, method: str, params: list) -> Optional[Dict]:
        """Make JSON-RPC request with retries."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.rpc_url,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                if 'error' in result:
                    logger.error(f"RPC error: {result['error']}")
                    return None
                return result.get('result')
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return None
                sleep(RETRY_DELAY)
        return None

    def set_eth_balance(self, address: str, eth_amount: float) -> bool:
        """Set ETH balance for an address."""
        try:
            checksum_addr = self.w3.to_checksum_address(address)
            wei_amount = self.w3.to_wei(eth_amount, 'ether')
            
            logger.info(f"Setting balance for {checksum_addr[:10]}... to {eth_amount} ETH")
            
            result = self.make_rpc_request(
                "hardhat_setBalance",
                [checksum_addr, hex(wei_amount)]
            )
            
            if result is None:
                logger.error("Failed to set balance via RPC")
                return False
                
            # Verify new balance
            new_balance = self.w3.eth.get_balance(checksum_addr)
            if new_balance >= wei_amount:
                logger.info(f"✅ New balance: {self.w3.from_wei(new_balance, 'ether')} ETH")
                return True
            else:
                logger.error(f"Balance verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error setting balance: {str(e)}")
            return False

    def transfer_tokens(self, token_symbol: str, token_config: Dict[str, Any]) -> bool:
        """Complete token transfer process."""
        whale_addr = None
        try:
            token_addr = self.w3.to_checksum_address(token_config['address'])
            whale_addr = self.w3.to_checksum_address(token_config['whale'])
            amount = int(token_config['amount'] * (10 ** token_config['decimals']))
            
            logger.info(f"\n=== Processing {token_symbol} transfer ===")
            
            # 1. Impersonate whale
            if not self._impersonate_account(whale_addr):
                return False
                
            # 2. Fund whale with ETH for gas
            if not self._ensure_eth_balance(whale_addr, MIN_ETH_BALANCE):
                return False
                
            # 3. Execute token transfer
            return self._execute_token_transfer(
                token_symbol,
                token_addr,
                whale_addr,
                amount,
                token_config['decimals']
            )
            
        except Exception as e:
            logger.error(f"Transfer failed: {str(e)}")
            return False
        finally:
            if whale_addr:
                self._stop_impersonating_account(whale_addr)

    def _impersonate_account(self, address: str) -> bool:
        """Impersonate an account."""
        logger.info(f"Impersonating {address[:10]}...")
        result = self.make_rpc_request("hardhat_impersonateAccount", [address])
        return result is not None

    def _stop_impersonating_account(self, address: str) -> bool:
        """Stop impersonating an account."""
        logger.info(f"Stopping impersonation of {address[:10]}...")
        result = self.make_rpc_request("hardhat_stopImpersonatingAccount", [address])
        return result is not None

    def _ensure_eth_balance(self, address: str, min_eth: float) -> bool:
        """Ensure address has sufficient ETH."""
        min_wei = self.w3.to_wei(min_eth, 'ether')
        current_balance = self.w3.eth.get_balance(address)
        
        if current_balance >= min_wei:
            logger.info(f"Account has sufficient ETH")
            return True
            
        funding_amount = max(min_eth, 1.0)
        logger.info(f"Funding with {funding_amount} ETH...")
        return self.set_eth_balance(address, funding_amount)

    def _execute_token_transfer(self, token_symbol: str, token_addr: str, 
                              whale_addr: str, amount_wei: int, decimals: int) -> bool:
        """Execute token transfer."""
        try:
            token_contract = self.w3.eth.contract(address=token_addr, abi=ERC20_ABI)
            
            # Check balance
            whale_balance = token_contract.functions.balanceOf(whale_addr).call()
            if whale_balance < amount_wei:
                logger.error("Insufficient token balance")
                return False
                
            # Prepare transaction
            tx_data = token_contract.functions.transfer(
                self.deployer_address,
                amount_wei
            ).build_transaction({
                'from': whale_addr,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Estimate gas
            try:
                gas_estimate = token_contract.functions.transfer(
                    self.deployer_address,
                    amount_wei
                ).estimate_gas({'from': whale_addr})
                tx_data['gas'] = int(gas_estimate * GAS_BUFFER)
            except Exception as e:
                logger.error(f"Gas estimation failed: {str(e)}")
                return False
                
            # Send transaction
            tx_hash = self.make_rpc_request("eth_sendTransaction", [{
                'from': whale_addr,
                'to': token_addr,
                'data': tx_data['data'],
                'gas': hex(tx_data['gas']),
                'gasPrice': hex(tx_data['gasPrice']),
                'value': '0x0'
            }])
            
            if not tx_hash:
                logger.error("Failed to send transaction")
                return False
                
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                logger.info("✅ Transfer successful")
                return True
            else:
                logger.error("Transaction reverted")
                return False
                
        except Exception as e:
            logger.error(f"Transfer execution failed: {str(e)}")
            return False

def main():
    try:
        logger.info("\n=== Starting Wallet Funding ===")
        
        funder = WalletFunder()
        success = True
        
        # Fund deployer with ETH
        logger.info("\n[1/2] Funding deployer with ETH...")
        if not funder.set_eth_balance(funder.deployer_address, 100.0):
            logger.error("ETH funding failed")
            success = False
        
        # Transfer tokens
        logger.info("\n[2/2] Transferring tokens...")
        for symbol, config in TOKEN_CONFIG.items():
            logger.info(f"\nProcessing {symbol}...")
            if not funder.transfer_tokens(symbol, config):
                logger.error(f"{symbol} transfer failed")
                success = False
        
        if success:
            logger.info("\n✅ All operations completed")
            sys.exit(0)
        else:
            logger.error("\n‼️ Some operations failed")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()