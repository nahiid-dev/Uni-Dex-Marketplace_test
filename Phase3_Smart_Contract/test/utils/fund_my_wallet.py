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
DEFAULT_RPC_URL = "http://127.0.0.1:8545" # این آدرس باید با آنچه در run_fork_test.sh استفاده می‌شود، یکی باشد
MAX_RETRIES = 5
RETRY_DELAY = 3 # ثانیه
GAS_BUFFER = 1.2 # ضریب اطمینان برای گاز
MIN_ETH_BALANCE_FOR_WHALE = 0.1 # حداقل اتر مورد نیاز برای حساب نهنگ برای پرداخت هزینه گاز

# Token Configuration - مقادیر amount برای خوانایی بیشتر است، در کد به واحد wei تبدیل می‌شوند
TOKEN_CONFIG = {
    "WETH": {
        "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "whale": "0x2f0b23f53734252bda2277357e97e1517d6b042a", # یک حساب شناخته شده با WETH زیاد
        "amount_readable": 300, # مقدار WETH برای انتقال به deployer
        "decimals": 18
    },
    "USDC": {
        "address": "0xA0b86991c6218b36c1d19D4A2e9Eb0cE3606eB48",
        "whale": "0x55fe002aeff02f77364de339a1292923a15844b8", # یک حساب شناخته شده با USDC زیاد
        "amount_readable": 50000, # مقدار USDC برای انتقال به deployer
        "decimals": 6
    }
}

# Minimal ERC20 ABI for transfer and balanceOf
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
        logging.StreamHandler(sys.stdout), # اطمینان از خروج لاگ‌ها به stdout
        logging.FileHandler('funding.log', mode='a') # اضافه کردن به فایل لاگ
    ]
)
logger = logging.getLogger('wallet_funder')

class WalletFunder:
    def __init__(self):
        """Initialize with Web3 connection and deployer address."""
        # استفاده از متغیر محیطی MAINNET_FORK_RPC_URL که توسط run_fork_test.sh تنظیم می‌شود
        # یا مقدار پیش‌فرض اگر مستقیماً اجرا شود.
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
                if w3.is_connected():
                    logger.info(f"Successfully connected to Ethereum node. Chain ID: {w3.eth.chain_id}")
                    return w3
                raise ConnectionError(f"Web3 provider at {self.rpc_url} is not connected.")
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1}/{MAX_RETRIES} to {self.rpc_url} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logger.critical("Failed to connect to Web3 provider after multiple retries.")
                    sys.exit(1)
                sleep(RETRY_DELAY * (attempt + 1)) # افزایش زمان انتظار
        raise ConnectionError("Web3 initialization failed") # نباید به اینجا برسد

    def _get_deployer_address(self) -> str:
        """Get and validate deployer address from environment variable."""
        address = os.getenv('DEPLOYER_ADDRESS')
        if not address:
            logger.critical("DEPLOYER_ADDRESS environment variable not set.")
            sys.exit(1)
        
        try:
            return self.w3.to_checksum_address(address)
        except ValueError as e:
            logger.critical(f"Invalid DEPLOYER_ADDRESS format ('{address}'): {str(e)}")
            sys.exit(1)

    def _make_rpc_request(self, method: str, params: list) -> Optional[Any]:
        """Make a direct JSON-RPC request with retries."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.w3.eth.get_block_number() # استفاده از شماره بلاک برای id منحصر به فرد
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(self.rpc_url, json=payload, timeout=60)
                response.raise_for_status() # خطا برای status code های بد (4xx یا 5xx)
                result = response.json()
                
                if 'error' in result and result['error']:
                    logger.error(f"RPC error for method '{method}': {result['error']}")
                    return None # یا می‌توان یک Exception خاص برگرداند
                return result.get('result')
            except requests.exceptions.RequestException as e:
                logger.error(f"RPC request for method '{method}' failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return None
                sleep(RETRY_DELAY * (attempt + 1))
        return None

    def set_eth_balance(self, address: str, eth_amount: float) -> bool:
        """Set ETH balance for a given address using hardhat_setBalance."""
        try:
            checksum_addr = self.w3.to_checksum_address(address)
            wei_amount = self.w3.to_wei(eth_amount, 'ether')
            
            logger.info(f"Attempting to set ETH balance for {checksum_addr[:10]}... to {eth_amount} ETH (Value: {hex(wei_amount)})")
            
            # استفاده از `hardhat_setBalance` که یک متد خاص Hardhat/Ganache است
            rpc_result = self._make_rpc_request("hardhat_setBalance", [checksum_addr, hex(wei_amount)])
            
            if rpc_result is None: # اگر _make_rpc_request در صورت خطا None برگرداند
                logger.error(f"Failed to set ETH balance for {checksum_addr} via RPC method hardhat_setBalance.")
                return False
                
            # تایید بالانس جدید
            # ممکن است کمی تاخیر برای اعمال تغییرات وجود داشته باشد.
            sleep(0.5) # تاخیر کوتاه برای اطمینان از اعمال تغییر
            new_balance_wei = self.w3.eth.get_balance(checksum_addr)
            # به دلیل округление، ممکن است دقیقاً برابر نباشد، پس بررسی می‌کنیم که حداقل برابر باشد
            if new_balance_wei >= wei_amount:
                logger.info(f"✅ ETH balance successfully set for {checksum_addr[:10]}.... New balance: {self.w3.from_wei(new_balance_wei, 'ether')} ETH")
                return True
            else:
                logger.error(f"ETH balance verification failed for {checksum_addr[:10]}.... Expected >= {eth_amount}, Got: {self.w3.from_wei(new_balance_wei, 'ether')} ETH.")
                return False
        except Exception as e:
            logger.error(f"Error in set_eth_balance for {address}: {str(e)}", exc_info=True)
            return False

    def transfer_tokens_from_whale(self, token_symbol: str, token_config: Dict[str, Any]) -> bool:
        """Manages the entire process of transferring tokens from a whale account."""
        whale_addr_checksum = None
        try:
            token_addr_checksum = self.w3.to_checksum_address(token_config['address'])
            whale_addr_checksum = self.w3.to_checksum_address(token_config['whale'])
            # تبدیل amount_readable به واحد wei توکن
            amount_to_transfer_wei = int(token_config['amount_readable'] * (10 ** token_config['decimals']))
            
            logger.info(f"\n=== Processing {token_symbol} transfer from whale {whale_addr_checksum[:10]}... ===")
            
            if not self._impersonate_account(whale_addr_checksum): return False
            if not self._ensure_eth_for_gas(whale_addr_checksum, MIN_ETH_BALANCE_FOR_WHALE): return False
            
            success = self._execute_token_transfer_as_impersonator(
                token_symbol,
                token_addr_checksum,
                whale_addr_checksum, # from_address
                self.deployer_address, # to_address
                amount_to_transfer_wei,
                token_config['decimals']
            )
            return success
            
        except Exception as e:
            logger.error(f"Critical error in transfer_tokens_from_whale for {token_symbol}: {str(e)}", exc_info=True)
            return False
        finally:
            if whale_addr_checksum:
                self._stop_impersonating_account(whale_addr_checksum)

    def _impersonate_account(self, address_to_impersonate: str) -> bool:
        """Impersonate an account using hardhat_impersonateAccount."""
        logger.info(f"Attempting to impersonate account {address_to_impersonate[:10]}...")
        rpc_result = self._make_rpc_request("hardhat_impersonateAccount", [address_to_impersonate])
        if rpc_result is None: # یا هر شرطی که نشان دهنده شکست باشد
             logger.error(f"Failed to impersonate account {address_to_impersonate}.")
             return False
        logger.info(f"Successfully started impersonating {address_to_impersonate[:10]}...")
        return True

    def _stop_impersonating_account(self, address_to_stop_impersonating: str) -> bool:
        """Stop impersonating an account using hardhat_stopImpersonatingAccount."""
        logger.info(f"Attempting to stop impersonating account {address_to_stop_impersonating[:10]}...")
        rpc_result = self._make_rpc_request("hardhat_stopImpersonatingAccount", [address_to_stop_impersonating])
        if rpc_result is None:
            logger.warning(f"Failed to explicitly stop impersonating {address_to_stop_impersonating}. This might be okay if node handles cleanup.")
            return False # یا True اگر عدم موفقیت بحرانی نباشد
        logger.info(f"Successfully stopped impersonating {address_to_stop_impersonating[:10]}...")
        return True

    def _ensure_eth_for_gas(self, account_address: str, min_eth_required: float) -> bool:
        """Ensure the specified account has enough ETH for gas fees."""
        min_wei_required = self.w3.to_wei(min_eth_required, 'ether')
        current_balance_wei = self.w3.eth.get_balance(account_address)
        
        if current_balance_wei >= min_wei_required:
            logger.info(f"Account {account_address[:10]}... has sufficient ETH ({self.w3.from_wei(current_balance_wei, 'ether')} ETH) for gas.")
            return True
            
        # اگر اتر کافی نیست، به آن اتر اضافه کن (مثلاً ۱ اتر کامل)
        eth_to_add = max(min_eth_required, 1.0) # حداقل یک اتر اضافه کن تا مطمئن باشیم
        logger.info(f"Account {account_address[:10]}... has insufficient ETH. Current: {self.w3.from_wei(current_balance_wei, 'ether')} ETH. Attempting to add {eth_to_add} ETH...")
        return self.set_eth_balance(account_address, eth_to_add)

    def _execute_token_transfer_as_impersonator(self, token_symbol: str, token_contract_address: str, 
                                                from_address_impersonated: str, to_address_recipient: str,
                                                amount_in_wei: int, token_decimals: int) -> bool:
        """Execute the token transfer transaction as the impersonated account."""
        try:
            token_contract = self.w3.eth.contract(address=token_contract_address, abi=ERC20_ABI)
            
            whale_balance_wei = token_contract.functions.balanceOf(from_address_impersonated).call()
            whale_balance_readable = whale_balance_wei / (10**token_decimals)
            amount_readable = amount_in_wei / (10**token_decimals)

            if whale_balance_wei < amount_in_wei:
                logger.error(f"Impersonated whale {from_address_impersonated[:10]}... has insufficient {token_symbol} balance. Has: {whale_balance_readable}, Needs: {amount_readable}")
                return False
            logger.info(f"Impersonated whale {from_address_impersonated[:10]}... has {whale_balance_readable} {token_symbol}. Attempting to transfer {amount_readable} {token_symbol} to {to_address_recipient[:10]}...")
            
            # ساخت تراکنش بدون ارسال مستقیم (برای استفاده با eth_sendTransaction از طریق RPC)
            # nonce به طور خودکار توسط نود Hardhat برای حساب impersonated مدیریت می‌شود.
            tx_dict = {
                'from': from_address_impersonated,
                'to': token_contract_address, # تراکنش به قرارداد توکن ارسال می‌شود
                'value': '0x0', # انتقال توکن است، نه اتر
                # 'gasPrice': self.w3.eth.gas_price # اجازه دهید Hardhat مدیریت کند یا از یک مقدار ثابت استفاده کنید
            }

            # تخمین گاز
            try:
                # gas_estimate = token_contract.functions.transfer(to_address_recipient, amount_in_wei).estimate_gas({'from': from_address_impersonated, 'to': token_contract_address})
                # tx_dict['gas'] = hex(int(gas_estimate * GAS_BUFFER))

                # data برای فراخوانی متد transfer
                tx_dict['data'] = token_contract.encodeABI(fn_name="transfer", args=[to_address_recipient, amount_in_wei])
                
                # برای estimate_gas با data خام
                gas_estimate_params = {'from': from_address_impersonated, 'to': token_contract_address, 'data': tx_dict['data']}
                gas_estimate = self.w3.eth.estimate_gas(gas_estimate_params)
                tx_dict['gas'] = hex(int(gas_estimate * GAS_BUFFER))

            except ContractLogicError as e: # اگر قرارداد به دلایلی revert کند
                logger.error(f"Gas estimation for {token_symbol} transfer failed due to contract logic: {str(e)}. This might indicate an issue with allowances or other contract checks if the whale is a contract itself.")
                return False
            except Exception as e:
                logger.error(f"Gas estimation for {token_symbol} transfer failed: {str(e)}. Using a default high gas limit.")
                tx_dict['gas'] = hex(500000) # مقدار پیش‌فرض بالا در صورت خطا


            logger.info(f"Sending {token_symbol} transfer transaction: {tx_dict}")
            # ارسال تراکنش از طریق متد RPC eth_sendTransaction چون از حساب impersonated است
            tx_hash_hex = self._make_rpc_request("eth_sendTransaction", [tx_dict])
            
            if not tx_hash_hex:
                logger.error(f"Failed to send {token_symbol} transfer transaction from {from_address_impersonated[:10]}.... RPC call did not return a transaction hash.")
                return False
            
            logger.info(f"{token_symbol} transfer transaction sent, TxHash: {tx_hash_hex}. Waiting for receipt...")
            # انتظار برای رسید تراکنش
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=180) # افزایش timeout
            
            if receipt and receipt.status == 1:
                logger.info(f"✅ {token_symbol} transfer successful from {from_address_impersonated[:10]}... to {to_address_recipient[:10]}.... Gas used: {receipt.gasUsed}")
                # بررسی بالانس دریافت کننده
                final_recipient_balance = token_contract.functions.balanceOf(to_address_recipient).call()
                logger.info(f"Recipient {to_address_recipient[:10]}... now has {final_recipient_balance / (10**token_decimals)} {token_symbol}.")
                return True
            else:
                logger.error(f"{token_symbol} transfer transaction from {from_address_impersonated[:10]}... reverted or receipt not found. Status: {receipt.status if receipt else 'N/A'}")
                return False
                
        except Exception as e:
            logger.error(f"Error during token transfer execution for {token_symbol}: {str(e)}", exc_info=True)
            return False

def main():
    try:
        logger.info("\n" + "="*30 + " Starting Wallet Funding Script " + "="*30)
        
        funder = WalletFunder()
        overall_success = True
        
        # 1. Fund deployer account with ETH
        logger.info("\n--- Step 1 of 2: Funding deployer account with ETH ---")
        deployer_eth_amount = 100.0 # مقدار اتری که به حساب deployer داده می‌شود
        if not funder.set_eth_balance(funder.deployer_address, deployer_eth_amount):
            logger.error(f"Failed to fund deployer {funder.deployer_address[:10]}... with {deployer_eth_amount} ETH.")
            overall_success = False
        
        # 2. Transfer specified tokens to the deployer account from whales
        logger.info("\n--- Step 2 of 2: Transferring ERC20 tokens to deployer account ---")
        if overall_success: # فقط اگر مرحله قبل موفق بود ادامه بده
            for token_symbol, config in TOKEN_CONFIG.items():
                logger.info(f"--- Processing {token_symbol} ---")
                if not funder.transfer_tokens_from_whale(token_symbol, config):
                    logger.error(f"Failed to transfer {token_symbol} to deployer {funder.deployer_address[:10]}...")
                    overall_success = False
                    # تصمیم بگیرید که آیا می‌خواهید در صورت شکست یک توکن، ادامه دهید یا خیر
                    # break # برای توقف در صورت اولین شکست
        else:
            logger.warning("Skipping token transfers due to failure in ETH funding for deployer.")
            
        logger.info("\n" + "="*30 + " Wallet Funding Script Finished " + "="*30)
        if overall_success:
            logger.info("✅✅✅ All funding operations completed successfully! ✅✅✅")
            sys.exit(0)
        else:
            logger.error("‼️‼️‼️ Some funding operations failed. Please check logs. ‼️‼️‼️")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()