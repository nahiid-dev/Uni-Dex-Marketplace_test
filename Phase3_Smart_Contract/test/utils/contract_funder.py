import logging
import time
from web3 import Web3
from eth_account import Account
from . import web3_utils

logger = logging.getLogger(__name__)

MIN_GAS_FOR_TRANSFER = 50000

def ensure_precise_token_balances(
    contract_address: str,
    token0_address: str,
    token0_decimals: int,
    target_token0_amount_readable: float,
    token1_address: str,
    token1_decimals: int,
    target_token1_amount_readable: float,
    funding_account_private_key: str
) -> bool:
    """
    Ensures the target contract has precise amounts of token0 and token1.
    Funds from the funding_account.
    """
    if not web3_utils.w3 or not web3_utils.w3.is_connected():
        if not web3_utils.init_web3():
            logger.error("Web3 connection failed in ensure_precise_token_balances.")
            return False
    
    w3 = web3_utils.w3
    try:
        funding_account = Account.from_key(funding_account_private_key)
        contract_checksum_addr = Web3.to_checksum_address(contract_address)

        tokens_to_fund = [
            {"name": "Token0", "address": token0_address, "decimals": token0_decimals, "target_readable": target_token0_amount_readable},
            {"name": "Token1", "address": token1_address, "decimals": token1_decimals, "target_readable": target_token1_amount_readable},
        ]

        for token_info in tokens_to_fund:
            token_contract = web3_utils.get_contract(token_info["address"], "IERC20")
            
            target_wei = int(token_info["target_readable"] * (10**token_info["decimals"]))
            current_bal_wei = token_contract.functions.balanceOf(contract_checksum_addr).call()
            
            needed_wei = target_wei - current_bal_wei

            if needed_wei == 0:
                logger.info(f"Contract already has target balance of {token_info['name']} ({token_info['target_readable']}).")
                continue
            elif needed_wei < 0:
                logger.warning(f"Contract has MORE {token_info['name']} ({current_bal_wei / (10**token_info['decimals'])}) than target ({token_info['target_readable']}). Manual adjustment might be needed if exactness is critical by reducing balance.")
                continue

            funder_bal_wei = token_contract.functions.balanceOf(funding_account.address).call()
            if funder_bal_wei < needed_wei:
                logger.error(f"Funding account {funding_account.address} has insufficient {token_info['name']} to fund contract. Has: {funder_bal_wei / (10**token_info['decimals'])}, Needs: {needed_wei / (10**token_info['decimals'])}")
                return False

            logger.info(f"Funding contract {contract_address} with {needed_wei / (10**token_info['decimals']):.6f} of {token_info['name']}.")
            
            tx_params = {
                'from': funding_account.address,
                'nonce': w3.eth.get_transaction_count(funding_account.address),
                'chainId': int(w3.net.version)
            }
            transfer_tx = token_contract.functions.transfer(contract_checksum_addr, needed_wei).build_transaction(tx_params)
            
            try:
                gas_estimate = w3.eth.estimate_gas(transfer_tx)
                transfer_tx['gas'] = int(gas_estimate * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for {token_info['name']} transfer: {e}. Using default.")
                transfer_tx['gas'] = MIN_GAS_FOR_TRANSFER * 2

            receipt = web3_utils.send_transaction(transfer_tx)
            if not receipt or receipt.status == 0:
                logger.error(f"Failed to transfer {token_info['name']} to contract {contract_address}. TxHash: {receipt.transactionHash.hex() if receipt else 'N/A'}")
                return False
            logger.info(f"{token_info['name']} transfer successful to {contract_address}. TxHash: {receipt.transactionHash.hex()}")
            time.sleep(1)

        return True
    except Exception as e:
        logger.exception(f"Error in ensure_precise_token_balances for {contract_address}: {e}")
        return False
