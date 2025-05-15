import os
import sys
import json
import time
import logging
import requests
import math
import csv
from datetime import datetime
from pathlib import Path
from web3 import Web3
from eth_account import Account
from decimal import Decimal, getcontext

# --- Adjust path imports ---
# This ensures that the 'Phase3_Smart_Contract' directory is in sys.path
current_file_path = Path(__file__).resolve()
# project_root should be Phase3_Smart_Contract
# test/predictive/predictive_test.py -> test/predictive -> test -> Phase3_Smart_Contract
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the web3_utils module itself
import test.utils.web3_utils as web3_utils
import test.utils.contract_funder as contract_funder

# Precision for Decimal calculations
getcontext().prec = 78

try:
    from test.utils.test_base import LiquidityTestBase
    from test.utils.web3_utils import send_transaction, get_contract
except ImportError as e:
    # This log might not be visible if the script itself fails on the module import above.
    # The primary error will be the ModuleNotFoundError from the `import test.utils.web3_utils`
    print(f"ERROR importing from test.utils in predictive_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6)  # 1000 USDC
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272
MAX_TICK_CONST = 887272

# --- Setup Logging ---
# Configure basicConfig at a higher level or ensure it's only called once.
# If run as main, this is fine. If imported, could conflict.
if not logging.getLogger('predictive_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("predictive_test.log"), # Log to a file in the execution directory
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger('predictive_test')

# --- Define path for addresses and results ---
ADDRESS_FILE_PREDICTIVE = project_root / 'predictiveManager_address.json'
RESULTS_FILE = project_root / 'position_results_predictive.csv'
LSTM_API_URL = os.getenv('LSTM_API_URL', 'http://95.216.156.73:5000/predict_price?symbol=ETHUSDT&interval=4h')

logger.info(f"Project Root for Predictive Test (from predictive_test.py): {project_root}")
logger.info(f"Predictive Address File: {ADDRESS_FILE_PREDICTIVE}")
logger.info(f"Predictive Results File: {RESULTS_FILE}")
logger.info(f"LSTM API URL: {LSTM_API_URL}")


class PredictiveTest(LiquidityTestBase):
    """Test implementation for PredictiveLiquidityManager contract testing on a fork."""

    def __init__(self, contract_address: str):
        self.ACTION_STATES = {
            "INIT": "init", "SETUP_FAILED": "setup_failed",
            "POOL_READ_FAILED": "pool_read_failed", "API_FAILED": "api_failed",
            "CALCULATION_FAILED": "calculation_failed", "FUNDING_FAILED": "funding_failed",
            "TX_SENT": "tx_sent", "TX_SUCCESS_ADJUSTED": "tx_success_adjusted",
            "TX_REVERTED": "tx_reverted", "TX_WAIT_FAILED": "tx_wait_failed",
            "METRICS_UPDATE_FAILED": "metrics_update_failed",
            "UNEXPECTED_ERROR": "unexpected_error"
        }
        super().__init__(contract_address, "PredictiveLiquidityManager")
        self.metrics = self._reset_metrics()
        self.pool_address = None
        self.pool_contract = None

    def _reset_metrics(self):
        return {
            'timestamp': None, 'contract_type': 'Predictive',
            'action_taken': self.ACTION_STATES["INIT"], 'tx_hash': None,
            'range_width_multiplier_setting': None,
            'predictedPrice_api': None, 'predictedTick_calculated': None,
            'external_api_eth_price': None,
            'actualPrice_pool': None, 'sqrtPriceX96_pool': 0, 'currentTick_pool': 0,
            'targetTickLower_calculated': 0, 'targetTickUpper_calculated': 0,
            'initial_contract_balance_token0': None,
            'initial_contract_balance_token1': None,
            'finalTickLower_contract': 0, 'finalTickUpper_contract': 0, 'liquidity_contract': 0,
            'amount0_provided_to_mint': None,
            'amount1_provided_to_mint': None,
            'fees_collected_token0': None,
            'fees_collected_token1': None,
            'gas_used': 0, 'gas_cost_eth': 0.0, 'error_message': ""
        }

    def setup(self, desired_range_width_multiplier: int) -> bool:
        if not super().setup():
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("web3_utils.w3 not available in PredictiveTest setup after base.setup()")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup"
                return False

            factory_address = self.contract.functions.factory().call()
            factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            fee = self.contract.functions.fee().call()
            self.pool_address = factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Predictive pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
                
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            logger.info(f"Predictive Pool contract initialized at {self.pool_address}")

            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Predictive contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier
            tx_params = {
                'from': Account.from_key(os.getenv('PRIVATE_KEY')).address,
            }
            tx_set_rwm = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).build_transaction(tx_params)
            receipt_rwm = send_transaction(tx_set_rwm)
            if not receipt_rwm or receipt_rwm.status == 0:
                logger.error(f"Failed to set rangeWidthMultiplier for Predictive contract. TxHash: {receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'}")
                self.metrics['error_message'] += ";Failed to set RWM"
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Predictive contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Predictive setup failed getting pool: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup pool error: {str(e)}"
            return False

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        self.metrics = self._reset_metrics()
        try:
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from predictive contract for metrics.")

        adjustment_call_success = False
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position.")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx"
            self.save_metrics()
            return False
        funding_account = Account.from_key(private_key_env)
        try:
            predicted_price = self.get_predicted_price_from_api()
            if predicted_price is None: self.save_metrics(); return False
            predicted_tick = self.calculate_tick_from_price(predicted_price)
            if predicted_tick is None: self.save_metrics(); return False
            self.update_pool_and_position_metrics(final_update=False)
            logger.info("Ensuring precise token balances for Predictive contract...")
            if not contract_funder.ensure_precise_token_balances(
                contract_address=self.contract_address,
                token0_address=self.token0,
                token0_decimals=self.token0_decimals,
                target_token0_amount_readable=target_usdc_balance,
                token1_address=self.token1,
                token1_decimals=self.token1_decimals,
                target_token1_amount_readable=target_weth_balance,
                funding_account_private_key=private_key_env
            ):
                logger.error("Precise funding for Predictive contract failed.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Precise contract funding failed"
                self.save_metrics()
                return False
            try:
                token0_contract = get_contract(self.token0, "IERC20")
                token1_contract = get_contract(self.token1, "IERC20")
                self.metrics['initial_contract_balance_token0'] = token0_contract.functions.balanceOf(self.contract_address).call()
                self.metrics['initial_contract_balance_token1'] = token1_contract.functions.balanceOf(self.contract_address).call()
            except Exception as bal_err:
                logger.warning(f"Could not read initial contract balances for metrics: {bal_err}")
            logger.info(f"Calling updatePredictionAndAdjust with predictedTick: {predicted_tick}")
            try:
                tx_function = self.contract.functions.updatePredictionAndAdjust(predicted_tick)
                tx_params = {
                    'from': funding_account.address,
                    'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address),
                    'chainId': int(web3_utils.w3.net.version)
                }
                try:
                    pre_tx = tx_function.build_transaction({'from': funding_account.address, 'nonce': tx_params['nonce'], 'chainId': tx_params['chainId']})
                    estimated_gas = web3_utils.w3.eth.estimate_gas(pre_tx)
                    tx_params['gas'] = int(estimated_gas * 1.25)
                    logger.info(f"Estimated gas for adjustment: {estimated_gas}, using: {tx_params['gas']}")
                except Exception as est_err:
                    logger.warning(f"Gas estimation failed for adjustment: {est_err}. Using default 1,500,000")
                    tx_params['gas'] = 1500000

                final_tx_to_send = tx_function.build_transaction(tx_params)
                receipt = send_transaction(final_tx_to_send)

                self.metrics['tx_hash'] = receipt.transactionHash.hex() if receipt else None
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SENT"]

                if receipt and receipt.status == 1:
                    logger.info(f"Adjustment transaction successful (Status 1). Tx: {self.metrics['tx_hash']}")
                    self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                    if receipt.get('effectiveGasPrice'):
                        self.metrics['gas_cost_eth'] = float(Web3.from_wei(receipt.gasUsed * receipt.effectiveGasPrice, 'ether'))
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                    adjustment_call_success = True
                    try:
                        mint_logs = self.contract.events.LiquidityOperation().process_receipt(receipt, errors=logging.WARN)
                        for log in mint_logs:
                            if log.args.operationType == "MINT":
                                self.metrics['amount0_provided_to_mint'] = log.args.amount0
                                self.metrics['amount1_provided_to_mint'] = log.args.amount1
                            elif log.args.operationType == "REMOVE":
                                self.metrics['fees_collected_token0'] = log.args.amount0
                                self.metrics['fees_collected_token1'] = log.args.amount1
                        adj_metrics_logs = self.contract.events.PredictionAdjustmentMetrics().process_receipt(receipt, errors=logging.WARN)
                        if adj_metrics_logs:
                            self.metrics['finalTickLower_contract'] = adj_metrics_logs[0].args.finalTickLower
                            self.metrics['finalTickUpper_contract'] = adj_metrics_logs[0].args.finalTickUpper
                    except Exception as log_processing_error:
                        logger.error(f"Error processing logs for predictive: {log_processing_error}")
                elif receipt:
                    logger.error(f"Adjustment transaction reverted (Status 0). Tx: {self.metrics['tx_hash']}")
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                    self.metrics['error_message'] = "tx_reverted_onchain"
                    self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                    if receipt.get('effectiveGasPrice'):
                        self.metrics['gas_cost_eth'] = float(Web3.from_wei(receipt.gasUsed * receipt.effectiveGasPrice, 'ether'))
                    adjustment_call_success = False
                else:
                    logger.error("Adjustment transaction sending/receipt failed.")
                    if self.metrics['action_taken'] == self.ACTION_STATES["TX_SENT"]:
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                    if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for adjustment failed"
                    adjustment_call_success = False

            except Exception as tx_err:
                logger.exception(f"Error during adjustment transaction call/wait: {tx_err}")
                self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = f"TxError: {str(tx_err)}"
                self.save_metrics()
                return False

            self.update_pool_and_position_metrics(final_update=True)
            self.save_metrics()
            return adjustment_call_success
        except Exception as e:
            logger.exception(f"Error in adjust_position: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"Adjust position error: {str(e)}"
            self.save_metrics()
            return False

    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'range_width_multiplier_setting',
            'predictedPrice_api', 'predictedTick_calculated',
            'external_api_eth_price',
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_calculated', 'targetTickUpper_calculated',
            'initial_contract_balance_token0',
            'initial_contract_balance_token1',
            'finalTickLower_contract', 'finalTickUpper_contract', 'liquidity_contract',
            'amount0_provided_to_mint',
            'amount1_provided_to_mint',
            'fees_collected_token0',
            'fees_collected_token1',
            'gas_used', 'gas_cost_eth', 'error_message'
        ]
        try:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_exists = RESULTS_FILE.is_file()
            row_data = {col: self.metrics.get(col, "") for col in columns}

            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Predictive metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save predictive metrics: {e}")

# --- Main Function ---
def main():
    logger.info("="*50)
    logger.info("Starting Predictive Liquidity Manager Test on Fork")
    logger.info("="*50)

    if not web3_utils.init_web3():
        logger.critical("Web3 initialization failed. Exiting predictive test.")
        return
    
    if not web3_utils.w3 or not web3_utils.w3.is_connected():
        logger.critical("web3_utils.w3 instance not available or not connected after init. Exiting predictive test.")
        return

    predictive_address = None
    try:
        if not ADDRESS_FILE_PREDICTIVE.exists():
            logger.error(f"Predictive address file not found: {ADDRESS_FILE_PREDICTIVE}")
            raise FileNotFoundError(f"File not found: {ADDRESS_FILE_PREDICTIVE}")

        logger.info(f"Reading predictive address from: {ADDRESS_FILE_PREDICTIVE}")
        with open(ADDRESS_FILE_PREDICTIVE, 'r') as f:
            content = f.read()
            logger.debug(f"Predictive address file content: {content}")
            addresses_data = json.loads(content)
            predictive_address = addresses_data.get('address')
            if not predictive_address:
                logger.error(f"Key 'address' not found in {ADDRESS_FILE_PREDICTIVE}")
                raise ValueError(f"Key 'address' not found in {ADDRESS_FILE_PREDICTIVE}")
        logger.info(f"Loaded Predictive Manager Address: {predictive_address}")

        test = PredictiveTest(predictive_address)
        test.execute_test_steps()

    except FileNotFoundError as e:
        logger.error(f"Setup Error - Address file not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration Error - Problem reading address file or address key missing: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during predictive main execution:")
    finally:
        logger.info("="*50)
        logger.info("Predictive test run finished.")
        logger.info("="*50)

if __name__ == "__main__":
    main()