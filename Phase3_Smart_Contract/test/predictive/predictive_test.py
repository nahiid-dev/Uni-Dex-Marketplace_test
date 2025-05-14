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
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(0.02, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 20 * (10**6) # 20 USDC
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
            'predictedPrice_api': None, 'predictedTick_calculated': None,
            'actualPrice_pool': None, 'sqrtPriceX96_pool': 0, 'currentTick_pool': 0,
            'targetTickLower_calculated': 0, 'targetTickUpper_calculated': 0,
            'finalTickLower_contract': 0, 'finalTickUpper_contract': 0, 'liquidity_contract': 0,
            'gas_used': 0, 'gas_cost_eth': 0.0, 'error_message': ""
        }

    def setup(self) -> bool:
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
            return True
        except Exception as e:
            logger.exception(f"Predictive setup failed getting pool: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup pool error: {str(e)}"
            return False

    def get_predicted_price_from_api(self) -> float | None:
        try:
            logger.info(f"Querying LSTM API at {LSTM_API_URL}...")
            response = requests.get(LSTM_API_URL, timeout=15)
            response.raise_for_status()
            data = response.json()
            predicted_price_str = data.get('predicted_price')

            if predicted_price_str is None:
                predicted_price_str = data.get('price') or data.get('prediction')

            if predicted_price_str is None:
                logger.error(f"Predicted price key not found in API response. Data: {data}")
                raise ValueError("Predicted price not in API response")

            if isinstance(predicted_price_str, str):
                predicted_price_str = predicted_price_str.replace("USD", "").strip()

            predicted_price = float(predicted_price_str)
            logger.info(f"Received predicted ETH price from API: {predicted_price:.2f} USD")
            self.metrics['predictedPrice_api'] = predicted_price
            return predicted_price
        except requests.exceptions.Timeout:
            logger.error(f"Timeout when querying LSTM API at {LSTM_API_URL}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Timeout: {LSTM_API_URL}"
            return None
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error getting prediction from API {LSTM_API_URL}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Request Error: {str(e)}"
            return None
        except (ValueError, KeyError) as e:
            logger.exception(f"Error processing API response from {LSTM_API_URL}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Response Processing Error: {str(e)}"
            return None


    def calculate_tick_from_price(self, price: float) -> int | None:
        if self.token0_decimals is None or self.token1_decimals is None: # Check for None
            logger.error("Token decimals not available for tick calculation.")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Token decimals missing for tick calc"
            return None
        try:
            price_decimal = Decimal(str(price))
            
            # tick = floor( log_{sqrt(1.0001)}(sqrt_price_adjusted_for_decimals) )
            # sqrt_price_adjusted_for_decimals = sqrt(price_T1/T0 * 10^(decimals_T0 - decimals_T1))
            effective_sqrt_price_arg = price_decimal * (Decimal(10)**(self.token0_decimals - self.token1_decimals))
            if effective_sqrt_price_arg <= 0:
                logger.error(f"Argument for sqrt in tick calculation is non-positive: {effective_sqrt_price_arg}")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Invalid arg for sqrt in tick calc"
                return None
            
            effective_sqrt_price = effective_sqrt_price_arg.sqrt()
            
            if effective_sqrt_price <= 0: # Should not happen if arg > 0, but as a safeguard
                logger.error(f"Effective sqrt price for tick calculation is non-positive: {effective_sqrt_price}")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Invalid effective_sqrt_price for tick calc"
                return None

            # log_base_sqrt_1_0001_of_X  =  ln(X) / ln(sqrt(1.0001))
            tick = math.floor(math.log(float(effective_sqrt_price)) / math.log(math.sqrt(1.0001)))

            tick = max(MIN_TICK_CONST, min(MAX_TICK_CONST, tick))
            logger.info(f"Calculated tick {tick} from price {price:.2f}")
            self.metrics['predictedTick_calculated'] = tick
            return tick
        except Exception as e:
            logger.exception(f"Failed to calculate predicted tick from price {price}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Tick Calculation Error: {str(e)}"
            return None

    def update_pool_and_position_metrics(self, final_update=False):
        try:
            if self.pool_contract:
                slot0 = self.pool_contract.functions.slot0().call()
                sqrt_price_x96_pool, current_tick_pool = slot0[0], slot0[1]
                self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96_pool
                self.metrics['currentTick_pool'] = current_tick_pool
                self.metrics['actualPrice_pool'] = self._calculate_actual_price(sqrt_price_x96_pool)
            else:
                logger.warning("Pool contract not available for metrics update.")
                self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
                self.metrics['error_message'] = self.metrics.get('error_message',"") + ";Pool contract missing for metrics"

            position_info = self.get_position_info()
            if position_info:
                if final_update:
                    self.metrics['finalTickLower_contract'] = position_info.get('tickLower', 0)
                    self.metrics['finalTickUpper_contract'] = position_info.get('tickUpper', 0)
                    self.metrics['liquidity_contract'] = position_info.get('liquidity', 0)
            else:
                logger.warning("Could not get position info from contract for metrics.")

        except Exception as e:
            logger.exception(f"Error updating pool/position metrics: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["METRICS_UPDATE_FAILED"]
            if not self.metrics.get('error_message'): self.metrics['error_message'] = f"Metrics Update Error: {str(e)}"


    def fund_contract_if_needed(self, min_weth=MIN_WETH_TO_FUND_CONTRACT, min_usdc=MIN_USDC_TO_FUND_CONTRACT) -> bool:
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            if not web3_utils.init_web3():
                logger.error("Web3 connection failed in fund_contract_if_needed.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "W3 init fail in fund_contract"
                return False
        
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("web3_utils.w3 is still not available after init attempt in fund_contract_if_needed.")
            self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
            self.metrics['error_message'] = "W3 unavailable post init in fund_contract"
            return False

        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY environment variable not set for funding.")
            self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for funding"
            return False
        
        try:
            account = Account.from_key(private_key_env)
        except Exception as e:
            logger.error(f"Failed to create account from PRIVATE_KEY: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
            self.metrics['error_message'] = f"Bad PRIVATE_KEY: {e}"
            return False
            
        contract_addr_checksum = Web3.to_checksum_address(self.contract_address)

        try:
            token0_is_usdc = self.token0_decimals == 6
            weth_token_addr = self.token1 if token0_is_usdc else self.token0
            usdc_token_addr = self.token0 if token0_is_usdc else self.token1
            usdc_decimals_val = self.token0_decimals if token0_is_usdc else self.token1_decimals


            weth_contract = get_contract(weth_token_addr, "IERC20")
            usdc_contract = get_contract(usdc_token_addr, "IERC20")

            contract_weth_bal = weth_contract.functions.balanceOf(contract_addr_checksum).call()
            contract_usdc_bal = usdc_contract.functions.balanceOf(contract_addr_checksum).call()
            logger.info(f"Contract balances before funding check: WETH={Web3.from_wei(contract_weth_bal, 'ether')}, USDC={contract_usdc_bal / (10**usdc_decimals_val):.6f}")


            fund_weth = contract_weth_bal < min_weth
            fund_usdc = contract_usdc_bal < min_usdc

            if not fund_weth and not fund_usdc:
                logger.info("Contract already has sufficient WETH and USDC.")
                return True

            logger.info("Attempting to fund contract...")
            current_nonce = web3_utils.w3.eth.get_transaction_count(account.address)

            if fund_weth:
                needed_weth = min_weth - contract_weth_bal
                logger.info(f"Contract needs {Web3.from_wei(needed_weth, 'ether')} WETH.")
                deployer_weth_bal = weth_contract.functions.balanceOf(account.address).call()

                if deployer_weth_bal < needed_weth:
                    logger.warning(f"Deployer has insufficient WETH ({Web3.from_wei(deployer_weth_bal, 'ether')}). Attempting to wrap ETH...")
                    eth_needed_for_wrap = needed_weth - deployer_weth_bal + Web3.to_wei(0.001, 'ether')
                    if web3_utils.wrap_eth_to_weth(eth_needed_for_wrap):
                        time.sleep(3) 
                        deployer_weth_bal = weth_contract.functions.balanceOf(account.address).call()
                    else:
                        logger.error("Failed to wrap ETH for WETH funding.")
                        self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                        self.metrics['error_message'] = "ETH wrapping failed"
                        return False

                if deployer_weth_bal >= needed_weth:
                    logger.info(f"Transferring {Web3.from_wei(needed_weth, 'ether')} WETH from deployer to contract {contract_addr_checksum}...")
                    tx_transfer_params = {'from': account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
                    built_tx = weth_contract.functions.transfer(contract_addr_checksum, needed_weth).build_transaction(tx_transfer_params)
                    receipt = send_transaction(built_tx) 

                    if receipt and receipt.status == 1:
                        logger.info(f"WETH transfer successful. Tx: {receipt.transactionHash.hex()}")
                        current_nonce += 1
                        time.sleep(3) 
                    else:
                        logger.error(f"WETH transfer to contract failed. Receipt: {receipt}")
                        self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                        self.metrics['error_message'] = "WETH transfer tx failed"
                        return False
                else:
                    logger.error(f"Deployer still has insufficient WETH ({Web3.from_wei(deployer_weth_bal, 'ether')}) after wrap attempt.")
                    self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                    self.metrics['error_message'] = "Insufficient WETH post-wrap"
                    return False
            
            if fund_usdc:
                needed_usdc = min_usdc - contract_usdc_bal
                logger.info(f"Contract needs {needed_usdc / (10**usdc_decimals_val):.6f} USDC.")
                deployer_usdc_bal = usdc_contract.functions.balanceOf(account.address).call()

                if deployer_usdc_bal >= needed_usdc:
                    logger.info(f"Transferring {needed_usdc / (10**usdc_decimals_val):.6f} USDC from deployer to contract {contract_addr_checksum}...")
                    tx_transfer_params = {'from': account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
                    built_tx = usdc_contract.functions.transfer(contract_addr_checksum, needed_usdc).build_transaction(tx_transfer_params)
                    receipt = send_transaction(built_tx)

                    if receipt and receipt.status == 1:
                        logger.info(f"USDC transfer successful. Tx: {receipt.transactionHash.hex()}")
                        time.sleep(3)
                    else:
                        logger.error(f"USDC transfer to contract failed. Receipt: {receipt}")
                        self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                        self.metrics['error_message'] = "USDC transfer tx failed"
                        return False
                else:
                    logger.error(f"Deployer has insufficient USDC ({deployer_usdc_bal / (10**usdc_decimals_val):.6f}).")
                    self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                    self.metrics['error_message'] = "Insufficient USDC"
                    return False

            contract_weth_bal_final = weth_contract.functions.balanceOf(contract_addr_checksum).call()
            contract_usdc_bal_final = usdc_contract.functions.balanceOf(contract_addr_checksum).call()
            logger.info(f"Balances after funding attempt: WETH={Web3.from_wei(contract_weth_bal_final, 'ether')}, USDC={contract_usdc_bal_final / (10**usdc_decimals_val):.6f}")
            
            if contract_weth_bal_final < min_weth or contract_usdc_bal_final < min_usdc:
                logger.error("Contract balances still below minimum after funding attempt.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Balances low post-funding"
                return False
            return True

        except Exception as e:
            logger.exception(f"Error during fund_contract_if_needed: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
            self.metrics['error_message'] = f"Fund contract exception: {str(e)}"
            return False

    def adjust_position(self) -> bool:
        self.metrics = self._reset_metrics()
        adjustment_call_success = False

        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("Web3 not connected at start of adjust_position.")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"] 
                self.metrics['error_message'] = "W3 unavailable in adjust_position"
                self.save_metrics()
                return False

            predicted_price = self.get_predicted_price_from_api()
            if predicted_price is None:
                self.save_metrics()
                return False

            predicted_tick = self.calculate_tick_from_price(predicted_price)
            if predicted_tick is None:
                self.save_metrics()
                return False

            self.update_pool_and_position_metrics(final_update=False)

            if not self.fund_contract_if_needed():
                logger.error("Funding contract failed. Cannot proceed with adjustment.")
                self.save_metrics()
                return False

            logger.info(f"Calling updatePredictionAndAdjust with predictedTick: {predicted_tick}")
            private_key_env = os.getenv('PRIVATE_KEY')
            if not private_key_env:
                logger.error("PRIVATE_KEY not found for adjust_position.")
                self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx"
                self.save_metrics()
                return False
            
            try:
                account = Account.from_key(private_key_env)
            except Exception as e:
                logger.error(f"Failed to load account from PRIVATE_KEY for adjustment: {e}")
                self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = f"Bad PRIVATE_KEY for adjustment: {e}"
                self.save_metrics()
                return False


            current_nonce = web3_utils.w3.eth.get_transaction_count(account.address)

            try:
                tx_function = self.contract.functions.updatePredictionAndAdjust(predicted_tick)
                tx_params = {
                    'from': account.address,
                    'nonce': current_nonce,
                    'chainId': int(web3_utils.w3.net.version)
                }
                
                try:
                    pre_tx = tx_function.build_transaction({'from': account.address, 'nonce': current_nonce, 'chainId': tx_params['chainId']})
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
            logger.exception("Unexpected error in adjust_position:")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            if not self.metrics.get('error_message'): self.metrics['error_message'] = str(e)
            self.save_metrics()
            return False

    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'predictedPrice_api', 'predictedTick_calculated',
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_calculated', 'targetTickUpper_calculated',
            'finalTickLower_contract', 'finalTickUpper_contract', 'liquidity_contract',
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