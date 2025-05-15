import os
import sys
import json
import time
import logging
import csv
import math
from datetime import datetime
from pathlib import Path
from web3 import Web3
from eth_account import Account
from decimal import Decimal, getcontext


# --- Adjust path imports ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent # test/baseline/baseline_test.py -> Phase3_Smart_Contract
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
    print(f"ERROR importing from test.utils in baseline_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6)  # 1000 USDC
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272
MAX_TICK_CONST = 887272


# --- Setup Logging ---
if not logging.getLogger('baseline_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("baseline_test.log"), # Log to a file in execution directory
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger('baseline_test')

# --- Define path for addresses and results ---
ADDRESS_FILE_BASELINE = project_root / 'baselineMinimal_address.json'
RESULTS_FILE = project_root / 'position_results_baseline.csv'

logger.info(f"Project Root for Baseline Test (from baseline_test.py): {project_root}")
logger.info(f"Baseline Address File: {ADDRESS_FILE_BASELINE}")
logger.info(f"Baseline Results File: {RESULTS_FILE}")


class BaselineTest(LiquidityTestBase):
    """Test implementation for BaselineMinimal with token funding."""

    def __init__(self, contract_address: str):
        super().__init__(contract_address, "BaselineMinimal")
        self.ACTION_STATES = {
            "INIT": "init", "SETUP_FAILED": "setup_failed",
            "POOL_READ_FAILED": "pool_read_failed", "CALCULATION_FAILED": "calculation_failed",
            "SKIPPED_PROXIMITY": "skipped_proximity", "FUNDING_FAILED": "funding_failed",
            "TX_SENT": "tx_sent", "TX_SUCCESS_ADJUSTED": "tx_success_adjusted",
            "TX_SUCCESS_SKIPPED_ONCHAIN": "tx_success_skipped_onchain",
            "TX_REVERTED": "tx_reverted", "TX_WAIT_FAILED": "tx_wait_failed",
            "METRICS_UPDATE_FAILED": "metrics_update_failed",
            "UNEXPECTED_ERROR": "unexpected_error"
        }
        self.factory_contract = None
        self.tick_spacing = None
        self.pool_address = None
        self.pool_contract = None
        self.metrics = self._reset_metrics()
        # Token contracts for estimation
        self.token0_contract = None
        self.token1_contract = None

    def _reset_metrics(self):
        return {
            'timestamp': None, 'contract_type': 'Baseline',
            'action_taken': self.ACTION_STATES["INIT"], 'tx_hash': None,
            'range_width_multiplier_setting': None,
            'external_api_eth_price': None,
            'actualPrice_pool': None, 'sqrtPriceX96_pool': None, 'currentTick_pool': None,
            'targetTickLower_offchain': None, 'targetTickUpper_offchain': None,
            'initial_contract_balance_token0': None,
            'initial_contract_balance_token1': None,
            'currentTickLower_contract': None, 'currentTickUpper_contract': None, 'currentLiquidity_contract': None,
            'finalTickLower_contract': None, 'finalTickUpper_contract': None, 'finalLiquidity_contract': None,
            'amount0_provided_to_mint': None,
            'amount1_provided_to_mint': None,
            'fees_collected_token0': None,
            'fees_collected_token1': None,
            'gas_used': None, 'gas_cost_eth': None, 'error_message': ""
        }

    def setup(self, desired_range_width_multiplier: int) -> bool:
        if not super().setup():
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("web3_utils.w3 not available in BaselineTest setup after base.setup()")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup"
                return False
            factory_address = self.contract.functions.factory().call()
            self.factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            fee = self.contract.functions.fee().call()
            self.pool_address = self.factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Baseline pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            logger.info(f"Baseline Pool contract initialized at {self.pool_address}")
            self.tick_spacing = self.pool_contract.functions.tickSpacing().call()
            if not self.tick_spacing or self.tick_spacing <= 0:
                logger.error(f"Invalid tickSpacing read from pool: {self.tick_spacing}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Invalid tickSpacing: {self.tick_spacing}"
                return False
            logger.info(f"Baseline Tick spacing from pool: {self.tick_spacing}")
            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Baseline contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier
            tx_params = {
                'from': Account.from_key(os.getenv('PRIVATE_KEY')).address,
            }
            tx_set_rwm = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).build_transaction(tx_params)
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm)
            if not receipt_rwm or receipt_rwm.status == 0:
                logger.error(f"Failed to set rangeWidthMultiplier for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'}")
                self.metrics['error_message'] += ";Failed to set RWM for Baseline"
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed getting pool/tickSpacing: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup pool/tick error: {str(e)}"
            return False

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        self.metrics = self._reset_metrics()
        try:
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.")
        adjustment_call_success = False
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position.")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx"
            self.save_metrics()
            return False
        try:
            _, current_tick = self.get_pool_state()
            if current_tick is None:
                self.save_metrics()
                return False
            pos_info_before = self.get_position_info()
            if pos_info_before:
                self.metrics['currentTickLower_contract'] = pos_info_before.get('tickLower')
                self.metrics['currentTickUpper_contract'] = pos_info_before.get('tickUpper')
                self.metrics['currentLiquidity_contract'] = pos_info_before.get('liquidity', 0)
            target_lower_tick, target_upper_tick = self.calculate_target_ticks_offchain(current_tick)
            if target_lower_tick is None or target_upper_tick is None:
                self.save_metrics()
                return False
            logger.info("Ensuring precise token balances for Baseline contract...")
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
                logger.error("Precise funding for Baseline contract failed.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Precise contract funding failed"
                self.save_metrics()
                return False
            try:
                token0_contract = web3_utils.get_contract(self.token0, "IERC20")
                token1_contract = web3_utils.get_contract(self.token1, "IERC20")
                self.metrics['initial_contract_balance_token0'] = token0_contract.functions.balanceOf(self.contract_address).call()
                self.metrics['initial_contract_balance_token1'] = token1_contract.functions.balanceOf(self.contract_address).call()
            except Exception as bal_err:
                logger.warning(f"Could not read initial contract balances for baseline metrics: {bal_err}")
            logger.info(f"Calling adjustLiquidityWithCurrentPrice...")
            final_pos_info = self.get_position_info()
            if final_pos_info:
                self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower')
                self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper')
                self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity', 0)
                if final_pos_info.get('active') and self.metrics['finalLiquidity_contract'] == 0:
                    logger.warning("Baseline: Active position reported with 0 liquidity after adjustment.")
            else:
                logger.warning("Baseline: Could not get final position info after adjustment.")
            self.save_metrics()
            return adjustment_call_success
        except Exception as e:
            logger.exception("Unexpected error in baseline adjust_position:")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            if not self.metrics.get('error_message'): self.metrics['error_message'] = str(e)
            self.save_metrics()
            return False

    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'range_width_multiplier_setting',
            'external_api_eth_price',
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_offchain', 'targetTickUpper_offchain',
            'initial_contract_balance_token0',
            'initial_contract_balance_token1',
            'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
            'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract',
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
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save baseline metrics: {e}")

# --- Main Function ---
def main():
    logger.info("="*50)
    logger.info("Starting Baseline Minimal Liquidity Manager Test on Fork")
    logger.info("="*50)

    if not web3_utils.init_web3():
        logger.critical("Web3 initialization failed. Exiting baseline test.")
        sys.exit(1) 

    if not web3_utils.w3 or not web3_utils.w3.is_connected():
        logger.critical("web3_utils.w3 instance not available or not connected after init. Exiting baseline test.")
        sys.exit(1)


    baseline_address_val = None 
    try:
        if not ADDRESS_FILE_BASELINE.exists():
            logger.error(f"Baseline address file not found: {ADDRESS_FILE_BASELINE}")
            raise FileNotFoundError(f"File not found: {ADDRESS_FILE_BASELINE}")

        logger.info(f"Reading baseline address from: {ADDRESS_FILE_BASELINE}")
        with open(ADDRESS_FILE_BASELINE, 'r') as f:
            content = f.read()
            logger.debug(f"Baseline address file content: {content}")
            addresses_data = json.loads(content)
            baseline_address_val = addresses_data.get('address')
            if not baseline_address_val:
                logger.error(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
                raise ValueError(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
        logger.info(f"Loaded Baseline Minimal Address: {baseline_address_val}")

        test = BaselineTest(baseline_address_val)
        test.execute_test_steps()

    except FileNotFoundError as e:
        logger.error(f"Setup Error - Address file not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration Error - Problem reading address file or address key missing: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during baseline main execution:")
    finally:
        logger.info("="*50)
        logger.info("Baseline test run finished.")
        logger.info("="*50)

if __name__ == "__main__":
    main()