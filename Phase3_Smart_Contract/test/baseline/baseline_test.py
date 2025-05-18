# -*- coding: utf-8 -*-
# filepath: d:\Uni-Dex-Marketplace_test\Phase3_Smart_Contract\test\baseline\baseline_test.py
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
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
# Required values for contract funding, using new values
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6) # 1000 USDC
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272 # Uniswap V3 min tick
MAX_TICK_CONST = 887272 # Uniswap V3 max tick


# --- Setup Logging ---
if not logging.getLogger('baseline_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("baseline_test.log", mode='a'),
            logging.StreamHandler(sys.stdout)
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

    def _reset_metrics(self):
        """Initialize or reset all metrics to their default values."""
        return {
            'timestamp': None,
            'contract_type': 'Baseline',
            'action_taken': "init",  # Using string directly since ACTION_STATES isn't set yet
            'tx_hash': None,
            'range_width_multiplier_setting': None,
            'external_api_eth_price': None,
            'actualPrice_pool': None,
            'sqrtPriceX96_pool': None,
            'currentTick_pool': None,
            'targetTickLower_offchain': None,
            'targetTickUpper_offchain': None,
            'initial_contract_balance_token0': None,
            'initial_contract_balance_token1': None,
            'currentTickLower_contract': None,
            'currentTickUpper_contract': None,
            'currentLiquidity_contract': None,
            'finalTickLower_contract': None,
            'finalTickUpper_contract': None,
            'finalLiquidity_contract': None,
            'amount0_provided_to_mint': None,
            'amount1_provided_to_mint': None,
            'fees_collected_token0': None,
            'fees_collected_token1': None,
            'gas_used': None,
            'gas_cost_eth': None,
            'error_message': ""
        }

    def __init__(self, contract_address: str):
        """Initialize the BaselineTest instance."""
        super().__init__(contract_address, "BaselineMinimal")      
        self.ACTION_STATES = {
            "INIT": "init",
            "SETUP_FAILED": "setup_failed",
            "POOL_READ_FAILED": "pool_read_failed",
            "CALCULATION_FAILED": "calculation_failed",
            "SKIPPED_PROXIMITY": "skipped_proximity",
            "FUNDING_FAILED": "funding_failed",
            "TX_SENT": "tx_sent",
            "TX_SUCCESS_ADJUSTED": "tx_success_adjusted",
            "TX_SUCCESS_SKIPPED_ONCHAIN": "tx_success_skipped_onchain",
            "TX_REVERTED": "tx_reverted",
            "TX_WAIT_FAILED": "tx_wait_failed",
            "METRICS_UPDATE_FAILED": "metrics_update_failed",
            "UNEXPECTED_ERROR": "unexpected_error"
        }
        self.factory_contract = None
        self.tick_spacing = None
        self.pool_address = None
        self.pool_contract = None
        self.token0_contract_instance = None
        self.token1_contract_instance = None
        self.metrics = self._reset_metrics()
    def _estimate_liquidity(self, tick_lower: int, tick_upper: int) -> int: # According to old version [202-206]
        """Estimate liquidity based on token balances and tick range (simple approximation)."""
        # This method is a simple estimate and should not be used as an accurate source of liquidity.
        # The contract itself should manage and report liquidity.
        try:
            if not self.token0_contract_instance and self.token0:
                self.token0_contract_instance = get_contract(self.token0, "IERC20")
            if not self.token1_contract_instance and self.token1:
                self.token1_contract_instance = get_contract(self.token1, "IERC20")

            if not self.token0_contract_instance or not self.token1_contract_instance or not self.pool_contract:
                logger.warning("Cannot estimate liquidity: token contracts or pool contract not initialized.")
                return 0

            token0_bal = self.token0_contract_instance.functions.balanceOf(self.contract_address).call()
            token1_bal = self.token1_contract_instance.functions.balanceOf(self.contract_address).call()
            
            slot0 = self.pool_contract.functions.slot0().call()
            current_tick = slot0[1]
              # Simple estimation logic (may not be accurate)
            if current_tick < tick_lower: # Price below range, all liquidity should be token0
                # This estimation is very simplistic for Uniswap V3 liquidity.
                # Actual liquidity depends on more complex calculations.
                return int(token0_bal) # This is not correct, liquidity has a different unit.
            elif current_tick >= tick_upper: # Price above range, all liquidity should be token1
                return int(token1_bal) # This is also not correct.
            else: # Price within range
                 # This is just a simple average of balances and not real liquidity.
                return int((token0_bal + token1_bal) / 2)
        except Exception as e:
            logger.error(f"Error estimating liquidity: {e}")
            return 0

    def get_position_info(self) -> dict | None: # According to old version [198-202]
        """Get current position details from the contract."""
        try:
            # getCurrentPosition must be defined in BaselineMinimal.sol
            # and return information like (tokenId, active, tickLower, tickUpper, liquidity)
            # or if you're using NonfungiblePositionManager, you should have tokenId
            # and read from positions(tokenId) in NPM.
            # We assume getCurrentPosition exists in your contract.
            position_data = self.contract.functions.getCurrentPosition().call()
            
            # Example: If getCurrentPosition returns a tuple:
            # token_id, active, tick_lower, tick_upper, liquidity_contract = position_data            # Or if it returns a dictionary:
            # active = position_data.get('active')
            # tick_lower = position_data.get('tickLower')
            # ...
              # This section must be adapted to match your getCurrentPosition output structure
            # Assuming the output is a list/tuple with a specific order:
            if len(position_data) >= 5:
                token_id, active, tick_lower, tick_upper, liquidity_contract = position_data[0], position_data[1], position_data[2], position_data[3], position_data[4]
                
                # If contract returns 0 liquidity but position is active, don't attempt estimation
                # unless you have a reliable mechanism for it. The contract's value is preferred.
                # if active and liquidity_contract == 0:
                # logger.warning("Contract reports active position with 0 liquidity. Using reported 0.")
                
                return {
                    'tokenId': token_id, 
                    'active': active,
                    'tickLower': tick_lower,
                    'tickUpper': tick_upper,
                    'liquidity': liquidity_contract
                }
            else:
                logger.error(f"Unexpected data structure from getCurrentPosition(): {position_data}")
                return None
        except Exception as e:
            logger.error(f"Error getting position info from contract: {e}")
            self.metrics['error_message'] += f";PosInfoError: {str(e)}"
            return None
            
    def setup(self, desired_range_width_multiplier: int) -> bool: # Parameter added according to new version
        if not super().setup(desired_range_width_multiplier):
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
            if not self.factory_contract: return False

            fee = self.contract.functions.fee().call()
            self.pool_address = self.factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Baseline pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
            
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            if not self.pool_contract: return False
            logger.info(f"Baseline Pool contract initialized at {self.pool_address}")
            
            self.tick_spacing = self.pool_contract.functions.tickSpacing().call()
            if not self.tick_spacing or self.tick_spacing <= 0: # tickSpacing must be positive
                logger.error(f"Invalid tickSpacing read from pool: {self.tick_spacing}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Invalid tickSpacing: {self.tick_spacing}"
                return False
            logger.info(f"Baseline Tick spacing from pool: {self.tick_spacing}")            # Set rangeWidthMultiplier according to new version
            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Baseline contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier
            
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                logger.error("PRIVATE_KEY not set for setting rangeWidthMultiplier (Baseline).")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "PRIVATE_KEY missing for RWM setup (Baseline)"
                return False
            
            tx_account = Account.from_key(private_key)
            tx_params_rwm = {
                'from': tx_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(tx_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }
            try:
                gas_estimate_rwm = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).estimate_gas({'from': tx_account.address})
                tx_params_rwm['gas'] = int(gas_estimate_rwm * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for setRangeWidthMultiplier (Baseline): {e}. Using default 200000.")
                tx_params_rwm['gas'] = 200000

            tx_set_rwm_build = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier)
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm_build.build_transaction(tx_params_rwm), private_key)
            
            if not receipt_rwm or receipt_rwm.status == 0:
                tx_hash_rwm_str = receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'
                logger.error(f"Failed to set rangeWidthMultiplier for Baseline contract. TxHash: {tx_hash_rwm_str}")
                self.metrics['error_message'] += f";Failed to set RWM for Baseline (tx: {tx_hash_rwm_str})"
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed: {e}") # Changed log message
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}" # Changed log message
            return False

    def get_pool_state(self) -> tuple[int | None, int | None]:
        if not self.pool_contract:
            logger.error("Pool contract not initialized for get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "Pool contract missing for get_pool_state"
            return None, None
        if not web3_utils.w3 or not web3_utils.w3.is_connected(): # Check web3 connection
            logger.error("Web3 not connected in get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "W3 not connected for get_pool_state"
            return None, None
        try:
            slot0 = self.pool_contract.functions.slot0().call()
            sqrt_price_x96, tick = slot0[0], slot0[1]
              # Update metrics here
            self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96
            self.metrics['currentTick_pool'] = tick
            # _calculate_actual_price must be defined in LiquidityTestBase
            self.metrics['actualPrice_pool'] = self._calculate_actual_price(sqrt_price_x96) 
            
            logger.info(f"Pool state read: Tick={tick}, SqrtPriceX96={sqrt_price_x96}, ActualPrice={self.metrics['actualPrice_pool']:.4f}")
            return sqrt_price_x96, tick
        except Exception as e:
            logger.exception(f"Failed to get pool state (Baseline): {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = f"Pool state read error: {str(e)}"
            return None, None
    
    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]:
        """
        Calculate target tick range centered around current_tick.
        For WETH/USDC pool where USDC is token0 and WETH is token1:
        - When ETH price is high (e.g. $2000), current_tick will be negative
        - rangeWidthMultiplier * tickSpacing determines total width of range
        - Range should be approximately +/-5% in price terms
        """
        if self.tick_spacing is None or current_tick is None:
            logger.error("Tick spacing or current_tick not available for target tick calculation (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Missing data for target tick calc (Baseline)"
            return None, None
        try:
            # Get the width multiplier from contract or metrics
            width_multiplier = self.metrics.get('range_width_multiplier_setting')
            if width_multiplier is None or width_multiplier <= 0:
                width_multiplier = self.contract.functions.rangeWidthMultiplier().call()
            
            if width_multiplier is None or width_multiplier <= 0:
                logger.error(f"Invalid rangeWidthMultiplier ({width_multiplier}) for tick calculation.")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = f"Invalid RWM ({width_multiplier}) for tick calc"
                return None, None

            # Calculate half width to achieve approximately ±5% range
            # For 5% range: log_1.0001(1.05) ≈ 487 ticks
            target_half_tick_width = 487

            # Scale the width by the multiplier and ensure it's a multiple of tick_spacing
            scaled_half_width = (target_half_tick_width * width_multiplier) // 100
            tick_spacing_multiple = math.ceil(scaled_half_width / self.tick_spacing)
            half_total_tick_width = tick_spacing_multiple * self.tick_spacing

            # Ensure minimum width of one tick spacing
            if half_total_tick_width < self.tick_spacing:
                logger.warning(f"Calculated half width {half_total_tick_width} is less than tick spacing {self.tick_spacing}. Using minimum width.")
                half_total_tick_width = self.tick_spacing

            # Calculate target ticks ensuring they align with tick spacing
            target_lower_tick = math.floor(current_tick / self.tick_spacing) * self.tick_spacing - half_total_tick_width
            target_upper_tick = math.ceil(current_tick / self.tick_spacing) * self.tick_spacing + half_total_tick_width

            # Ensure ticks are within valid range
            target_lower_tick = max(MIN_TICK_CONST, target_lower_tick)
            target_upper_tick = min(MAX_TICK_CONST, target_upper_tick)

            # Ensure lower < upper
            if target_lower_tick >= target_upper_tick:
                logger.error(f"Invalid tick range calculated: [{target_lower_tick}, {target_upper_tick}]")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Generated invalid tick range L >= U"
                return None, None

            logger.info("Target ticks calculation details:")
            logger.info(f"  Current tick: {current_tick}")
            logger.info(f"  Range width multiplier: {width_multiplier}")
            logger.info(f"  Tick spacing: {self.tick_spacing}")
            logger.info(f"  Half width in ticks: {half_total_tick_width}")
            logger.info(f"  Target range: [{target_lower_tick}, {target_upper_tick}]")

            # Update metrics and return
            self.metrics['targetTickLower_offchain'] = target_lower_tick
            self.metrics['targetTickUpper_offchain'] = target_upper_tick
            return target_lower_tick, target_upper_tick
            
        except Exception as e:
            logger.exception(f"Error calculating target ticks off-chain (Baseline): {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Target tick calc error (Baseline): {str(e)}"
            return None, None    # ---- End of restored methods ----

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool: # Parameters according to new version        self.metrics = self._reset_metrics() # Reset metrics
        try:
            # Get current ETH price from API
            self.get_current_eth_price()
            
            # Read current range_width_multiplier from contract
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.")

        adjustment_call_success = False
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx (Baseline)"
            self.save_metrics()
            return False
        
        funding_account = Account.from_key(private_key_env)

        try:            # 1. Get current pool state
            _, current_tick = self.get_pool_state() # Call restored method
            if current_tick is None:
                # self.metrics already updated in get_pool_state
                self.save_metrics()
                return False            # 2. Get current position information from contract
            pos_info_before = self.get_position_info() # Call restored method
            current_pos_active = False
            if pos_info_before:
                current_pos_active = pos_info_before.get('active', False)
                self.metrics['currentTickLower_contract'] = pos_info_before.get('tickLower')
                self.metrics['currentTickUpper_contract'] = pos_info_before.get('tickUpper')
                self.metrics['currentLiquidity_contract'] = pos_info_before.get('liquidity', 0)
            else: # If position information could not be read
                logger.warning("Could not get current position info for Baseline. Assuming no active position.")            # 3. Calculate target ticks off-chain
            target_lower_tick, target_upper_tick = self.calculate_target_ticks_offchain(current_tick) # Call method
            if target_lower_tick is None or target_upper_tick is None:
                self.save_metrics()
                return False
            
            # 4. Proximity check - (Logic from old version)
            if current_pos_active and \
               self.metrics['currentTickLower_contract'] is not None and \
               self.metrics['currentTickUpper_contract'] is not None and \
               self.tick_spacing is not None:
                # Define a threshold for proximity, e.g., the tickSpacing itself
                TICK_PROXIMITY_THRESHOLD = self.tick_spacing 
                
                is_lower_close = abs(target_lower_tick - self.metrics['currentTickLower_contract']) <= TICK_PROXIMITY_THRESHOLD
                is_upper_close = abs(target_upper_tick - self.metrics['currentTickUpper_contract']) <= TICK_PROXIMITY_THRESHOLD

                if is_lower_close and is_upper_close:
                    logger.info(f"Baseline Off-chain proximity check: Target ticks ({target_lower_tick}, {target_upper_tick}) are close to current on-chain ({self.metrics['currentTickLower_contract']}, {self.metrics['currentTickUpper_contract']}). Skipping adjustment call.")
                    self.metrics['action_taken'] = self.ACTION_STATES["SKIPPED_PROXIMITY"]                    # Fill final values with current values since no change occurred
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return True # Operation was successful (because no change was needed)
              # 5. Precise contract funding (according to new version)
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
                self.metrics['error_message'] = "Precise contract funding failed (Baseline)"                # Fill final values with current values (before failed funding)
                self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                self.save_metrics()
                return False            # 6. Read initial contract balance (after funding and before main transaction)
            try:
                token0_c = get_contract(self.token0, "IERC20")
                token1_c = get_contract(self.token1, "IERC20")
                if token0_c and token1_c:
                    self.metrics['initial_contract_balance_token0'] = token0_c.functions.balanceOf(self.contract_address).call()
                    self.metrics['initial_contract_balance_token1'] = token1_c.functions.balanceOf(self.contract_address).call()
            except Exception as bal_err:
                logger.warning(f"Could not read initial contract balances for baseline metrics: {bal_err}")

            # 7. Call the adjustLiquidityWithCurrentPrice function in the smart contract

            logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract...")
            
            tx_function_call_base = self.contract.functions.adjustLiquidityWithCurrentPrice()
            tx_params_base = {
                'from': funding_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }           
            try: # Gas estimation
                gas_estimate_base = tx_function_call_base.estimate_gas({'from': funding_account.address})
                tx_params_base['gas'] = int(gas_estimate_base * 1.25)
                logger.info(f"Estimated gas for 'adjustLiquidityWithCurrentPrice': {gas_estimate_base}, using: {tx_params_base['gas']}")
            except Exception as est_err_base:
                logger.warning(f"Gas estimation failed for 'adjustLiquidityWithCurrentPrice': {est_err_base}. Using default 1,500,000")
                tx_params_base['gas'] = 1500000
            
            built_tx_base = tx_function_call_base.build_transaction(tx_params_base)
            receipt_base = web3_utils.send_transaction(built_tx_base, private_key_env)

            self.metrics['tx_hash'] = receipt_base.transactionHash.hex() if receipt_base else None
            self.metrics['action_taken'] = self.ACTION_STATES["TX_SENT"]

            if receipt_base and receipt_base.status == 1:
                logger.info(f"Baseline adjustment transaction successful (Status 1). Tx: {self.metrics['tx_hash']}. Processing events...")
                self.metrics['gas_used'] = receipt_base.get('gasUsed', 0)
                eff_gas_price_base = receipt_base.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base, 'ether'))
                adjustment_call_success = True                # Process events for Baseline
                # BaselineMinimal.sol may have different events.
                # We assume it has PositionMinted and PositionRemoved/FeesCollected events
                try:
                    # For MINT (if adjustLiquidityWithCurrentPrice creates a new position)
                    mint_logs_base = self.contract.events.PositionMinted().process_receipt(receipt_base, errors=logging.WARN)
                    for log_m_b in mint_logs_base:
                        self.metrics['amount0_provided_to_mint'] = log_m_b.args.amount0Actual
                        self.metrics['amount1_provided_to_mint'] = log_m_b.args.amount1Actual
                        # Final tick and liquidity values are read from this event or get_position_info
                        self.metrics['finalTickLower_contract'] = log_m_b.args.tickLower
                        self.metrics['finalTickUpper_contract'] = log_m_b.args.tickUpper
                        self.metrics['finalLiquidity_contract'] = log_m_b.args.liquidity
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"] # If minted, it means position was adjusted                    # For REMOVE/COLLECT (if previous position is removed and fees are collected)
                    # Match the event name with your contract
                    # remove_logs_base = self.contract.events.PositionRemovedAndFeesCollected().process_receipt(receipt_base, errors=logging.WARN)
                    # for log_r_b in remove_logs_base:
                    #     self.metrics['fees_collected_token0'] = log_r_b.args.fees0
                    #     self.metrics['fees_collected_token1'] = log_r_b.args.fees1
                    
                    # If contract emits a general status event like BaselineAdjustmentMetrics in old version
                    # This section of old code [263-268] might be useful
                    # event_name_base = "BaselineAdjustmentMetrics" # or a similar name
                    # if hasattr(self.contract.events, event_name_base):
                    #     logs_adj_base = self.contract.events[event_name_base]().process_receipt(receipt_base, errors=logging.WARN)
                    #     if logs_adj_base and len(logs_adj_base) > 0:
                    #         adjusted_onchain = logs_adj_base[0]['args'].get('adjusted', False) # Assuming 'adjusted' field exists                    #         self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"] if adjusted_onchain else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"]
                    #         # Final tick values and liquidity from event
                    #         self.metrics['finalTickLower_contract'] = logs_adj_base[0]['args'].get('finalTickLower', self.metrics['finalTickLower_contract'])
                    #         self.metrics['finalTickUpper_contract'] = logs_adj_base[0]['args'].get('finalTickUpper', self.metrics['finalTickUpper_contract'])
                    #         self.metrics['finalLiquidity_contract'] = logs_adj_base[0]['args'].get('finalLiquidity', self.metrics['finalLiquidity_contract'])
                    #     else: # If event was not found but transaction was successful
                    #         self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                    # else: # If contract doesn't have this event
                    #     self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]

                except Exception as log_err_base:
                    logger.error(f"Error processing logs for baseline transaction: {log_err_base}")
            elif receipt_base: # Failed transaction (status 0)
                logger.error(f"Baseline adjustment transaction reverted (Status 0). Tx: {self.metrics['tx_hash']}")
                self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                self.metrics['error_message'] = "tx_reverted_onchain (Baseline)"
                self.metrics['gas_used'] = receipt_base.get('gasUsed', 0)
                eff_gas_price_base = receipt_base.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base, 'ether'))
                adjustment_call_success = False
            else: # Transaction sending failed
                logger.error("Baseline adjustment transaction sending/receipt failed.")
                if self.metrics['action_taken'] == self.ACTION_STATES["TX_SENT"]:
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for baseline adjustment failed"
                adjustment_call_success = False

        except Exception as tx_err_base:
            logger.exception(f"Error during baseline adjustment transaction call/wait: {tx_err_base}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"TxError (Baseline): {str(tx_err_base)}"            # self.save_metrics() # Will be saved in finally block
            return False
        
        finally: # Ensure metrics and final position info are always saved
            # 8. Final update of position information (after transaction)
            # If events didn't include complete final position info, use get_position_info
            final_pos_info = self.get_position_info()
            if final_pos_info:        
                self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower', self.metrics['finalTickLower_contract']) # If not filled from event
                self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper', self.metrics['finalTickUpper_contract'])
                self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity', self.metrics['finalLiquidity_contract'])
            else:
                logger.warning("Baseline: Could not get final position info after tx. Metrics might be incomplete for final state.")
                # If final position info is not available, at least use target ticks as final ticks
                if self.metrics['finalTickLower_contract'] is None : self.metrics['finalTickLower_contract'] = target_lower_tick
                if self.metrics['finalTickUpper_contract'] is None : self.metrics['finalTickUpper_contract'] = target_upper_tick
                # Liquidity might remain unknown
            self.save_metrics()           
            return adjustment_call_success    
    
    def save_metrics(self): # According to your new file version
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Columns match _reset_metrics in new version
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',            'range_width_multiplier_setting',
            'external_api_eth_price', # This is usually null for baseline
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
            row_data = {col: self.metrics.get(col) for col in columns}

            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save baseline metrics: {e}")

# --- Main Function ---
def main(): # According to your new file version
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
            temp_test_b_error = BaselineTest("0x0")
            temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
            temp_test_b_error.metrics['error_message'] = f"Address file not found: {ADDRESS_FILE_BASELINE}"
            temp_test_b_error.save_metrics()
            raise FileNotFoundError(f"File not found: {ADDRESS_FILE_BASELINE}")

        logger.info(f"Reading baseline address from: {ADDRESS_FILE_BASELINE}")
        with open(ADDRESS_FILE_BASELINE, 'r') as f:
            content = f.read()
            logger.debug(f"Baseline address file content: {content}")
            addresses_data = json.loads(content)
            baseline_address_val = addresses_data.get('address')
            if not baseline_address_val:
                logger.error(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
                temp_test_b_error = BaselineTest("0x0")
                temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
                temp_test_b_error.metrics['error_message'] = f"Key 'address' not found in {ADDRESS_FILE_BASELINE}"
                temp_test_b_error.save_metrics()
                raise ValueError(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
        logger.info(f"Loaded Baseline Minimal Address: {baseline_address_val}")      
        test_baseline = BaselineTest(baseline_address_val) # Variable name changed
        
        # Read parameters from environment variables or default values
        desired_rwm_base = int(os.getenv('BASELINE_RWM', '50'))
        target_weth_base = float(os.getenv('BASELINE_TARGET_WETH', '1.0'))
        target_usdc_base = float(os.getenv('BASELINE_TARGET_USDC', '2000.0'))
        
        test_baseline.execute_test_steps(
            desired_range_width_multiplier=desired_rwm_base,
            target_weth_balance=target_weth_base,
            target_usdc_balance=target_usdc_base
        )

    except FileNotFoundError as e:
        logger.error(f"Setup Error - Address file not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration Error - Problem reading address file or address key missing: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during baseline main execution:")
        if 'test_baseline' not in locals() and baseline_address_val:
             test_baseline = BaselineTest(baseline_address_val)
        elif 'test_baseline' not in locals():
             test_baseline = BaselineTest("0x0")
        
        test_baseline.metrics['action_taken'] = test_baseline.ACTION_STATES["UNEXPECTED_ERROR"]
        test_baseline.metrics['error_message'] = str(e)
        test_baseline.save_metrics()
    finally:
        logger.info("="*50)
        logger.info("Baseline test run finished.")
        logger.info("="*50)

if __name__ == "__main__":
    main()