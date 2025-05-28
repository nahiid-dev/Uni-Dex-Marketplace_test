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
from web3.logs import DISCARD

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
    print(f"ERROR importing from test.utils in baseline_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr) # [cite: 180]
    sys.exit(1)

# --- Constants ---
# Required values for contract funding, using new values
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6) # 1000 USDC
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272 # Uniswap V3 min tick
MAX_TICK_CONST = 887272 # Uniswap V3 max tick
Q96 = Decimal(2**96)
TOKEN0_DECIMALS = 6  # USDC decimals
TOKEN1_DECIMALS = 18 # WETH decimals


# --- Setup Logging ---
if not logging.getLogger('baseline_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("baseline_test.log", mode='a'), # [cite: 181]
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
            'action_taken': "init", # Using string directly since ACTION_STATES isn't set yet # [cite: 182]
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
            'currentTickLower_contract': None, # [cite: 183]
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
        } # [cite: 184]

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
            "TX_SUCCESS_SKIPPED_ONCHAIN": "tx_success_skipped_onchain", # [cite: 185]
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

    def sqrt_price_x96_to_price_token0_in_token1(self, sqrt_price_x96_str: str) -> Decimal:
        """Convert sqrtPriceX96 to price of token0 in terms of token1 (USDC in WETH)""" # [cite: 186]
        sqrt_price_x96 = Decimal(sqrt_price_x96_str)
        price_t1_in_t0 = (sqrt_price_x96 / Q96) ** 2
        price_t0_in_t1 = Decimal(1) / price_t1_in_t0
        decimals_adjustment = Decimal(10) ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS)
        return price_t0_in_t1 / decimals_adjustment

    def _estimate_liquidity(self, tick_lower: int, tick_upper: int) -> int:
        """Estimate liquidity based on token balances and tick range (simple approximation)."""
        # This method is a simple estimate and should not be used as an accurate source of liquidity. [cite: 187]
        # The contract itself should manage and report liquidity. [cite: 188]
        try:
            if not self.token0_contract_instance and self.token0:
                self.token0_contract_instance = get_contract(self.token0, "IERC20")
            if not self.token1_contract_instance and self.token1:
                self.token1_contract_instance = get_contract(self.token1, "IERC20")

            if not self.token0_contract_instance or not self.token1_contract_instance or not self.pool_contract:
                logger.warning("Cannot estimate liquidity: token contracts or pool contract not initialized.")
                return 0

            token0_bal = self.token0_contract_instance.functions.balanceOf(self.contract_address).call() # [cite: 189]
            token1_bal = self.token1_contract_instance.functions.balanceOf(self.contract_address).call()
            
            slot0 = self.pool_contract.functions.slot0().call()
            current_tick = slot0[1]
            if current_tick < tick_lower:
                return int(token0_bal) # [cite: 190]
            elif current_tick >= tick_upper:
                return int(token1_bal) # [cite: 191]
            else:
                return int((token0_bal + token1_bal) / 2) # [cite: 193]
        except Exception as e:
            logger.error(f"Error estimating liquidity: {e}")
            return 0

    def get_position_info(self) -> dict | None: # [cite: 194]
        """Get current position details from the contract."""
        try:
            # We assume getCurrentPosition exists in your contract. [cite: 195]
            position_data = self.contract.functions.getCurrentPosition().call()
            
            # This section must be adapted to match your getCurrentPosition output structure [cite: 196]
            if len(position_data) >= 5:
                token_id, active, tick_lower, tick_upper, liquidity_contract = position_data[0], position_data[1], position_data[2], position_data[3], position_data[4]
                # The contract's value is preferred. [cite: 197]
                return {
                    'tokenId': token_id, 
                    'active': active,
                    'tickLower': tick_lower,
                    'tickUpper': tick_upper,
                    'liquidity': liquidity_contract # [cite: 198]
                }
            else:
                logger.error(f"Unexpected data structure from getCurrentPosition(): {position_data}")
                return None
        except Exception as e:
            logger.error(f"Error getting position info from contract: {e}")
            self.metrics['error_message'] += f";PosInfoError: {str(e)}"
            return None
            
    def setup(self, desired_range_width_multiplier: int) -> bool: # [cite: 199]
        if not super().setup(desired_range_width_multiplier):
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("web3_utils.w3 not available in BaselineTest setup after base.setup()")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup" # [cite: 200]
                return False

            factory_address = self.contract.functions.factory().call()
            self.factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            if not self.factory_contract: return False

            fee = self.contract.functions.fee().call()
            self.pool_address = self.factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Baseline pool address not found for {self.token0}/{self.token1} fee {fee}") # [cite: 201]
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
            
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            if not self.pool_contract: return False
            logger.info(f"Baseline Pool contract initialized at {self.pool_address}")
            
            self.tick_spacing = self.pool_contract.functions.tickSpacing().call()
            if not self.tick_spacing or self.tick_spacing <= 0: # tickSpacing must be positive # [cite: 202]
                logger.error(f"Invalid tickSpacing read from pool: {self.tick_spacing}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Invalid tickSpacing: {self.tick_spacing}"
                return False
            logger.info(f"Baseline Tick spacing from pool: {self.tick_spacing}")
            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Baseline contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier # [cite: 203]
            
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                logger.error("PRIVATE_KEY not set for setting rangeWidthMultiplier (Baseline).")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "PRIVATE_KEY missing for RWM setup (Baseline)"
                return False
            
            tx_account = Account.from_key(private_key)
            tx_params_rwm = { # [cite: 204]
                'from': tx_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(tx_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }
            try:
                gas_estimate_rwm = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).estimate_gas({'from': tx_account.address})
                tx_params_rwm['gas'] = int(gas_estimate_rwm * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for setRangeWidthMultiplier (Baseline): {e}. Using default 200000.") # [cite: 205]
                tx_params_rwm['gas'] = 200000

            tx_set_rwm_build = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier)
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm_build.build_transaction(tx_params_rwm), private_key)
            
            if not receipt_rwm or receipt_rwm.status == 0:
                tx_hash_rwm_str = receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'
                logger.error(f"Failed to set rangeWidthMultiplier for Baseline contract. TxHash: {tx_hash_rwm_str}")
                self.metrics['error_message'] += f";Failed to set RWM for Baseline (tx: {tx_hash_rwm_str})" # [cite: 206]
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}"
            return False

    def get_pool_state(self) -> tuple[int | None, int | None]: # [cite: 207]
        if not self.pool_contract:
            logger.error("Pool contract not initialized for get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "Pool contract missing for get_pool_state"
            return None, None
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("Web3 not connected in get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "W3 not connected for get_pool_state" # [cite: 208]
            return None, None
        try:
            slot0 = self.pool_contract.functions.slot0().call()
            sqrt_price_x96, tick = slot0[0], slot0[1]
            self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96
            self.metrics['currentTick_pool'] = tick
            self.metrics['actualPrice_pool'] = self.sqrt_price_x96_to_price_token0_in_token1(str(sqrt_price_x96))
            
            logger.info(f"Pool state read: Tick={tick}, SqrtPriceX96={sqrt_price_x96}, ActualPrice={self.metrics['actualPrice_pool']:.4f}") # [cite: 209]
            return sqrt_price_x96, tick
        except Exception as e:
            logger.exception(f"Failed to get pool state (Baseline): {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = f"Pool state read error: {str(e)}"
            return None, None
    
    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]: # [cite: 210]
        if self.tick_spacing is None or current_tick is None:
            logger.error("Tick spacing or current_tick not available for target tick calculation (Baseline).") # [cite: 211]
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Missing data for target tick calc (Baseline)"
            return None, None
        try:
            width_multiplier = self.metrics.get('range_width_multiplier_setting')
            if width_multiplier is None or width_multiplier <= 0:
                width_multiplier = self.contract.functions.rangeWidthMultiplier().call()
            
            if width_multiplier is None or width_multiplier <= 0: # [cite: 212]
                logger.error(f"Invalid rangeWidthMultiplier ({width_multiplier}) for tick calculation.")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = f"Invalid RWM ({width_multiplier}) for tick calc"
                return None, None

            target_half_tick_width = 487 # [cite: 213]

            scaled_half_width = (target_half_tick_width * width_multiplier) // 100
            tick_spacing_multiple = math.ceil(scaled_half_width / self.tick_spacing)
            half_total_tick_width = tick_spacing_multiple * self.tick_spacing

            if half_total_tick_width < self.tick_spacing:
                logger.warning(f"Calculated half width {half_total_tick_width} is less than tick spacing {self.tick_spacing}. Using minimum width.") # [cite: 214]
                half_total_tick_width = self.tick_spacing

            target_lower_tick = math.floor(current_tick / self.tick_spacing) * self.tick_spacing - half_total_tick_width
            target_upper_tick = math.ceil(current_tick / self.tick_spacing) * self.tick_spacing + half_total_tick_width

            target_lower_tick = max(MIN_TICK_CONST, target_lower_tick)
            target_upper_tick = min(MAX_TICK_CONST, target_upper_tick)

            if target_lower_tick >= target_upper_tick: # [cite: 215]
                logger.error(f"Invalid tick range calculated: [{target_lower_tick}, {target_upper_tick}]")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Generated invalid tick range L >= U"
                return None, None

            logger.info("Target ticks calculation details:")
            logger.info(f"  Current tick: {current_tick}")
            logger.info(f"  Range width multiplier: {width_multiplier}") # [cite: 216]
            logger.info(f"  Tick spacing: {self.tick_spacing}")
            logger.info(f"  Half width in ticks: {half_total_tick_width}")
            logger.info(f"  Target range: [{target_lower_tick}, {target_upper_tick}]")

            self.metrics['targetTickLower_offchain'] = target_lower_tick
            self.metrics['targetTickUpper_offchain'] = target_upper_tick
            return target_lower_tick, target_upper_tick
            
        except Exception as e:
            logger.exception(f"Error calculating target ticks off-chain (Baseline): {e}") # [cite: 217]
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Target tick calc error (Baseline): {str(e)}"
            return None, None

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        self.metrics = self._reset_metrics()
        try:
            self.get_current_eth_price()
            current_rwm = self.contract.functions.rangeWidthMultiplier().call() # [cite: 218]
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.")

        adjustment_call_success = False
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx (Baseline)" # [cite: 219]
            self.save_metrics()
            return False
        
        funding_account = Account.from_key(private_key_env)

        try:
            _, current_tick = self.get_pool_state()
            if current_tick is None:
                self.save_metrics()
                return False # [cite: 220]
            
            pos_info_before = self.get_position_info()
            current_pos_active = False
            if pos_info_before:
                current_pos_active = pos_info_before.get('active', False)
                self.metrics['currentTickLower_contract'] = pos_info_before.get('tickLower')
                self.metrics['currentTickUpper_contract'] = pos_info_before.get('tickUpper')
                self.metrics['currentLiquidity_contract'] = pos_info_before.get('liquidity', 0)
            else: # [cite: 221]
                logger.warning("Could not get current position info for Baseline. Assuming no active position.") # [cite: 222]
            
            target_lower_tick, target_upper_tick = self.calculate_target_ticks_offchain(current_tick)
            if target_lower_tick is None or target_upper_tick is None:
                self.save_metrics()
                return False
            
            if current_pos_active and \
               self.metrics['currentTickLower_contract'] is not None and \
               self.metrics['currentTickUpper_contract'] is not None and \
               self.tick_spacing is not None: # [cite: 223]
                TICK_PROXIMITY_THRESHOLD = self.tick_spacing 
                
                is_lower_close = abs(target_lower_tick - self.metrics['currentTickLower_contract']) <= TICK_PROXIMITY_THRESHOLD
                is_upper_close = abs(target_upper_tick - self.metrics['currentTickUpper_contract']) <= TICK_PROXIMITY_THRESHOLD

                if is_lower_close and is_upper_close: # [cite: 224]
                    logger.info(f"Baseline Off-chain proximity check: Target ticks ({target_lower_tick}, {target_upper_tick}) are close to current on-chain ({self.metrics['currentTickLower_contract']}, {self.metrics['currentTickUpper_contract']}). Skipping adjustment call.") # [cite: 225]
                    self.metrics['action_taken'] = self.ACTION_STATES["SKIPPED_PROXIMITY"]
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return True # [cite: 226]
            
            logger.info("Ensuring precise token balances for Baseline contract...")
            if not contract_funder.ensure_precise_token_balances(
                contract_address=self.contract_address,
                token0_address=self.token0,
                token0_decimals=self.token0_decimals,
                target_token0_amount_readable=target_usdc_balance,
                token1_address=self.token1,
                token1_decimals=self.token1_decimals,
                target_token1_amount_readable=target_weth_balance, # [cite: 227]
                funding_account_private_key=private_key_env
            ):
                logger.error("Precise funding for Baseline contract failed.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Precise contract funding failed (Baseline)"
                self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract'] # [cite: 228]
                self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                self.save_metrics()
                return False
            
            try:
                token0_c = get_contract(self.token0, "IERC20")
                token1_c = get_contract(self.token1, "IERC20")
                if token0_c and token1_c:
                    self.metrics['initial_contract_balance_token0'] = token0_c.functions.balanceOf(self.contract_address).call() # [cite: 229]
                    self.metrics['initial_contract_balance_token1'] = token1_c.functions.balanceOf(self.contract_address).call()
            except Exception as bal_err:
                logger.warning(f"Could not read initial contract balances for baseline metrics: {bal_err}")

            logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract...")
            
            tx_function_call_base = self.contract.functions.adjustLiquidityWithCurrentPrice()
            tx_params_base = { # [cite: 230]
                'from': funding_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }
            try:
                gas_estimate_base = tx_function_call_base.estimate_gas({'from': funding_account.address})
                tx_params_base['gas'] = int(gas_estimate_base * 1.25)
                logger.info(f"Estimated gas for 'adjustLiquidityWithCurrentPrice': {gas_estimate_base}, using: {tx_params_base['gas']}")
            except Exception as est_err_base: # [cite: 231]
                logger.warning(f"Gas estimation failed for 'adjustLiquidityWithCurrentPrice': {est_err_base}. Using default 1,500,000") # [cite: 232]
                tx_params_base['gas'] = 1500000
            
            built_tx_base = tx_function_call_base.build_transaction(tx_params_base)
            receipt_base = web3_utils.send_transaction(built_tx_base, private_key_env)

            self.metrics['tx_hash'] = receipt_base.transactionHash.hex() if receipt_base else None
            self.metrics['action_taken'] = self.ACTION_STATES["TX_SENT"]

            if receipt_base and receipt_base.status == 1:
                logger.info(f"Baseline adjustment transaction successful (Status 1). Tx: {self.metrics['tx_hash']}. Processing events...")
                self.metrics['gas_used'] = receipt_base.get('gasUsed', 0) # [cite: 233]
                eff_gas_price_base = receipt_base.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base, 'ether'))
                adjustment_call_success = True
                
                # MODIFIED BLOCK STARTS HERE
                try:
                    logger.info("Processing 'PositionMinted' event for Baseline...")
                    mint_logs_base = self.contract.events.PositionMinted().process_receipt(receipt_base, errors=DISCARD) # [cite: 234]
                    
                    found_mint_event = False
                    if not mint_logs_base:
                        logger.warning("No 'PositionMinted' event found in the transaction receipt for Baseline.")
                    
                    for log_entry in mint_logs_base: 
                        if hasattr(log_entry.args, 'amount0Actual') and hasattr(log_entry.args, 'amount1Actual'):
                            self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual # Use amount0Actual
                            self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual # Use amount1Actual
                            self.metrics['finalTickLower_contract'] = log_entry.args.tickLower
                            self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper
                            self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity
                            found_mint_event = True
                            logger.info(f"Baseline PositionMinted event processed: amount0={log_entry.args.amount0Actual}, amount1={log_entry.args.amount1Actual}, liquidity={log_entry.args.liquidity}")
                            break 
                        else:
                            logger.error(f"Baseline PositionMinted event found, but attribute 'amount0Actual' or 'amount1Actual' is missing. Args: {log_entry.args}")
                    
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"] # [cite: 235]
                    if not found_mint_event:
                        logger.info("Baseline transaction successful, but no new position was minted (or PositionMinted event was not found/had missing args). Liquidity amounts provided to mint will be empty.")
                        # Fees collected would also be from a previous position or not applicable if no removal happened.
                        # self.metrics['fees_collected_token0'] = ... # [cite: 236] (This part needs logic if PositionRemoved event is used)
                        # self.metrics['fees_collected_token1'] = ...

                except Exception as log_err_base:
                    logger.exception(f"Error processing logs for baseline transaction: {log_err_base}")
                    if adjustment_call_success: 
                         self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"] 
                         self.metrics['error_message'] += f";LogProcessingError: {str(log_err_base)}"
                # MODIFIED BLOCK ENDS HERE

            elif receipt_base: 
                logger.error(f"Baseline adjustment transaction reverted (Status 0). Tx: {self.metrics['tx_hash']}") # [cite: 237]
                self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                self.metrics['error_message'] = "tx_reverted_onchain (Baseline)"
                self.metrics['gas_used'] = receipt_base.get('gasUsed', 0)
                eff_gas_price_base = receipt_base.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base, 'ether'))
                adjustment_call_success = False
            else: 
                logger.error("Baseline adjustment transaction sending/receipt failed.")
                if self.metrics['action_taken'] == self.ACTION_STATES["TX_SENT"]: # [cite: 238]
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for baseline adjustment failed"
                adjustment_call_success = False

        except Exception as tx_err_base:
            logger.exception(f"Error during baseline adjustment transaction call/wait: {tx_err_base}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"TxError (Baseline): {str(tx_err_base)}"
            # self.save_metrics() will be called in finally # [cite: 239]
            return False # Ensure to return False if an exception occurs before adjustment_call_success is determined
        
        finally:
            logger.info("Updating final position info in 'finally' block for Baseline...")
            final_pos_info = self.get_position_info()
            if final_pos_info:    
                # Only update from get_position_info if not already set by a successful mint event
                if self.metrics.get('finalTickLower_contract') is None or self.metrics.get('finalTickLower_contract') == 0 :
                    self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower')
                if self.metrics.get('finalTickUpper_contract') is None or self.metrics.get('finalTickUpper_contract') == 0:
                    self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper')
                if self.metrics.get('finalLiquidity_contract') is None or self.metrics.get('finalLiquidity_contract') == 0:
                    self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity') # [cite: 240]
            else:
                logger.warning("Baseline: Could not get final position info after tx. Metrics might be incomplete for final state.") # [cite: 241]
                if self.metrics.get('finalTickLower_contract') is None and target_lower_tick is not None: self.metrics['finalTickLower_contract'] = target_lower_tick
                if self.metrics.get('finalTickUpper_contract') is None and target_upper_tick is not None: self.metrics['finalTickUpper_contract'] = target_upper_tick
            
            self.save_metrics()
            
        return adjustment_call_success
    
    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_value = self.metrics.get('actualPrice_pool') # [cite: 242]
        if isinstance(metric_value, Decimal):
            self.metrics['actualPrice_pool'] = f"{metric_value:.6f}"
        elif metric_value is None:
            self.metrics['actualPrice_pool'] = ""
        
        columns = [ # [cite: 243]
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
            file_exists = RESULTS_FILE.is_file() # [cite: 244]
            row_data = {col: self.metrics.get(col) for col in columns}
    
            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}")
        except Exception as e: # [cite: 245]
            logger.exception(f"Failed to save baseline metrics: {e}")

def main():
    logger.info("="*50)
    logger.info("Starting Baseline Minimal Liquidity Manager Test on Fork")
    logger.info("="*50)

    if not web3_utils.init_web3():
        logger.critical("Web3 initialization failed. Exiting baseline test.") # [cite: 246]
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
            temp_test_b_error.save_metrics() # [cite: 247]
            raise FileNotFoundError(f"File not found: {ADDRESS_FILE_BASELINE}")

        logger.info(f"Reading baseline address from: {ADDRESS_FILE_BASELINE}")
        with open(ADDRESS_FILE_BASELINE, 'r') as f:
            content = f.read()
            logger.debug(f"Baseline address file content: {content}")
            addresses_data = json.loads(content)
            baseline_address_val = addresses_data.get('address')
            if not baseline_address_val:
                logger.error(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
                temp_test_b_error = BaselineTest("0x0") # [cite: 248]
                temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
                temp_test_b_error.metrics['error_message'] = f"Key 'address' not found in {ADDRESS_FILE_BASELINE}"
                temp_test_b_error.save_metrics()
                raise ValueError(f"Key 'address' not found in {ADDRESS_FILE_BASELINE}")
        logger.info(f"Loaded Baseline Minimal Address: {baseline_address_val}")
        test_baseline = BaselineTest(baseline_address_val)
        
        desired_rwm_base = int(os.getenv('BASELINE_RWM', '100')) # DEFAULT CHANGED TO 100
        target_weth_base = float(os.getenv('BASELINE_TARGET_WETH', '1.0')) # [cite: 249]
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
    except Exception as e: # [cite: 250]
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
    main() # [cite: 251]