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
from web3.exceptions import ContractLogicError # Added for more specific error handling

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
Q96 = Decimal(2**96)
TOKEN0_DECIMALS = 6  # USDC decimals (Global constant from your file)
TOKEN1_DECIMALS = 18 # WETH decimals (Global constant from your file)

# ADDED: For swap functionality
TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV = os.getenv('TOKEN_MANAGER_OPTIMIZED_ADDRESS')
WETH_MAINNET_ADDR = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_MAINNET_ADDR = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


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
        # YOUR ORIGINAL _reset_metrics - UNCHANGED
        return {
            'timestamp': None,
            'contract_type': 'Baseline',
            'action_taken': "init", 
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
        # YOUR ORIGINAL __init__ (ACTION_STATES kept as is)
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
            "UNEXPECTED_ERROR": "unexpected_error",
            # ADDED for new stages (can be refined or removed if not strictly used by this simpler structure)
            "TX_SUCCESS_ADJUSTED_INITIAL": "tx_success_adjusted_initial",
            "TX_SUCCESS_SWAP_FEES": "tx_success_swap_fees", 
            "TX_SUCCESS_ADJUSTED_FINAL": "tx_success_adjusted_final", 
            "SWAP_FOR_FEES_FAILED": "swap_for_fees_failed"
        }
        self.factory_contract = None
        self.tick_spacing = None
        self.pool_address = None
        self.pool_contract = None
        self.token0_contract_instance = None
        self.token1_contract_instance = None
        self.metrics = self._reset_metrics() # This will now use the ACTION_STATES['INIT']

        # ADDED: For swap functionality
        self.token_manager_optimized_address = None
        self.token_manager_contract = None
        if TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV:
            try:
                self.token_manager_optimized_address = Web3.to_checksum_address(TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV)
                logger.info(f"TokenOperationsManagerOptimized address loaded for Baseline: {self.token_manager_optimized_address}")
            except ValueError:
                logger.error(f"Invalid TokenOperationsManagerOptimized address from ENV for Baseline: {TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV}")
        else:
            logger.warning("TOKEN_MANAGER_OPTIMIZED_ADDRESS env var not set for Baseline. Swap for fee generation will be skipped.")


    def sqrt_price_x96_to_price_token0_in_token1(self, sqrt_price_x96_str: str) -> Decimal:
        """Convert sqrtPriceX96 to price of token0 in terms of token1 (USDC in WETH)""" 
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        sqrt_price_x96 = Decimal(sqrt_price_x96_str)
        price_t1_in_t0 = (sqrt_price_x96 / Q96) ** 2
        if price_t1_in_t0 == Decimal(0): 
            return Decimal("inf") if sqrt_price_x96 != Decimal(0) else Decimal(0)
        price_t0_in_t1 = Decimal(1) / price_t1_in_t0
        # Using global TOKEN0_DECIMALS and TOKEN1_DECIMALS as in your original file
        decimals_adjustment = Decimal(10) ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS)
        if decimals_adjustment == Decimal(0): 
            return Decimal("-1") 
        return price_t0_in_t1 / decimals_adjustment

    def _estimate_liquidity(self, tick_lower: int, tick_upper: int) -> int:
        """Estimate liquidity based on token balances and tick range (simple approximation)."""
        # YOUR ORIGINAL FUNCTION - UNCHANGED
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
            if current_tick < tick_lower:
                return int(token0_bal) 
            elif current_tick >= tick_upper:
                return int(token1_bal) 
            else:
                return int((token0_bal + token1_bal) / 2) 
        except Exception as e:
            logger.error(f"Error estimating liquidity: {e}")
            return 0

    def get_position_info(self) -> dict | None: 
        """Get current position details from the contract."""
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        try:
            position_data = self.contract.functions.getCurrentPosition().call()
            
            if len(position_data) >= 5:
                token_id, active, tick_lower, tick_upper, liquidity_contract = position_data[0], position_data[1], position_data[2], position_data[3], position_data[4]
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
            
    def setup(self, desired_range_width_multiplier: int) -> bool: 
        # YOUR ORIGINAL FUNCTION - UNCHANGED
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
            if not self.tick_spacing or self.tick_spacing <= 0: 
                logger.error(f"Invalid tickSpacing read from pool: {self.tick_spacing}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Invalid tickSpacing: {self.tick_spacing}"
                return False
            logger.info(f"Baseline Tick spacing from pool: {self.tick_spacing}")
            
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
            logger.exception(f"Baseline setup failed: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}"
            return False

    def get_pool_state(self) -> tuple[int | None, int | None]: 
        # YOUR ORIGINAL FUNCTION - UNCHANGED (except formatting of ActualPrice log)
        if not self.pool_contract:
            logger.error("Pool contract not initialized for get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "Pool contract missing for get_pool_state"
            return None, None
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("Web3 not connected in get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "W3 not connected for get_pool_state" 
            return None, None
        try:
            slot0 = self.pool_contract.functions.slot0().call()
            sqrt_price_x96, tick = slot0[0], slot0[1]
            self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96
            self.metrics['currentTick_pool'] = tick
            self.metrics['actualPrice_pool'] = self.sqrt_price_x96_to_price_token0_in_token1(str(sqrt_price_x96))
            
            logger.info(f"Pool state read: Tick={tick}, SqrtPriceX96={sqrt_price_x96}, ActualPrice={self.metrics['actualPrice_pool']}") 
            return sqrt_price_x96, tick
        except Exception as e:
            logger.exception(f"Failed to get pool state (Baseline): {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = f"Pool state read error: {str(e)}"
            return None, None
    
    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]: 
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if self.tick_spacing is None or current_tick is None:
            logger.error("Tick spacing or current_tick not available for target tick calculation (Baseline).") 
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Missing data for target tick calc (Baseline)"
            return None, None
        try:
            width_multiplier = self.metrics.get('range_width_multiplier_setting')
            if width_multiplier is None or width_multiplier <= 0:
                width_multiplier = self.contract.functions.rangeWidthMultiplier().call()
            
            if width_multiplier is None or width_multiplier <= 0: 
                logger.error(f"Invalid rangeWidthMultiplier ({width_multiplier}) for tick calculation.")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = f"Invalid RWM ({width_multiplier}) for tick calc"
                return None, None

            target_half_tick_width = 487 

            scaled_half_width = (target_half_tick_width * width_multiplier) // 100
            tick_spacing_multiple = math.ceil(scaled_half_width / self.tick_spacing)
            half_total_tick_width = tick_spacing_multiple * self.tick_spacing

            if half_total_tick_width < self.tick_spacing:
                logger.warning(f"Calculated half width {half_total_tick_width} is less than tick spacing {self.tick_spacing}. Using minimum width.") 
                half_total_tick_width = self.tick_spacing

            target_lower_tick = math.floor(current_tick / self.tick_spacing) * self.tick_spacing - half_total_tick_width
            target_upper_tick = math.ceil(current_tick / self.tick_spacing) * self.tick_spacing + half_total_tick_width

            target_lower_tick = max(MIN_TICK_CONST, target_lower_tick)
            target_upper_tick = min(MAX_TICK_CONST, target_upper_tick)

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

            self.metrics['targetTickLower_offchain'] = target_lower_tick
            self.metrics['targetTickUpper_offchain'] = target_upper_tick
            return target_lower_tick, target_upper_tick
            
        except Exception as e:
            logger.exception(f"Error calculating target ticks off-chain (Baseline): {e}") 
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Target tick calc error (Baseline): {str(e)}"
            return None, None

    # ADDED: _perform_swap_for_fees method
    def _perform_swap_for_fees(self, funding_account, private_key_env, swap_token_in_addr: str, swap_token_out_addr: str, swap_amount_readable: Decimal, token_in_decimals: int, token_out_decimals: int):
        """Helper function to perform a swap using TokenOperationsManagerOptimized."""
        if not self.token_manager_optimized_address:
            logger.warning("TokenManagerOptimized address not set for Baseline. Skipping swap for fees.")
            return False
        
        if not self.token_manager_contract: 
            self.token_manager_contract = web3_utils.get_contract(self.token_manager_optimized_address, "TokenOperationsManagerOptimized")
            if not self.token_manager_contract:
                logger.error("Failed to get TokenOperationsManagerOptimized contract instance for Baseline.")
                return False

        logger.info(f"Attempting swap for Baseline: {swap_amount_readable} of {swap_token_in_addr} for {swap_token_out_addr} via TokenManager.")

        try:
            pool_fee_for_swap = self.contract.functions.fee().call() 
            amount_to_swap_wei = int(swap_amount_readable * (Decimal(10) ** token_in_decimals))

            token_in_contract_for_approve = web3_utils.get_contract(swap_token_in_addr, "IERC20")
            
            logger.info(f"Approving TokenManagerOptimized ({self.token_manager_optimized_address}) to spend {swap_amount_readable} of token {swap_token_in_addr} from deployer {funding_account.address}...")
            current_nonce_approve = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            approve_tx_params = {'from': funding_account.address, 'nonce': current_nonce_approve}
            
            try:
                gas_est_approve = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).estimate_gas({'from': funding_account.address})
                approve_tx_params['gas'] = int(gas_est_approve * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation for TokenManager approval (Baseline) failed: {e}. Using default 100,000.")
                approve_tx_params['gas'] = 100000

            built_approve_tx = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).build_transaction(approve_tx_params)
            receipt_approve = web3_utils.send_transaction(built_approve_tx, private_key_env)

            if not receipt_approve or receipt_approve.status == 0:
                logger.error(f"Failed to approve TokenManagerOptimized (Baseline). Receipt: {receipt_approve}")
                return False
            logger.info(f"TokenManagerOptimized approved for Baseline. Tx: {receipt_approve.transactionHash.hex()}")
            
            current_nonce_swap = web3_utils.w3.eth.get_transaction_count(funding_account.address) 
            logger.info(f"Calling swap on TokenManagerOptimized for Baseline: {swap_amount_readable} {swap_token_in_addr} -> {swap_token_out_addr}, fee {pool_fee_for_swap}...")
            swap_tx_params = {'from': funding_account.address, 'nonce': current_nonce_swap }
            
            try:
                checksum_token_in = Web3.to_checksum_address(swap_token_in_addr)
                checksum_token_out = Web3.to_checksum_address(swap_token_out_addr)
                gas_est_swap = self.token_manager_contract.functions.swap(
                    checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0 
                ).estimate_gas({'from': funding_account.address})
                swap_tx_params['gas'] = int(gas_est_swap * 1.30) 
            except ContractLogicError as cle:
                logger.error(f"Gas estimation for TokenManager swap (Baseline) failed due to contract logic: {cle}.")
                return False
            except Exception as e:
                logger.warning(f"Gas estimation for TokenManager swap (Baseline) failed: {e}. Using default 700,000.")
                swap_tx_params['gas'] = 700000

            built_swap_tx = self.token_manager_contract.functions.swap(
                checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0
            ).build_transaction(swap_tx_params)
            
            receipt_swap = web3_utils.send_transaction(built_swap_tx, private_key_env)

            if receipt_swap and receipt_swap.status == 1:
                logger.info(f"Swap via TokenManagerOptimized for Baseline successful. Tx: {receipt_swap.transactionHash.hex()}")
                swap_logs = self.token_manager_contract.events.Operation().process_receipt(receipt_swap, errors=DISCARD)
                for log_entry in swap_logs:
                    op_type_bytes32 = log_entry.args.opType
                    is_swap_op = (web3_utils.w3.to_hex(op_type_bytes32) == web3_utils.w3.to_hex(web3_utils.w3.solidity_keccak(['string'],['SWAP'])))

                    if is_swap_op:
                        # Use self.token0_decimals and self.token1_decimals from LiquidityTestBase
                        decimals_for_amount_out = self.token0_decimals if Web3.to_checksum_address(log_entry.args.tokenB) == Web3.to_checksum_address(self.token0) else self.token1_decimals
                        amount_out_readable = Decimal(log_entry.args.amount) / (Decimal(10) ** decimals_for_amount_out)
                        logger.info(f"TokenManager Swap Event (Baseline context): TokenIn={log_entry.args.tokenA}, TokenOut={log_entry.args.tokenB}, AmountOut={amount_out_readable:.6f}")
                return True
            else:
                logger.error(f"Swap via TokenManagerOptimized for Baseline failed. Receipt: {receipt_swap}")
                return False
        except Exception as e:
            logger.exception(f"Error during _perform_swap_for_fees for Baseline: {e}")
            return False

    # MODIFIED adjust_position
    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        self.metrics = self._reset_metrics() 
        try:
            self.get_current_eth_price()
            current_rwm = self.contract.functions.rangeWidthMultiplier().call() 
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.")

        initial_adjustment_call_success = False
        second_adjustment_call_success = False 
        swap_for_fees_call_success = False
        
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx (Baseline)" 
            self.save_metrics()
            return False
        
        funding_account = Account.from_key(private_key_env)
        
        # Store current on-chain position from strategy contract before any action
        pos_info_before_all = self.get_position_info()
        if pos_info_before_all:
            self.metrics['currentTickLower_contract'] = pos_info_before_all.get('tickLower')
            self.metrics['currentTickUpper_contract'] = pos_info_before_all.get('tickUpper')
            self.metrics['currentLiquidity_contract'] = pos_info_before_all.get('liquidity', 0)
        else:
            self.metrics['currentTickLower_contract'] = 0 
            self.metrics['currentTickUpper_contract'] = 0
            self.metrics['currentLiquidity_contract'] = 0


        # --- STAGE 1: Initial Position Adjustment (Original logic flow from your file) ---
        logger.info("\n--- STAGE 1: Baseline Strategy - Initial position adjustment ---")
        try:
            # This uses YOUR original get_pool_state
            _, current_tick_initial = self.get_pool_state() 
            if current_tick_initial is None:
                self.save_metrics() 
                return False 
            
            # This uses YOUR original calculate_target_ticks_offchain
            target_lower_tick_initial, target_upper_tick_initial = self.calculate_target_ticks_offchain(current_tick_initial)
            if target_lower_tick_initial is None or target_upper_tick_initial is None:
                self.save_metrics()
                return False
            
            # Your original proximity check logic (from the provided file)
            current_pos_active_stage1 = self.metrics.get('currentLiquidity_contract', 0) > 0 
            if current_pos_active_stage1 and \
               self.metrics['currentTickLower_contract'] is not None and \
               self.metrics['currentTickUpper_contract'] is not None and \
               self.tick_spacing is not None: 
                TICK_PROXIMITY_THRESHOLD = self.tick_spacing  
                
                is_lower_close = abs(target_lower_tick_initial - self.metrics['currentTickLower_contract']) <= TICK_PROXIMITY_THRESHOLD
                is_upper_close = abs(target_upper_tick_initial - self.metrics['currentTickUpper_contract']) <= TICK_PROXIMITY_THRESHOLD

                if is_lower_close and is_upper_close: 
                    logger.info(f"Baseline Off-chain proximity check: Target ticks ({target_lower_tick_initial}, {target_upper_tick_initial}) are close to current on-chain ({self.metrics['currentTickLower_contract']}, {self.metrics['currentTickUpper_contract']}). Skipping initial adjustment call.") 
                    self.metrics['action_taken'] = self.ACTION_STATES["SKIPPED_PROXIMITY"]
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    initial_adjustment_call_success = True # Considered success for this stage as it's a valid skip
                    # If skipped, we might still want to proceed to swap if a position exists
                    # For now, if proximity skip, we will also skip swap and second adjustment.
                    # If you want to swap even if proximity is met, this logic needs adjustment.
                    if not swap_for_fees_call_success: # Only save and return if we are not proceeding to swap
                        self.save_metrics()
                        return True 
            
            if not initial_adjustment_call_success or self.metrics.get('action_taken') != self.ACTION_STATES["SKIPPED_PROXIMITY"]:
                logger.info("Ensuring precise token balances for Baseline contract (initial)...")
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
                    logger.error("Precise funding for Baseline contract failed (initial).")
                    self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                    self.metrics['error_message'] = "Precise contract funding failed (Baseline initial)"
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return False
                
                try:
                    token0_c = get_contract(self.token0, "IERC20")
                    token1_c = get_contract(self.token1, "IERC20")
                    if token0_c and token1_c:
                        self.metrics['initial_contract_balance_token0'] = token0_c.functions.balanceOf(self.contract_address).call() 
                        self.metrics['initial_contract_balance_token1'] = token1_c.functions.balanceOf(self.contract_address).call()
                except Exception as bal_err:
                    logger.warning(f"Could not read initial contract balances for baseline metrics: {bal_err}")

                logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract (initial)...")
                
                tx_function_call_base_initial = self.contract.functions.adjustLiquidityWithCurrentPrice()
                current_nonce_initial_adjust = web3_utils.w3.eth.get_transaction_count(funding_account.address) 
                tx_params_base_initial = { 
                    'from': funding_account.address,
                    'nonce': current_nonce_initial_adjust,
                    'chainId': int(web3_utils.w3.net.version)
                }
                try:
                    gas_estimate_base_initial = tx_function_call_base_initial.estimate_gas({'from': funding_account.address})
                    tx_params_base_initial['gas'] = int(gas_estimate_base_initial * 1.25)
                except Exception as est_err_base: 
                    logger.warning(f"Gas estimation for 'adjustLiquidityWithCurrentPrice' (initial) failed: {est_err_base}. Using default 1,500,000") 
                    tx_params_base_initial['gas'] = 1500000
                
                built_tx_base_initial = tx_function_call_base_initial.build_transaction(tx_params_base_initial)
                receipt_base_initial = web3_utils.send_transaction(built_tx_base_initial, private_key_env)

                self.metrics['tx_hash'] = receipt_base_initial.transactionHash.hex() if receipt_base_initial else None
                
                if receipt_base_initial and receipt_base_initial.status == 1:
                    logger.info(f"Baseline initial adjustment transaction successful. Tx: {self.metrics['tx_hash']}.")
                    self.metrics['gas_used'] = receipt_base_initial.get('gasUsed', 0) 
                    eff_gas_price_base_initial = receipt_base_initial.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                    self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base_initial, 'ether'))
                    initial_adjustment_call_success = True
                    
                    adj_metrics_logs_initial = self.contract.events.BaselineAdjustmentMetrics().process_receipt(receipt_base_initial, errors=DISCARD)
                    was_adjusted_onchain_initial = False
                    if adj_metrics_logs_initial:
                        was_adjusted_onchain_initial = adj_metrics_logs_initial[0].args.adjusted
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"] if was_adjusted_onchain_initial else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"]
                        self.metrics['targetTickLower_offchain'] = adj_metrics_logs_initial[0].args.targetTickLower 
                        self.metrics['targetTickUpper_offchain'] = adj_metrics_logs_initial[0].args.targetTickUpper
                    else:
                         self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"] 

                    if was_adjusted_onchain_initial:
                        mint_logs_base_initial = self.contract.events.PositionMinted().process_receipt(receipt_base_initial, errors=DISCARD) 
                        if mint_logs_base_initial: 
                            log_entry = mint_logs_base_initial[0] 
                            if hasattr(log_entry.args, 'amount0Actual') and hasattr(log_entry.args, 'amount1Actual'): 
                                self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual 
                                self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual 
                                self.metrics['finalTickLower_contract'] = log_entry.args.tickLower 
                                self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper 
                                self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity 
                            else: 
                                logger.error(f"Baseline PositionMinted (initial) event missing amount0Actual/amount1Actual attributes.") 
                        else: 
                            logger.warning("Baseline initial tx successful & adjusted, but no PositionMinted event found.") 
                    else: 
                        logger.info("Baseline initial adjustment call successful but skipped on-chain position change.")
                        self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                        self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                        self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                        self.metrics['amount0_provided_to_mint'] = 0 
                        self.metrics['amount1_provided_to_mint'] = 0 

                elif receipt_base_initial: 
                    logger.error(f"Baseline initial adjustment transaction reverted. Tx: {self.metrics['tx_hash']}") 
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                    self.metrics['error_message'] = "tx_reverted_onchain (Baseline initial)"
                    self.metrics['gas_used'] = receipt_base_initial.get('gasUsed', 0)
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return False
                else: 
                    logger.error("Baseline initial adjustment transaction sending/receipt failed.")
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                    if not self.metrics.get('error_message'): self.metrics['error_message'] = "send_transaction for baseline initial adjustment failed"
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return False

        except Exception as tx_err_initial: 
            logger.exception(f"Error during baseline initial adjustment: {tx_err_initial}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"TxError (Baseline initial): {str(tx_err_initial)}"
            self.save_metrics() 
            return False 
        
        # STAGE 2: Perform Swap via TokenManager to Generate Fees
        if initial_adjustment_call_success:
            logger.info("\n--- STAGE 2: Baseline Strategy - Performing swap for fees ---")
            # self.token0 is USDC, self.token1 is WETH (from LiquidityTestBase.setup)
            swap_token_in_addr = self.token1 
            swap_token_out_addr = self.token0 
            swap_amount_readable = Decimal("0.05") 
            token_in_decimals_for_swap = self.token1_decimals 
            token_out_decimals_for_swap = self.token0_decimals

            swap_for_fees_call_success = self._perform_swap_for_fees(
                funding_account, 
                private_key_env, 
                swap_token_in_addr, 
                swap_token_out_addr, 
                swap_amount_readable, 
                token_in_decimals_for_swap,
                token_out_decimals_for_swap
            )
            if swap_for_fees_call_success:
                 self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]
            else:
                 self.metrics['action_taken'] = self.ACTION_STATES["SWAP_FOR_FEES_FAILED"]
                 self.metrics['error_message'] += ";Swap for fees failed or skipped"


        # STAGE 3: Second Position Adjustment by Strategy Contract (to collect fees)
        if initial_adjustment_call_success and swap_for_fees_call_success: 
            logger.info("\n--- STAGE 3: Baseline Strategy - Second position adjustment (fee collection) ---")
            try:
                logger.info("Ensuring precise token balances for Baseline contract (final)...")
                if not contract_funder.ensure_precise_token_balances(
                    contract_address=self.contract_address,
                    token0_address=self.token0, token0_decimals=self.token0_decimals,
                    target_token0_amount_readable=target_usdc_balance, 
                    token1_address=self.token1, token1_decimals=self.token1_decimals,
                    target_token1_amount_readable=target_weth_balance,
                    funding_account_private_key=private_key_env
                ):
                    logger.error("Precise funding for Baseline contract failed (final).")
                    self.metrics['error_message'] += ";Funding failed (final)"
                
                logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract (final)...")
                tx_function_call_base_final = self.contract.functions.adjustLiquidityWithCurrentPrice()
                current_nonce_final_adjust = web3_utils.w3.eth.get_transaction_count(funding_account.address) 
                tx_params_base_final = {
                    'from': funding_account.address,
                    'nonce': current_nonce_final_adjust,
                    'chainId': int(web3_utils.w3.net.version)
                }
                try:
                    gas_estimate_base_final = tx_function_call_base_final.estimate_gas({'from': funding_account.address})
                    tx_params_base_final['gas'] = int(gas_estimate_base_final * 1.25)
                except Exception as est_err_base_final:
                    logger.warning(f"Gas estimation failed for 'adjustLiquidityWithCurrentPrice' (final): {est_err_base_final}. Using default 1,500,000")
                    tx_params_base_final['gas'] = 1500000
                
                built_tx_base_final = tx_function_call_base_final.build_transaction(tx_params_base_final)
                receipt_base_final = web3_utils.send_transaction(built_tx_base_final, private_key_env)

                self.metrics['tx_hash'] = receipt_base_final.transactionHash.hex() if receipt_base_final else self.metrics.get('tx_hash')

                if receipt_base_final and receipt_base_final.status == 1:
                    logger.info(f"Baseline final adjustment transaction successful. Tx: {self.metrics['tx_hash']}.")
                    self.metrics['gas_used'] = receipt_base_final.get('gasUsed', 0)
                    eff_gas_price_base_final = receipt_base_final.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                    self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base_final, 'ether'))
                    second_adjustment_call_success = True
                    
                    adj_metrics_logs_final = self.contract.events.BaselineAdjustmentMetrics().process_receipt(receipt_base_final, errors=DISCARD)
                    was_adjusted_onchain_final = False
                    if adj_metrics_logs_final:
                        was_adjusted_onchain_final = adj_metrics_logs_final[0].args.adjusted
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_FINAL"] if was_adjusted_onchain_final else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"]
                        self.metrics['targetTickLower_offchain'] = adj_metrics_logs_final[0].args.targetTickLower
                        self.metrics['targetTickUpper_offchain'] = adj_metrics_logs_final[0].args.targetTickUpper
                    else: 
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_FINAL"] 

                    if was_adjusted_onchain_final:
                        mint_logs_base_final = self.contract.events.PositionMinted().process_receipt(receipt_base_final, errors=DISCARD)
                        if mint_logs_base_final:
                            log_entry = mint_logs_base_final[0]
                            self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual 
                            self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual 
                            self.metrics['finalTickLower_contract'] = log_entry.args.tickLower
                            self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper
                            self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity
                        else:
                            logger.warning("Baseline final tx successful & adjusted, but no PositionMinted event found.")
                    else: 
                        logger.info("Baseline final adjustment call successful but skipped on-chain position change.")
                    
                    logger.info("Baseline: Fees (if any) were collected into contract. Specific fee event not present.")
                    self.metrics['fees_collected_token0'] = 0 
                    self.metrics['fees_collected_token1'] = 0 
                
                elif receipt_base_final: 
                    logger.error(f"Baseline final adjustment transaction reverted. Tx: {self.metrics['tx_hash']}")
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                    self.metrics['error_message'] += ";Final adjustment reverted (Baseline)"
                    self.metrics['gas_used'] = receipt_base_final.get('gasUsed', 0)
                else: 
                    logger.error("Baseline final adjustment transaction sending/receipt failed.")
                    if self.metrics['action_taken'] != self.ACTION_STATES["TX_REVERTED"]:
                         self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                    self.metrics['error_message'] += ";Final adjustment send failed (Baseline)"

            except Exception as tx_err_final:
                logger.exception(f"Error during baseline final adjustment: {tx_err_final}")
                if self.metrics['action_taken'] == self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]:
                    self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] += f";TxError (Baseline final): {str(tx_err_final)}"
        
         
            logger.info("Updating final position info in 'finally' block for Baseline (absolute final state)...") 
            final_pos_info = self.get_position_info() 
            if final_pos_info:  
                self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower') 
                self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper') 
                self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity') 
            else: 
                logger.warning("Baseline: Could not get final position info from contract in finally block.") 
            # This uses YOUR original save_metrics
            self.save_metrics() 
            
        if not initial_adjustment_call_success: return False
        if self.token_manager_optimized_address: 
            if not swap_for_fees_call_success: return False
            if not second_adjustment_call_success: return False 
        return True


    def save_metrics(self): 
        # YOUR ORIGINAL save_metrics - UNCHANGED
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_value = self.metrics.get('actualPrice_pool') 
        if isinstance(metric_value, Decimal):
            self.metrics['actualPrice_pool'] = f"{metric_value}" # Keeps E notation
        elif metric_value is None:
            self.metrics['actualPrice_pool'] = ""
        
        # Ensure all numeric metrics that might be None are converted to a suitable format for CSV
        # Using the columns list defined below
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
        
        for key_to_check in columns: # Iterate over defined columns
            if key_to_check not in self.metrics or self.metrics.get(key_to_check) is None:
                 # Default to 0 for typical numeric fields, else empty string
                if key_to_check in ['sqrtPriceX96_pool', 'currentTick_pool', 
                                    'targetTickLower_offchain', 'targetTickUpper_offchain',
                                    'initial_contract_balance_token0', 'initial_contract_balance_token1',
                                    'currentTickLower_contract', 'currentTickUpper_contract', 
                                    'currentLiquidity_contract', 'finalTickLower_contract', 
                                    'finalTickUpper_contract', 'finalLiquidity_contract', 
                                    'amount0_provided_to_mint', 'amount1_provided_to_mint',
                                    'fees_collected_token0', 'fees_collected_token1',
                                    'gas_used', 'gas_cost_eth', 'range_width_multiplier_setting']:
                    self.metrics[key_to_check] = 0
                elif key_to_check not in ['tx_hash', 'error_message', 'action_taken', 'contract_type', 'timestamp', 'actualPrice_pool', 'external_api_eth_price']:
                     self.metrics[key_to_check] = ""
                elif self.metrics.get(key_to_check) is None : # For those specific string/decimal fields if None
                    self.metrics[key_to_check] = ""


        try:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True) 
            file_exists = RESULTS_FILE.is_file() 
            row_data = {col: self.metrics.get(col) for col in columns} # Use .get to avoid KeyError if a metric was missed in reset
    
            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f: 
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore') 
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0: 
                    writer.writeheader() 
                writer.writerow(row_data) 
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}") 
        except Exception as e: 
            logger.exception(f"Failed to save baseline metrics: {e}") 

def main(): 
    # YOUR ORIGINAL main FUNCTION - Corrected NameError
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
    test_baseline = None 
    try:
        baseline_address_env = os.getenv('BASELINE_MINIMAL_ADDRESS')
        if not baseline_address_env :
            logger.error(f"BASELINE_MINIMAL_ADDRESS not found in environment variables.")
            temp_test_b_error = BaselineTest("0x"+"0"*40) 
            temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
            temp_test_b_error.metrics['error_message'] = "BASELINE_MINIMAL_ADDRESS env var not found"
            temp_test_b_error.save_metrics()
            raise ValueError("BASELINE_MINIMAL_ADDRESS env var not found")
        
        baseline_address_val = Web3.to_checksum_address(baseline_address_env)
        logger.info(f"Loaded Baseline Minimal Address from ENV: {baseline_address_val}") 
        
        test_baseline = BaselineTest(baseline_address_val) 
        
        desired_rwm_base = int(os.getenv('BASELINE_RWM', '100')) 
        # Corrected variable names here:
        target_weth_for_test = float(os.getenv('BASELINE_TARGET_WETH', '1.0')) 
        target_usdc_for_test = float(os.getenv('BASELINE_TARGET_USDC', '2000.0')) 
        
        test_baseline.execute_test_steps( 
            desired_range_width_multiplier=desired_rwm_base, 
            target_weth_balance=target_weth_for_test, # Use corrected name
            target_usdc_balance=target_usdc_for_test  # Use corrected name
        )

    except FileNotFoundError as e: 
        logger.error(f"Setup Error - Address file not found: {e}") 
    except ValueError as e: 
        logger.error(f"Configuration Error - Problem reading address or address key missing: {e}") 
    except Exception as e: 
        logger.exception(f"An unexpected error occurred during baseline main execution:") 
        if test_baseline is None: 
            if baseline_address_val and Web3.is_address(baseline_address_val): 
                test_baseline = BaselineTest(baseline_address_val) 
            else: 
                test_baseline = BaselineTest("0x"+"0"*40) 
        
        test_baseline.metrics['action_taken'] = test_baseline.ACTION_STATES["UNEXPECTED_ERROR"] 
        test_baseline.metrics['error_message'] = str(e) 
        test_baseline.save_metrics() 
     
        logger.info("="*50) 
        logger.info("Baseline test run finished.") 
        logger.info("="*50) 

if __name__ == "__main__":
    main() 