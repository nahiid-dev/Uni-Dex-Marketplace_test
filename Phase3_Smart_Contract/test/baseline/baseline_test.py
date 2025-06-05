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
    print(f"ERROR importing from test.utils in baseline_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr) # [cite: 2]
    sys.exit(1)

# --- Constants ---
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6) # 1000 USDC
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272 # Uniswap V3 min tick
MAX_TICK_CONST = 887272 # Uniswap V3 max tick
Q96 = Decimal(2**96) # Kept for consistency with your original file
TOKEN0_DECIMALS = 6  # USDC decimals (Global constant from your file)
TOKEN1_DECIMALS = 18 # WETH decimals (Global constant from your file)

TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV = os.getenv('TOKEN_MANAGER_OPTIMIZED_ADDRESS')

# --- Setup Logging ---
if not logging.getLogger('baseline_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # [cite: 3]
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

    def __init__(self, contract_address: str): # [cite: 4]
        super().__init__(contract_address, "BaselineMinimal") # [cite: 4]
        self.ACTION_STATES = { 
            "INIT": "init", "SETUP_FAILED": "setup_failed",
            "POOL_READ_FAILED": "pool_read_failed", "CALCULATION_FAILED": "calculation_failed",
            "SKIPPED_PROXIMITY": "skipped_proximity", "FUNDING_FAILED": "funding_failed",
            "TX_SENT": "tx_sent", "TX_SUCCESS_ADJUSTED": "tx_success_adjusted", 
            "TX_SUCCESS_SKIPPED_ONCHAIN": "tx_success_skipped_onchain", 
            "TX_REVERTED": "tx_reverted", "TX_WAIT_FAILED": "tx_wait_failed", # [cite: 5]
            "METRICS_UPDATE_FAILED": "metrics_update_failed", "UNEXPECTED_ERROR": "unexpected_error",
            "TX_SUCCESS_ADJUSTED_INITIAL": "tx_success_adjusted_initial",
            "TX_SUCCESS_SWAP_FEES": "tx_success_swap_fees", 
            "TX_SUCCESS_ADJUSTED_FINAL": "tx_success_adjusted_final", 
            "SWAP_FOR_FEES_FAILED": "swap_for_fees_failed",
            "FEES_COLLECT_ONLY_SUCCESS": "fees_collect_only_success", 
            "FEES_COLLECT_ONLY_FAILED": "fees_collect_only_failed" # [cite: 6]
        }
        self.metrics = self._reset_metrics() 
        self.factory_contract = None
        self.tick_spacing = None
        self.pool_address = None
        self.pool_contract = None
        self.token0_contract_instance = None
        self.token1_contract_instance = None
        
        self.nft_position_manager_address = None # [cite: 7]
        self.nft_manager_contract_for_events = None 

        self.token_manager_optimized_address = None
        self.token_manager_contract = None
        if TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV:
            try:
                self.token_manager_optimized_address = Web3.to_checksum_address(TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV)
                logger.info(f"TokenOperationsManagerOptimized address loaded for Baseline: {self.token_manager_optimized_address}")
            except ValueError: # [cite: 8]
                logger.error(f"Invalid TokenOperationsManagerOptimized address from ENV for Baseline: {TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV}") # [cite: 8]
        else:
            logger.warning("TOKEN_MANAGER_OPTIMIZED_ADDRESS env var not set for Baseline. Swap for fee generation will be skipped.") # [cite: 9]

    def _reset_metrics(self):
        base_metrics = super()._reset_metrics()
        baseline_specific_metrics = {
            'contract_type': 'Baseline',
            'action_taken': self.ACTION_STATES["INIT"] if hasattr(self, 'ACTION_STATES') else "init",
            'targetTickLower_offchain': None, 
            'targetTickUpper_offchain': None, 
            'currentTickLower_contract': None, # [cite: 10]
            'currentTickUpper_contract': None, # [cite: 10]
            'currentLiquidity_contract': None,
            'finalTickLower_contract': None, 
            'finalTickUpper_contract': None,
            'finalLiquidity_contract': None, 
            # New fields for explicit fee collection, initialized to None
            'fees_collected_token0_via_collect_only': None, # [cite: 11]
            'fees_collected_token1_via_collect_only': None # [cite: 11]
        }
        final_metrics = {**base_metrics, **baseline_specific_metrics}
        original_keys_defaults = { 
            'timestamp': None, 'tx_hash': None, 'range_width_multiplier_setting': None,
            'external_api_eth_price': None, 'actualPrice_pool': None, 
            'sqrtPriceX96_pool': None, 'currentTick_pool': None,
            'initial_contract_balance_token0': None, 'initial_contract_balance_token1': None, # [cite: 12]
            'amount0_provided_to_mint': None, 'amount1_provided_to_mint': None,
            'fees_collected_token0': None, 'fees_collected_token1': None, 
            'gas_used': None, 'gas_cost_eth': None, 'error_message': ""
        }
        for key, default_value in original_keys_defaults.items():
            if key not in final_metrics:
                final_metrics[key] = default_value # [cite: 13]
        return final_metrics

    def setup(self, desired_range_width_multiplier: int) -> bool: 
        if not super().setup(desired_range_width_multiplier):
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected(): # [cite: 14]
                logger.error("web3_utils.w3 not available in BaselineTest setup after base.setup()") # [cite: 14]
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup"; return False # [cite: 15]

            try:
                self.nft_position_manager_address = Web3.to_checksum_address(
                    self.contract.functions.positionManager().call() 
                )
                logger.info(f"BaselineTest.setup: Uniswap INonfungiblePositionManager address from OUR contract: {self.nft_position_manager_address}")
            except Exception as e_fetch_nfpm: # [cite: 16]
                logger.error(f"BaselineTest.setup: Failed to fetch INonfungiblePositionManager address from our contract: {e_fetch_nfpm}") # [cite: 16]
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"] # [cite: 16]
                self.metrics['error_message'] = f"Failed to get NFPM address: {str(e_fetch_nfpm)}" # [cite: 16]
                return False # [cite: 16]

            if self.nft_position_manager_address and self.nft_position_manager_address != ('0x' + '0'*40): # [cite: 17]
                nft_manager_abi_collect_event = [{"anonymous": False, "inputs": [{"indexed": True, "internalType": "uint256", "name": "tokenId", "type": "uint256"},{"indexed": False, "internalType": "address", "name": "recipient", "type": "address"},{"indexed": False, "internalType": "uint256", "name": "amount0", "type": "uint256"},{"indexed": False, "internalType": "uint256", "name": "amount1", "type": "uint256"}],"name": "Collect","type": "event"}] # [cite: 17]
                try:
                    self.nft_manager_contract_for_events = web3_utils.w3.eth.contract( # [cite: 18]
                        address=self.nft_position_manager_address,
                        abi=nft_manager_abi_collect_event
                    )
                    logger.info(f"BaselineTest.setup: Initialized contract instance for INonfungiblePositionManager Collect events at {self.nft_position_manager_address}.") # [cite: 19]
                except Exception as e_init_nft_events:
                    logger.error(f"BaselineTest.setup: Failed to create contract instance for NFPM Collect events using address '{self.nft_position_manager_address}': {e_init_nft_events}")
                    self.nft_manager_contract_for_events = None
            else:
                logger.warning(f"BaselineTest.setup: self.nft_position_manager_address is not validly set (value: {self.nft_position_manager_address}). Cannot initialize for NFPM Collect events.") # [cite: 20, 21]
                self.nft_manager_contract_for_events = None

            factory_address = self.contract.functions.factory().call()
            self.factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            if not self.factory_contract: self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = "Factory contract not found"; return False

            fee = self.contract.functions.fee().call()
            self.pool_address = self.factory_contract.functions.getPool(self.token0, self.token1, fee).call() # [cite: 22]
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Baseline pool address not found for {self.token0}/{self.token1} fee {fee}") 
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"; return False
            
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool") # [cite: 23]
            if not self.pool_contract: self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = "Pool contract object could not be created"; return False # [cite: 24]
            logger.info(f"Baseline Pool contract initialized at {self.pool_address}")
            
            self.tick_spacing = self.pool_contract.functions.tickSpacing().call() 
            if not self.tick_spacing or self.tick_spacing <= 0: 
                logger.error(f"Invalid tickSpacing read from pool: {self.tick_spacing}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = f"Invalid tickSpacing: {self.tick_spacing}"; return False # [cite: 25, 26]
            logger.info(f"Baseline Tick spacing from pool: {self.tick_spacing}")
            
            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Baseline contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier 
            
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key: # [cite: 27]
                logger.error("PRIVATE_KEY not set for setting rangeWidthMultiplier (Baseline).") # [cite: 27]
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = "PRIVATE_KEY missing for RWM setup (Baseline)"; return False # [cite: 28]
            
            tx_account = Account.from_key(private_key)
            tx_params_rwm = { 
                'from': tx_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(tx_account.address),
                'chainId': int(web3_utils.w3.net.version) # [cite: 29]
            }
            try:
                gas_estimate_rwm = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).estimate_gas({'from': tx_account.address})
                tx_params_rwm['gas'] = int(gas_estimate_rwm * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for setRangeWidthMultiplier (Baseline): {e}. Using default 200000.") # [cite: 30]
                tx_params_rwm['gas'] = 200000

            tx_set_rwm_build = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier)
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm_build.build_transaction(tx_params_rwm), private_key)
            
            if not receipt_rwm or receipt_rwm.status == 0:
                tx_hash_rwm_str = receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A' # [cite: 31]
                logger.error(f"Failed to set rangeWidthMultiplier for Baseline contract. TxHash: {tx_hash_rwm_str}") # [cite: 32]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";Failed to set RWM for Baseline (tx: {tx_hash_rwm_str})").strip(";") # [cite: 32]
                return False # [cite: 32]
            logger.info(f"rangeWidthMultiplier set successfully for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed: {e}") # [cite: 33]
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]; self.metrics['error_message'] = f"Setup error: {str(e)}"; return False # [cite: 33]

    # ==========================================================================================
    # === START OF MODIFIED SECTION ===
    # بازنویسی متد get_position_info برای کلاس BaselineTest جهت مدیریت صحیح ترتیب داده‌های پوزیشن
    # این متد فرض می‌کند که قرارداد BaselineMinimal تاپلی با ترتیب زیر برمی‌گرداند:
    # (tokenId, active_bool, tickLower, tickUpper, liquidity_uint)
    def get_position_info(self) -> dict | None:
        """Get current position info from the contract for BaselineMinimal."""
        if not self.contract:
            logger.error(f"({self.contract_name}) Contract not initialized. Run setup first (Baseline Override).")
            return None
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error(f"({self.contract_name}) Web3 not connected in get_position_info (Baseline Override).")
            return None
        
        try:
            pos_data = None
            # تلاش برای فراخوانی متدهای رایج برای دریافت اطلاعات پوزیشن
            if hasattr(self.contract.functions, 'currentPosition'):
                pos_data = self.contract.functions.currentPosition().call()
            elif hasattr(self.contract.functions, 'getCurrentPosition'):
                pos_data = self.contract.functions.getCurrentPosition().call()
            else:
                logger.error(f"({self.contract_name}) No known position info method (currentPosition, getCurrentPosition) found on contract (Baseline Override).")
                return None

            if pos_data:
                # بررسی فرمت مورد انتظار برای BaselineMinimal (5 عنصر)
                if len(pos_data) == 5:
                    # فرض ساختار بازگشتی برای BaselineMinimal:
                    # pos_data[0]: tokenId (uint256)
                    # pos_data[1]: active (bool)
                    # pos_data[2]: tickLower (int24)
                    # pos_data[3]: tickUpper (int24)
                    # pos_data[4]: liquidity (uint128)
                    position = {
                        'tokenId': pos_data[0],
                        'active': pos_data[1],    # active_status استخراج شده از ایندکس 1
                        'tickLower': pos_data[2],
                        'tickUpper': pos_data[3],
                        'liquidity': pos_data[4]  # liquidity عددی استخراج شده از ایندکس 4
                    }
                    logger.debug(f"({self.contract_name}) Fetched Position Info (Baseline Override): {position}")
                    return position
                else:
                    logger.error(f"({self.contract_name}) Position data format unexpected for BaselineMinimal (expected 5 elements). Data: {pos_data} (Baseline Override)")
                    return None
            else:
                logger.error(f"({self.contract_name}) Position data not found or empty for BaselineMinimal (Baseline Override).")
                return None

        except Exception as e:
            logger.exception(f"({self.contract_name}) Failed to get position info from contract (Baseline Override): {e}")
            return None
    # === END OF MODIFIED SECTION ===
    # ==========================================================================================

    def sqrt_price_x96_to_price_token0_in_token1(self, sqrt_price_x96_str: str) -> Decimal:
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        sqrt_price_x96 = Decimal(sqrt_price_x96_str)
        price_t1_in_t0 = (sqrt_price_x96 / Q96) ** 2 
        if price_t1_in_t0 == Decimal(0): return Decimal("inf") if sqrt_price_x96 != Decimal(0) else Decimal(0)
        price_t0_in_t1 = Decimal(1) / price_t1_in_t0 # [cite: 34]
        decimals_adjustment = Decimal(10) ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS)
        if decimals_adjustment == Decimal(0): return Decimal("-1") 
        return price_t0_in_t1 / decimals_adjustment

    def _estimate_liquidity(self, tick_lower: int, tick_upper: int) -> int:
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        try:
            if not self.token0_contract_instance and self.token0: self.token0_contract_instance = get_contract(self.token0, "IERC20")
            if not self.token1_contract_instance and self.token1: self.token1_contract_instance = get_contract(self.token1, "IERC20") # [cite: 35]
            if not self.token0_contract_instance or not self.token1_contract_instance or not self.pool_contract: logger.warning("Cannot estimate liquidity: token contracts or pool contract not initialized."); return 0 # [cite: 36]
            token0_bal = self.token0_contract_instance.functions.balanceOf(self.contract_address).call(); token1_bal = self.token1_contract_instance.functions.balanceOf(self.contract_address).call() # [cite: 37]
            slot0 = self.pool_contract.functions.slot0().call(); current_tick = slot0[1] # [cite: 38]
            if current_tick < tick_lower: return int(token0_bal) 
            elif current_tick >= tick_upper: return int(token1_bal) 
            else: return int((token0_bal + token1_bal) / 2) 
        except Exception as e: logger.error(f"Error estimating liquidity: {e}"); return 0 # [cite: 39]

    def get_pool_state(self) -> tuple[int | None, int | None]: # [cite: 40]
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if not self.pool_contract: logger.error("Pool contract not initialized for get_pool_state (Baseline)."); self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]; self.metrics['error_message'] = "Pool contract missing for get_pool_state"; return None, None # [cite: 41, 42]
        if not web3_utils.w3 or not web3_utils.w3.is_connected(): logger.error("Web3 not connected in get_pool_state (Baseline)."); self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]; self.metrics['error_message'] = "W3 not connected for get_pool_state" ; return None, None # [cite: 43, 44]
        try:
            slot0 = self.pool_contract.functions.slot0().call(); sqrt_price_x96, tick = slot0[0], slot0[1] # [cite: 45]
            self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96; self.metrics['currentTick_pool'] = tick # [cite: 46]
            self.metrics['actualPrice_pool'] = self.sqrt_price_x96_to_price_token0_in_token1(str(sqrt_price_x96))
            logger.info(f"Pool state read: Tick={tick}, SqrtPriceX96={sqrt_price_x96}, ActualPrice={self.metrics['actualPrice_pool']}") 
            return sqrt_price_x96, tick
        except Exception as e: logger.exception(f"Failed to get pool state (Baseline): {e}"); self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]; self.metrics['error_message'] = f"Pool state read error: {str(e)}"; return None, None # [cite: 47, 48]
    
    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]: # [cite: 49]
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if self.tick_spacing is None or current_tick is None: logger.error("Tick spacing or current_tick not available for target tick calculation (Baseline)."); self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]; self.metrics['error_message'] = "Missing data for target tick calc (Baseline)"; return None, None # [cite: 50, 51]
        try:
            width_multiplier = self.metrics.get('range_width_multiplier_setting')
            if width_multiplier is None or width_multiplier <= 0: width_multiplier = self.contract.functions.rangeWidthMultiplier().call()
            if width_multiplier is None or width_multiplier <= 0: logger.error(f"Invalid rangeWidthMultiplier ({width_multiplier}) for tick calculation."); self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]; self.metrics['error_message'] = f"Invalid RWM ({width_multiplier}) for tick calc"; return None, None # [cite: 52, 53]
            target_half_tick_width_base = 487 
            scaled_half_width = (target_half_tick_width_base * width_multiplier) // 100; tick_spacing_multiple = math.ceil(scaled_half_width / self.tick_spacing); half_total_tick_width = tick_spacing_multiple * self.tick_spacing # [cite: 54]
            if half_total_tick_width < self.tick_spacing: logger.warning(f"Calculated half width {half_total_tick_width} is less than tick spacing {self.tick_spacing}. Using minimum width."); half_total_tick_width = self.tick_spacing # [cite: 55]
            target_lower_tick = math.floor(current_tick / self.tick_spacing) * self.tick_spacing - half_total_tick_width
            target_upper_tick = math.ceil(current_tick / self.tick_spacing) * self.tick_spacing + half_total_tick_width
            target_lower_tick = max(MIN_TICK_CONST, target_lower_tick); target_upper_tick = min(MAX_TICK_CONST, target_upper_tick) # [cite: 56]
            if target_lower_tick >= target_upper_tick: logger.error(f"Invalid tick range calculated: [{target_lower_tick}, {target_upper_tick}]"); self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]; self.metrics['error_message'] = "Generated invalid tick range L >= U"; return None, None # [cite: 57, 58]
            logger.info("Target ticks calculation details:"); logger.info(f"  Current tick: {current_tick}"); logger.info(f"  Range width multiplier: {width_multiplier}"); logger.info(f"  Tick spacing: {self.tick_spacing}"); logger.info(f"  Half width in ticks: {half_total_tick_width}"); logger.info(f"  Target range: [{target_lower_tick}, {target_upper_tick}]") # [cite: 59, 60]
            self.metrics['targetTickLower_offchain'] = target_lower_tick; self.metrics['targetTickUpper_offchain'] = target_upper_tick # [cite: 61]
            return target_lower_tick, target_upper_tick
        except Exception as e: logger.exception(f"Error calculating target ticks off-chain (Baseline): {e}"); self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]; self.metrics['error_message'] = f"Target tick calc error (Baseline): {str(e)}"; return None, None # [cite: 62, 63]

    # ADDED THIS METHOD
    def _get_contract_token_balances_individually(self):
        balance0_wei, balance1_wei = 0, 0
        try:
            if self.token0:
                token0_contract_instance = web3_utils.get_contract(self.token0, "IERC20")
                if token0_contract_instance: balance0_wei = token0_contract_instance.functions.balanceOf(self.contract_address).call()
            if self.token1:
                token1_contract_instance = web3_utils.get_contract(self.token1, "IERC20") # [cite: 64]
                if token1_contract_instance: balance1_wei = token1_contract_instance.functions.balanceOf(self.contract_address).call() # [cite: 64]
        except Exception as e: logger.warning(f"Could not read contract token balances individually (Baseline): {e}")
        return balance0_wei, balance1_wei

    def _perform_swap_for_fees(self, funding_account, private_key_env, swap_token_in_addr: str, swap_token_out_addr: str, swap_amount_readable: Decimal, token_in_decimals: int, token_out_decimals: int):
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if not self.token_manager_optimized_address: logger.warning("TokenManagerOptimized address not set for Baseline. Skipping swap for fees."); return False # [cite: 65, 66]
        if not self.token_manager_contract: self.token_manager_contract = web3_utils.get_contract(self.token_manager_optimized_address, "TokenOperationsManagerOptimized")
        if not self.token_manager_contract: logger.error("Failed to get TokenOperationsManagerOptimized contract instance for Baseline."); return False
        logger.info(f"Attempting swap for Baseline: {swap_amount_readable} of {swap_token_in_addr} for {swap_token_out_addr} via TokenManager.")
        try:
            pool_fee_for_swap = self.contract.functions.fee().call(); amount_to_swap_wei = int(swap_amount_readable * (Decimal(10) ** token_in_decimals))
            token_in_contract_for_approve = web3_utils.get_contract(swap_token_in_addr, "IERC20") # [cite: 67]
            logger.info(f"Approving TokenManagerOptimized ({self.token_manager_optimized_address}) to spend {swap_amount_readable} of token {swap_token_in_addr} from deployer {funding_account.address}...")
            current_nonce_approve = web3_utils.w3.eth.get_transaction_count(funding_account.address); approve_tx_params = {'from': funding_account.address, 'nonce': current_nonce_approve} # [cite: 68]
            try: gas_est_approve = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).estimate_gas({'from': funding_account.address}); approve_tx_params['gas'] = int(gas_est_approve * 1.2) # [cite: 69]
            except Exception as e: logger.warning(f"Gas estimation for TokenManager approval (Baseline) failed: {e}. Using default 100,000."); approve_tx_params['gas'] = 100000 # [cite: 70]
            built_approve_tx = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).build_transaction(approve_tx_params)
            receipt_approve = web3_utils.send_transaction(built_approve_tx, private_key_env)
            if not receipt_approve or receipt_approve.status == 0: logger.error(f"Failed to approve TokenManagerOptimized (Baseline). Receipt: {receipt_approve}"); return False # [cite: 71]
            logger.info(f"TokenManagerOptimized approved for Baseline. Tx: {receipt_approve.transactionHash.hex()}")
            current_nonce_swap = web3_utils.w3.eth.get_transaction_count(funding_account.address); logger.info(f"Calling swap on TokenManagerOptimized for Baseline: {swap_amount_readable} {swap_token_in_addr} -> {swap_token_out_addr}, fee {pool_fee_for_swap}...") # [cite: 72]
            swap_tx_params = {'from': funding_account.address, 'nonce': current_nonce_swap }
            try:
                checksum_token_in = Web3.to_checksum_address(swap_token_in_addr); checksum_token_out = Web3.to_checksum_address(swap_token_out_addr) # [cite: 73]
                gas_est_swap = self.token_manager_contract.functions.swap(checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0).estimate_gas({'from': funding_account.address})
                swap_tx_params['gas'] = int(gas_est_swap * 1.30) 
            except ContractLogicError as cle: logger.error(f"Gas estimation for TokenManager swap (Baseline) failed due to contract logic: {cle}."); return False # [cite: 74]
            except Exception as e: logger.warning(f"Gas estimation for TokenManager swap (Baseline) failed: {e}. Using default 700,000."); swap_tx_params['gas'] = 700000 # [cite: 75]
            built_swap_tx = self.token_manager_contract.functions.swap(checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0).build_transaction(swap_tx_params)
            receipt_swap = web3_utils.send_transaction(built_swap_tx, private_key_env)
            if receipt_swap and receipt_swap.status == 1:
                logger.info(f"Swap via TokenManagerOptimized for Baseline successful. Tx: {receipt_swap.transactionHash.hex()}")
                swap_logs = self.token_manager_contract.events.Operation().process_receipt(receipt_swap, errors=DISCARD)
                for log_entry in swap_logs: # [cite: 76]
                    op_type_bytes32 = log_entry.args.opType; is_swap_op = (web3_utils.w3.to_hex(op_type_bytes32) == web3_utils.w3.to_hex(web3_utils.w3.solidity_keccak(['string'],['SWAP']))) # [cite: 77]
                    if is_swap_op:
                        decimals_for_amount_out = self.token0_decimals if Web3.to_checksum_address(log_entry.args.tokenB) == Web3.to_checksum_address(self.token0) else self.token1_decimals
                        amount_out_readable = Decimal(log_entry.args.amount) / (Decimal(10) ** decimals_for_amount_out)
                        logger.info(f"TokenManager Swap Event (Baseline context): TokenIn={log_entry.args.tokenA}, TokenOut={log_entry.args.tokenB}, AmountOut={amount_out_readable:.6f}") # [cite: 78]
                return True
            else: logger.error(f"Swap via TokenManagerOptimized for Baseline failed. Receipt: {receipt_swap}"); return False # [cite: 79]
        except Exception as e: logger.exception(f"Error during _perform_swap_for_fees for Baseline: {e}"); return False # [cite: 80]

    # ADDED: _call_collect_fees_only method for BaselineTest
    def _call_collect_fees_only(self, funding_account, private_key_env) -> bool:
        logger.info("Attempting to call collectCurrentFeesOnly() on BaselineMinimal contract...")
        if not self.contract:
            logger.error("BaselineMinimal contract instance not available for collectCurrentFeesOnly."); return False # [cite: 81]
        
        if self.metrics.get('fees_collected_token0') is None: self.metrics['fees_collected_token0'] = 0
        if self.metrics.get('fees_collected_token1') is None: self.metrics['fees_collected_token1'] = 0
        if 'fees_collected_token0_via_collect_only' in self.metrics and self.metrics.get('fees_collected_token0_via_collect_only') is None:
            self.metrics['fees_collected_token0_via_collect_only'] = 0
        if 'fees_collected_token1_via_collect_only' in self.metrics and self.metrics.get('fees_collected_token1_via_collect_only') is None:
            self.metrics['fees_collected_token1_via_collect_only'] = 0

        tx_call = self.contract.functions.collectCurrentFeesOnly() # [cite: 82]
        current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
        tx_params = {'from': funding_account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
        try:
            tx_params['gas'] = int(tx_call.estimate_gas({'from': funding_account.address}) * 1.25)
        except ContractLogicError as cle:
            logger.error(f"Gas estimation for collectCurrentFeesOnly (Baseline) failed (Logic Error): {cle}. This might indicate an issue like no active position or other require failure.") # [cite: 83]
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";CollectFeesGasEstLogicError(Baseline): {str(cle)[:100]}").strip(';') # [cite: 83]
            return False # [cite: 83]
        except Exception as e:
            tx_params['gas'] = 300000 
            logger.warning(f"Gas estimation for collectCurrentFeesOnly (Baseline) failed: {e}. Using default {tx_params['gas']}.")

        try:
            built_tx = tx_call.build_transaction(tx_params) # [cite: 84]
            receipt = web3_utils.send_transaction(built_tx, private_key_env)

            if receipt and receipt.status == 1:
                logger.info(f"Baseline collectCurrentFeesOnly() transaction successful. Tx: {receipt.transactionHash.hex()}")
                if "FEES_COLLECT_ONLY_SUCCESS" in self.ACTION_STATES: self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_SUCCESS"]
                
                amount0_from_our_event, amount1_from_our_event = 0,0 # [cite: 85]
                our_fee_event_logs = self.contract.events.FeesOnlyCollected().process_receipt(receipt, errors=DISCARD)
                if our_fee_event_logs:
                    log_args = our_fee_event_logs[0].args
                    logger.info(f"BaselineContract FeesOnlyCollected Event: TokenId={log_args.tokenId}, Amount0={log_args.amount0Collected}, Amount1={log_args.amount1Collected}, Success={log_args.success}") # [cite: 86]
                    if log_args.success: amount0_from_our_event = log_args.amount0Collected; amount1_from_our_event = log_args.amount1Collected # [cite: 87]
                    else: logger.warning("Baseline FeesOnlyCollected event reported success=false from contract.") # [cite: 87]
                else: logger.warning("No FeesOnlyCollected event found from Baseline contract after collectCurrentFeesOnly call.")

                # Populate specific and main fee fields
                if 'fees_collected_token0_via_collect_only' in self.metrics: self.metrics['fees_collected_token0_via_collect_only'] = amount0_from_our_event
                if 'fees_collected_token1_via_collect_only' in self.metrics: self.metrics['fees_collected_token1_via_collect_only'] = amount1_from_our_event # [cite: 88]
                self.metrics['fees_collected_token0'] = amount0_from_our_event
                self.metrics['fees_collected_token1'] = amount1_from_our_event
                logger.info(f"Baseline Main fee metrics updated: T0={self.metrics['fees_collected_token0']}, T1={self.metrics['fees_collected_token1']}")

                if self.nft_manager_contract_for_events: 
                    logger.info(f"Checking Uniswap NFPM Collect events in Tx: {receipt.transactionHash.hex()} after Baseline collectCurrentFeesOnly call...") # [cite: 89]
                    collect_logs_nfpm = self.nft_manager_contract_for_events.events.Collect().process_receipt(receipt, errors=DISCARD)
                    found_nfpm_collect = False
                    for nfpm_log_entry in collect_logs_nfpm:
                        if nfpm_log_entry.args.recipient.lower() == self.contract_address.lower(): # [cite: 90]
                            found_nfpm_collect = True
                            logger.info(f"Uniswap NFPM Collect Event (Post Baseline collectCurrentFeesOnly, TokenId={nfpm_log_entry.args.tokenId}): Amount0={nfpm_log_entry.args.amount0}, Amount1={nfpm_log_entry.args.amount1}") # [cite: 90]
                            if not our_fee_event_logs or not our_fee_event_logs[0].args.success : # [cite: 91]
                                if 'fees_collected_token0_via_collect_only' in self.metrics: self.metrics['fees_collected_token0_via_collect_only'] = nfpm_log_entry.args.amount0 # [cite: 91]
                                if 'fees_collected_token1_via_collect_only' in self.metrics: self.metrics['fees_collected_token1_via_collect_only'] = nfpm_log_entry.args.amount1 # [cite: 92]
                                self.metrics['fees_collected_token0'] = nfpm_log_entry.args.amount0 # [cite: 92]
                                self.metrics['fees_collected_token1'] = nfpm_log_entry.args.amount1 # [cite: 92]
                                logger.info("Updated ALL Baseline fee metrics from NFPM event due to issue/absence of contract's event.") # [cite: 93]
                            elif log_args.amount0Collected != nfpm_log_entry.args.amount0 or log_args.amount1Collected != nfpm_log_entry.args.amount1:
                                logger.warning(f"Discrepancy (Baseline): OurEvent(0:{log_args.amount0Collected}, 1:{log_args.amount1Collected}) vs NFPMEvent(0:{nfpm_log_entry.args.amount0}, 1:{nfpm_log_entry.args.amount1}).")
                                break # [cite: 94]
                    if not found_nfpm_collect: logger.info("No direct Uniswap NFPM Collect event found with Baseline contract as recipient in this transaction.")
                return True
            else:
                tx_hash_str = receipt.transactionHash.hex() if receipt else "N/A"; logger.error(f"Baseline collectCurrentFeesOnly() transaction failed. Tx: {tx_hash_str}") # [cite: 95]
                if "FEES_COLLECT_ONLY_FAILED" in self.ACTION_STATES: self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";BaselineCollectFeesTxFailed: {tx_hash_str}").strip(';'); return False # [cite: 96]
        except Exception as e:
            logger.exception(f"Exception during Baseline _call_collect_fees_only: {e}")
            if "FEES_COLLECT_ONLY_FAILED" in self.ACTION_STATES: self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";BaselineCollectFeesException: {str(e)[:100]}").strip(';'); return False # [cite: 97]

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        # YOUR ORIGINAL adjust_position LOGIC, with _get_contract_token_balances_individually and stage_results
        self.metrics = self._reset_metrics() 
        try:
            self.get_current_eth_price()
            current_rwm = self.contract.functions.rangeWidthMultiplier().call() 
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception: logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.") # [cite: 98]

        # Renamed flags to use stage_results dictionary for clarity
        stage_results = {
            'initial_adjustment': False,
            'swap': False,
            'collect_only': False, # For explicit fee collect
            'final_adjustment': False
        }
        
        private_key_env = os.getenv('PRIVATE_KEY') # [cite: 99]
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position (Baseline)."); self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]; self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx (Baseline)"; self.save_metrics(); return False # [cite: 100, 101]
        funding_account = Account.from_key(private_key_env)
        
        pos_info_before_all = self.get_position_info() # This will now call the overridden get_position_info for BaselineTest
        if pos_info_before_all:
            self.metrics['currentTickLower_contract'] = pos_info_before_all.get('tickLower',0)
            self.metrics['currentTickUpper_contract'] = pos_info_before_all.get('tickUpper',0)
            self.metrics['currentLiquidity_contract'] = pos_info_before_all.get('liquidity', 0) # This should now get the correct numerical liquidity
        else: self.metrics['currentTickLower_contract'] = 0; self.metrics['currentTickUpper_contract'] = 0; self.metrics['currentLiquidity_contract'] = 0 # [cite: 102]

        # --- STAGE 1: Initial Position Adjustment ---
        logger.info("\n--- STAGE 1: Baseline Strategy - Initial position adjustment ---")
        try:
            _, current_tick_initial = self.get_pool_state() 
            if current_tick_initial is None: self.save_metrics(); return False # [cite: 103]
            target_lower_tick_initial, target_upper_tick_initial = self.calculate_target_ticks_offchain(current_tick_initial)
            if target_lower_tick_initial is None or target_upper_tick_initial is None: self.save_metrics(); return False # [cite: 104]
            
            current_pos_info_for_check = self.get_position_info() # This will also call the overridden method
            current_pos_active_stage1 = current_pos_info_for_check.get('active', False) if current_pos_info_for_check else False
            current_lower_for_check = current_pos_info_for_check.get('tickLower') if current_pos_info_for_check else None
            current_upper_for_check = current_pos_info_for_check.get('tickUpper') if current_pos_info_for_check else None

            if current_pos_active_stage1 and current_lower_for_check is not None and current_upper_for_check is not None and self.tick_spacing is not None and \
               target_lower_tick_initial == current_lower_for_check and target_upper_tick_initial == current_upper_for_check: # [cite: 105]
                logger.info(f"Baseline: Target ticks ({target_lower_tick_initial}, {target_upper_tick_initial}) match current on-chain. Skipping initial on-chain adjustment call.") 
                self.metrics['action_taken'] = self.ACTION_STATES["SKIPPED_PROXIMITY"]
                self.metrics['finalTickLower_contract'] = current_lower_for_check
                self.metrics['finalTickUpper_contract'] = current_upper_for_check # [cite: 106]
                self.metrics['finalLiquidity_contract'] = current_pos_info_for_check.get('liquidity',0) if current_pos_info_for_check else 0 # Correct usage
                stage_results['initial_adjustment'] = True # [cite: 106]
            else:
                logger.info("Ensuring precise token balances for Baseline contract (initial)...")
                if not contract_funder.ensure_precise_token_balances( # [cite: 107]
                    self.contract_address, self.token0, self.token0_decimals, target_usdc_balance,
                    self.token1, self.token1_decimals, target_weth_balance, private_key_env
                ):
                    logger.error("Precise funding for Baseline contract failed (initial)."); self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]; self.metrics['error_message'] = "Precise contract funding failed (Baseline initial)"; self.save_metrics(); return False # [cite: 108, 109]
                
                balance0_val, balance1_val = self._get_contract_token_balances_individually() # ADDED THIS CALL
                self.metrics['initial_contract_balance_token0'] = balance0_val
                self.metrics['initial_contract_balance_token1'] = balance1_val
                logger.info(f"Initial contract balances for baseline metrics (after funding): Token0={balance0_val}, Token1={balance1_val}") # [cite: 110]

                logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract (initial)...")
                tx_function_call_base_initial = self.contract.functions.adjustLiquidityWithCurrentPrice()
                current_nonce_initial_adjust = web3_utils.w3.eth.get_transaction_count(funding_account.address) 
                tx_params_base_initial = {'from': funding_account.address, 'nonce': current_nonce_initial_adjust, 'chainId': int(web3_utils.w3.net.version)}
                try: gas_estimate_base_initial = tx_function_call_base_initial.estimate_gas({'from': funding_account.address}); tx_params_base_initial['gas'] = int(gas_estimate_base_initial * 1.25) # [cite: 111]
                except Exception as est_err_base: logger.warning(f"Gas estimation for 'adjustLiquidityWithCurrentPrice' (initial) failed: {est_err_base}. Using default 1,500,000"); tx_params_base_initial['gas'] = 1500000 # [cite: 112]
                built_tx_base_initial = tx_function_call_base_initial.build_transaction(tx_params_base_initial)
                receipt_base_initial = web3_utils.send_transaction(built_tx_base_initial, private_key_env)
                self.metrics['tx_hash'] = receipt_base_initial.transactionHash.hex() if receipt_base_initial else None
                
                if receipt_base_initial and receipt_base_initial.status == 1:
                    logger.info(f"Baseline initial adjustment transaction successful. Tx: {self.metrics['tx_hash']}.") # [cite: 113]
                    self.metrics['gas_used'] = receipt_base_initial.get('gasUsed', 0) 
                    eff_gas_price_base_initial = receipt_base_initial.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                    self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * eff_gas_price_base_initial, 'ether'))
                    stage_results['initial_adjustment'] = True # [cite: 114]
                    adj_metrics_logs_initial = self.contract.events.BaselineAdjustmentMetrics().process_receipt(receipt_base_initial, errors=DISCARD)
                    was_adjusted_onchain_initial = False
                    if adj_metrics_logs_initial:
                        was_adjusted_onchain_initial = adj_metrics_logs_initial[0].args.adjusted # [cite: 115]
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"] if was_adjusted_onchain_initial else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"] # [cite: 115]
                        self.metrics['targetTickLower_offchain'] = adj_metrics_logs_initial[0].args.targetTickLower 
                        self.metrics['targetTickUpper_offchain'] = adj_metrics_logs_initial[0].args.targetTickUpper
                    else: self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"] # [cite: 116]
                    if was_adjusted_onchain_initial:
                        mint_logs_base_initial = self.contract.events.PositionMinted().process_receipt(receipt_base_initial, errors=DISCARD) 
                        if mint_logs_base_initial: 
                            log_entry = mint_logs_base_initial[0] # [cite: 117]
                            if hasattr(log_entry.args, 'amount0Actual') and hasattr(log_entry.args, 'amount1Actual'): 
                                self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual 
                                self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual # [cite: 118]
                                self.metrics['finalTickLower_contract'] = log_entry.args.tickLower 
                                self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper 
                                self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity # This reads from event, should be fine [cite: 119]
                            else: logger.error(f"Baseline PositionMinted (initial) event missing amount0Actual/amount1Actual attributes.") # [cite: 119]
                        else: logger.warning("Baseline initial tx successful & adjusted, but no PositionMinted event found.") # [cite: 120]
                    else: 
                        logger.info("Baseline initial adjustment call successful but skipped on-chain position change.")
                        pos_info_after_skip = self.get_position_info(); # Calls overridden method [cite: 121]
                        if pos_info_after_skip: # [cite: 121]
                            self.metrics['finalTickLower_contract'] = pos_info_after_skip.get('tickLower',0) # [cite: 121]
                            self.metrics['finalTickUpper_contract'] = pos_info_after_skip.get('tickUpper',0) # [cite: 121]
                            self.metrics['finalLiquidity_contract'] = pos_info_after_skip.get('liquidity',0) # Correct usage [cite: 121]
                        self.metrics['amount0_provided_to_mint'] = 0; self.metrics['amount1_provided_to_mint'] = 0 # [cite: 122, 123]
                elif receipt_base_initial: 
                    logger.error(f"Baseline initial adjustment transaction reverted. Tx: {self.metrics['tx_hash']}"); self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]; self.metrics['error_message'] = "tx_reverted_onchain (Baseline initial)"; self.metrics['gas_used'] = receipt_base_initial.get('gasUsed', 0); self.save_metrics(); return False # [cite: 124, 125]
                else: 
                    logger.error("Baseline initial adjustment transaction sending/receipt failed."); self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]; self.metrics['error_message'] = (self.metrics.get('error_message',"") + ";send_transaction for baseline initial adjustment failed").strip(";"); self.save_metrics(); return False # [cite: 126, 127]
        except Exception as tx_err_initial: 
            logger.exception(f"Error during baseline initial adjustment: {tx_err_initial}"); self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]; self.metrics['error_message'] = f"TxError (Baseline initial): {str(tx_err_initial)}"; self.save_metrics(); return False # [cite: 128, 129]
        
        # --- STAGE 2: Perform Swap ---
        if stage_results['initial_adjustment']:
            logger.info("\n--- STAGE 2: Baseline Strategy - Performing swap for fees ---")
            swap_token_in_addr = self.token1; swap_token_out_addr = self.token0; swap_amount_readable = Decimal("0.05") # [cite: 130]
            token_in_decimals_for_swap = self.token1_decimals; token_out_decimals_for_swap = self.token0_decimals # [cite: 131]
            stage_results['swap'] = self._perform_swap_for_fees(funding_account, private_key_env, swap_token_in_addr, swap_token_out_addr, swap_amount_readable, token_in_decimals_for_swap, token_out_decimals_for_swap) # [cite: 131]
            if stage_results['swap']: self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]
            else: self.metrics['action_taken'] = self.ACTION_STATES["SWAP_FOR_FEES_FAILED"]; self.metrics['error_message'] = (self.metrics.get('error_message',"") + ";Swap for fees failed or skipped").strip(";") # [cite: 132]

        # --- STAGE 2.5: Explicitly Collect Fees ---
        if stage_results['initial_adjustment'] and stage_results['swap']:
            logger.info("\n--- STAGE 2.5: Baseline Strategy - Explicit Fee Collection ---")
            stage_results['collect_only'] = self._call_collect_fees_only(funding_account, private_key_env)
        else:
            logger.info("Skipping Stage 2.5 (Explicit Fee Collection for Baseline) due to previous stage failure(s).") # [cite: 133]
            stage_results['collect_only'] = True # [cite: 133]

        # --- STAGE 3: Second Position Adjustment ---
        if stage_results['initial_adjustment'] and stage_results['swap'] and stage_results['collect_only']:
            logger.info("\n--- STAGE 3: Baseline Strategy - Second position adjustment (fee collection) ---")
            try:
                logger.info("Ensuring precise token balances for Baseline contract (final)...") # [cite: 134]
                if not contract_funder.ensure_precise_token_balances(
                    self.contract_address, self.token0, self.token0_decimals, target_usdc_balance, 
                    self.token1, self.token1_decimals, target_weth_balance, private_key_env
                ): logger.error("Precise funding for Baseline contract failed (final)."); self.metrics['error_message'] = (self.metrics.get('error_message',"") + ";Funding failed (final)").strip(";") # [cite: 135]
                
                balance0_before_final_adj, balance1_before_final_adj = self._get_contract_token_balances_individually() # Corrected
                logger.info(f"Balances before final adjustLiquidityWithCurrentPrice (Baseline): Token0={balance0_before_final_adj}, Token1={balance1_before_final_adj}")

                logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract (final)...")
                tx_function_call_base_final = self.contract.functions.adjustLiquidityWithCurrentPrice() # [cite: 136]
                current_nonce_final_adjust = web3_utils.w3.eth.get_transaction_count(funding_account.address) 
                tx_params_base_final = {'from': funding_account.address, 'nonce': current_nonce_final_adjust, 'chainId': int(web3_utils.w3.net.version)}
                try: gas_estimate_base_final = tx_function_call_base_final.estimate_gas({'from': funding_account.address}); tx_params_base_final['gas'] = int(gas_estimate_base_final * 1.25) # [cite: 137]
                except Exception as est_err_base_final: logger.warning(f"Gas estimation for 'adjustLiquidityWithCurrentPrice' (final) failed: {est_err_base_final}. Using default 1,500,000"); tx_params_base_final['gas'] = 1500000 # [cite: 138]
                built_tx_base_final = tx_function_call_base_final.build_transaction(tx_params_base_final)
                receipt_base_final = web3_utils.send_transaction(built_tx_base_final, private_key_env)
                self.metrics['tx_hash'] = receipt_base_final.transactionHash.hex() if receipt_base_final else self.metrics.get('tx_hash')

                if receipt_base_final and receipt_base_final.status == 1:
                    logger.info(f"Baseline final adjustment transaction successful. Tx: {self.metrics['tx_hash']}.") # [cite: 139]
                    self.metrics['gas_used'] += receipt_base_final.get('gasUsed', 0) 
                    eff_gas_price_base_final = receipt_base_final.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                    self.metrics['gas_cost_eth'] += float(Web3.from_wei(receipt_base_final.get('gasUsed', 0) * eff_gas_price_base_final, 'ether')) 
                    stage_results['final_adjustment'] = True # Changed from second_adjustment_call_success # [cite: 140]
                    
                    adj_metrics_logs_final = self.contract.events.BaselineAdjustmentMetrics().process_receipt(receipt_base_final, errors=DISCARD)
                    was_adjusted_onchain_final = False
                    if adj_metrics_logs_final:
                        was_adjusted_onchain_final = adj_metrics_logs_final[0].args.adjusted # [cite: 141]
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_FINAL"] if was_adjusted_onchain_final else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"] # [cite: 141]
                        self.metrics['targetTickLower_offchain'] = adj_metrics_logs_final[0].args.targetTickLower
                        self.metrics['targetTickUpper_offchain'] = adj_metrics_logs_final[0].args.targetTickUpper # [cite: 142]
                    else: self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_FINAL"] # [cite: 142]
                    if was_adjusted_onchain_final:
                        mint_logs_base_final = self.contract.events.PositionMinted().process_receipt(receipt_base_final, errors=DISCARD)
                        if mint_logs_base_final:
                            log_entry = mint_logs_base_final[0] # [cite: 143]
                            self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual 
                            self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual 
                            self.metrics['finalTickLower_contract'] = log_entry.args.tickLower # [cite: 144]
                            self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper # [cite: 144]
                            self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity # Reads from event [cite: 144]
                        else: logger.warning("Baseline final tx successful & adjusted, but no PositionMinted event found.") # [cite: 145]
                    else: logger.info("Baseline final adjustment call successful but skipped on-chain position change.")
                    
                    # Check NFPM Collect event if a full remove/add happened during this final adjustment
                    if self.nft_manager_contract_for_events and was_adjusted_onchain_final: # [cite: 146]
                        logger.info(f"Checking Uniswap NFPM Collect events for fees in FINAL Baseline Tx: {receipt_base_final.transactionHash.hex()} ...") # [cite: 146]
                        collect_logs_nfpm_final = self.nft_manager_contract_for_events.events.Collect().process_receipt(receipt_base_final, errors=DISCARD)
                        if not collect_logs_nfpm_final: logger.info(f"No NFPM Collect events found in FINAL Baseline Tx.") # [cite: 147]
                        for nfpm_log_entry in collect_logs_nfpm_final:
                            if nfpm_log_entry.args.recipient.lower() == self.contract_address.lower():
                                logger.info(f"Uniswap NFPM Collect Event (FINAL Baseline Tx, TokenId={nfpm_log_entry.args.tokenId}): Amount0={nfpm_log_entry.args.amount0}, Amount1={nfpm_log_entry.args.amount1}") # [cite: 148]
                                # Overwrite main fee fields if this collect is more comprehensive
                                self.metrics['fees_collected_token0'] = nfpm_log_entry.args.amount0
                                self.metrics['fees_collected_token1'] = nfpm_log_entry.args.amount1 # [cite: 149]
                                logger.info(f"Main fee metrics (Baseline fees_collected_token0/1) updated from FINAL Tx NFPM Collect: ({self.metrics['fees_collected_token0']}, {self.metrics['fees_collected_token1']})") # [cite: 149]
                                break # [cite: 150]
                elif receipt_base_final: 
                    logger.error(f"Baseline final adjustment transaction reverted. Tx: {self.metrics['tx_hash']}"); self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]; self.metrics['error_message'] = (self.metrics.get('error_message',"") + ";Final adjustment reverted (Baseline)").strip(";"); self.metrics['gas_used'] += receipt_base_final.get('gasUsed', 0); stage_results['final_adjustment'] = False # [cite: 151]
                else: 
                    logger.error("Baseline final adjustment transaction sending/receipt failed."); 
                    if self.metrics['action_taken'] != self.ACTION_STATES["TX_REVERTED"]: self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                    self.metrics['error_message'] = (self.metrics.get('error_message',"") + ";Final adjustment send failed (Baseline)").strip(";"); stage_results['final_adjustment'] = False # [cite: 152]
            except Exception as tx_err_final:
                logger.exception(f"Error during baseline final adjustment: {tx_err_final}")
                if self.metrics['action_taken'] == self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]: self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = (self.metrics.get('error_message',"") + f";TxError (Baseline final): {str(tx_err_final)}").strip(";"); stage_results['final_adjustment'] = False # [cite: 153]
        else:
            logger.info("Skipping Stage 3 (Baseline Final Position Adjustment) due to previous stage failure(s).")
            if stage_results['initial_adjustment'] and stage_results['swap'] and not stage_results['collect_only']: stage_results['final_adjustment'] = False
            else: stage_results['final_adjustment'] = True 

        logger.info("Updating final position info in 'finally' block for Baseline (absolute final state)...") 
        final_pos_info = self.get_position_info() # Calls overridden method [cite: 154]
        if final_pos_info:  
            self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower',0) 
            self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper',0) 
            self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity',0) # Correct usage
        else: logger.warning("Baseline: Could not get final position info from contract in finally block.") 
        self.save_metrics() 
        
        if not stage_results['initial_adjustment']: logger.error("Overall Baseline Test Failed: Initial adjustment did not succeed."); return False # [cite: 155, 156]
        if self.token_manager_optimized_address: 
            if not stage_results['swap']: logger.error("Overall Baseline Test Failed: Swap stage (intended) did not succeed."); return False # [cite: 157]
            if not stage_results['collect_only']: logger.error("Overall Baseline Test Failed: Explicit fee collection (intended) did not succeed."); return False # [cite: 158]
        if not stage_results['final_adjustment']: logger.error("Overall Baseline Test Failed: Final adjustment did not succeed."); return False # [cite: 159]
        
        logger.info("All intended Baseline stages reported success.")
        return True

    def save_metrics(self): 
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_value = self.metrics.get('actualPrice_pool') 
        if isinstance(metric_value, Decimal): self.metrics['actualPrice_pool'] = f"{metric_value}"
        elif metric_value is None: self.metrics['actualPrice_pool'] = ""
        
        columns = [ # [cite: 160]
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'range_width_multiplier_setting', 'external_api_eth_price',
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_offchain', 'targetTickUpper_offchain',
            'initial_contract_balance_token0', 'initial_contract_balance_token1',
            'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
            'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract', # [cite: 161]
            'amount0_provided_to_mint', 'amount1_provided_to_mint',
            'fees_collected_token0', 'fees_collected_token1',
            'gas_used', 'gas_cost_eth', 'error_message'
        ] # [cite: 162]
        
        row_data = {}
        for col in columns:
            val = self.metrics.get(col)
            if val is None:
                if col in ['sqrtPriceX96_pool', 'currentTick_pool', 'targetTickLower_offchain', 'targetTickUpper_offchain',
                           'initial_contract_balance_token0', 'initial_contract_balance_token1', # [cite: 163]
                           'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
                           'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract',
                           'amount0_provided_to_mint', 'amount1_provided_to_mint',
                           'fees_collected_token0', 'fees_collected_token1', # [cite: 164]
                           'fees_collected_token0_via_collect_only', 'fees_collected_token1_via_collect_only',
                           'gas_used', 'gas_cost_eth', 'range_width_multiplier_setting']:
                    row_data[col] = 0 if col not in ['external_api_eth_price'] else "" # [cite: 165]
                elif col not in ['tx_hash', 'error_message', 'action_taken', 'contract_type', 'timestamp', 'actualPrice_pool', 'external_api_eth_price']:
                     row_data[col] = ""
                elif self.metrics.get(col) is None :
                    row_data[col] = "" # [cite: 166]
            else:
                if col == 'external_api_eth_price' and isinstance(val, float): row_data[col] = f"{val:.2f}"
                elif col == 'gas_cost_eth' and isinstance(val, float): row_data[col] = f"{val:.18f}" 
                else: row_data[col] = val
        
        try:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True) # [cite: 167]
            file_exists = RESULTS_FILE.is_file() 
            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f: 
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore') 
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0: writer.writeheader() 
                writer.writerow(row_data) # [cite: 168]
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}") # [cite: 168]
        except Exception as e: 
            logger.exception(f"Failed to save baseline metrics: {e}") 

def main(): 
    logger.info("="*50); logger.info("Starting Baseline Minimal Liquidity Manager Test on Fork"); logger.info("="*50) # [cite: 169]
    if not web3_utils.init_web3(): logger.critical("Web3 initialization failed. Exiting baseline test."); sys.exit(1) # [cite: 170]
    if not web3_utils.w3 or not web3_utils.w3.is_connected(): logger.critical("web3_utils.w3 instance not available or not connected after init. Exiting baseline test."); sys.exit(1) # [cite: 171]

    baseline_address_val = None  
    test_baseline = None 
    try:
        baseline_address_env = os.getenv('BASELINE_MINIMAL_ADDRESS')
        if not baseline_address_env :
            logger.error(f"BASELINE_MINIMAL_ADDRESS not found in environment variables.")
            temp_test_b_error = BaselineTest("0x"+"0"*40) 
            temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
            temp_test_b_error.metrics['error_message'] = "BASELINE_MINIMAL_ADDRESS env var not found" # [cite: 172]
            temp_test_b_error.save_metrics(); raise ValueError("BASELINE_MINIMAL_ADDRESS env var not found") # [cite: 173]
        
        baseline_address_val = Web3.to_checksum_address(baseline_address_env)
        logger.info(f"Loaded Baseline Minimal Address from ENV: {baseline_address_val}") 
        
        test_baseline = BaselineTest(baseline_address_val) 
        
        desired_rwm_base = int(os.getenv('BASELINE_RWM', '100')) # Matching default from your logs
        # MODIFIED: Increased default target balances to match Predictive
        target_weth_for_test = float(os.getenv('BASELINE_TARGET_WETH', '5.0')) # [cite: 174]
        target_usdc_for_test = float(os.getenv('BASELINE_TARGET_USDC', '10000.0')) # [cite: 174]
        
        test_baseline.execute_test_steps( 
            desired_range_width_multiplier=desired_rwm_base, 
            target_weth_balance=target_weth_for_test,
            target_usdc_balance=target_usdc_for_test 
        )
    except FileNotFoundError as e: logger.error(f"Setup Error - Address file not found: {e}") # [cite: 175]
    except ValueError as e: logger.error(f"Configuration Error - Problem reading address or address key missing: {e}") # [cite: 175]
    except Exception as e: 
        logger.exception(f"An unexpected error occurred during baseline main execution:") 
        if test_baseline is None: 
            if baseline_address_val and Web3.is_address(baseline_address_val): test_baseline = BaselineTest(baseline_address_val) 
            else: test_baseline = BaselineTest("0x"+"0"*40) 
        if hasattr(test_baseline, 'metrics'): # [cite: 176]
            test_baseline.metrics['action_taken'] = test_baseline.ACTION_STATES["UNEXPECTED_ERROR"] # [cite: 176]
            test_baseline.metrics['error_message'] = (test_baseline.metrics.get('error_message',"") + f";MainException: {str(e)}").strip(";") # [cite: 176]
            test_baseline.save_metrics() # [cite: 176]
    finally: 
        logger.info("="*50); logger.info("Baseline test run finished."); logger.info("="*50) # [cite: 177]

if __name__ == "__main__":
    main()