# -*- coding: utf-8 -*-
# filepath: d:\Uni-Dex-Marketplace_test\Phase3_Smart_Contract\test\predictive\predictive_test.py
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
from web3.logs import DISCARD
from web3.exceptions import ContractLogicError # Added for better error handling

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
    print(f"ERROR importing from test.utils in predictive_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272
MAX_TICK_CONST = 887272

TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV = os.getenv('TOKEN_MANAGER_OPTIMIZED_ADDRESS')

# --- Setup Logging ---
if not logging.getLogger('predictive_test').hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("predictive_test.log", mode='a'),
            logging.StreamHandler(sys.stdout)
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
            "TX_SENT": "tx_sent",
            "TX_SUCCESS_ADJUSTED": "tx_success_adjusted",
            "TX_REVERTED": "tx_reverted", "TX_WAIT_FAILED": "tx_wait_failed",
            "METRICS_UPDATE_FAILED": "metrics_update_failed",
            "UNEXPECTED_ERROR": "unexpected_error",
            "TX_SUCCESS_ADJUSTED_INITIAL": "tx_success_adjusted_initial",
            "TX_SUCCESS_SWAP_FEES": "tx_success_swap_fees",
            "TX_SUCCESS_ADJUSTED_FINAL": "tx_success_adjusted_final",
            "SWAP_FOR_FEES_FAILED": "swap_for_fees_failed",
            "FEES_COLLECT_ONLY_SUCCESS": "fees_collect_only_success", # New state
            "FEES_COLLECT_ONLY_FAILED": "fees_collect_only_failed"  # New state
        }
        super().__init__(contract_address, "PredictiveLiquidityManager")
        self.metrics = self._reset_metrics() # Ensure metrics are reset with Predictive-specific defaults
        self.pool_address = None
        self.pool_contract = None
        self.nft_position_manager_address = None # To store the address of Uniswap's NFPM
        self.nft_manager_contract_for_events = None # Will be initialized in setup()

        self.token_manager_optimized_address = None
        self.token_manager_contract = None
        if TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV:
            try:
                self.token_manager_optimized_address = Web3.to_checksum_address(TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV)
                logger.info(f"TokenOperationsManagerOptimized address loaded for Predictive: {self.token_manager_optimized_address}")
            except ValueError:
                logger.error(f"Invalid TokenOperationsManagerOptimized address from ENV for Predictive: {TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV}")
        else:
            logger.warning("TOKEN_MANAGER_OPTIMIZED_ADDRESS env var not set for Predictive. Swap for fee generation will be skipped.")

    def _reset_metrics(self):
        # Based on your original _reset_metrics from last_test_code.txt
        base_metrics = super()._reset_metrics()
        predictive_metrics = {
            'contract_type': 'Predictive',
            'action_taken': self.ACTION_STATES["INIT"],
            'predictedPrice_api': None,
            'predictedTick_calculated': None,
            'actualPrice_pool': None,
            'fees_collected_token0_via_collect_only': None, # Initialize as None
            'fees_collected_token1_via_collect_only': None  # Initialize as None
        }
        final_metrics = {**base_metrics, **predictive_metrics}
        original_keys_defaults = {
            'timestamp': None, 'tx_hash': None, 'range_width_multiplier_setting': None,
            'external_api_eth_price': None, 'sqrtPriceX96_pool': 0, 'currentTick_pool': 0,
            'targetTickLower_calculated': 0, 'targetTickUpper_calculated': 0,
            'initial_contract_balance_token0': None, 'initial_contract_balance_token1': None,
            'finalTickLower_contract': 0, 'finalTickUpper_contract': 0, 'liquidity_contract': 0,
            'amount0_provided_to_mint': None, 'amount1_provided_to_mint': None,
            'fees_collected_token0': None, 'fees_collected_token1': None,
            'gas_used': 0, 'gas_cost_eth': 0.0, 'error_message': ""
        }
        for key, default_value in original_keys_defaults.items():
            if key not in final_metrics:
                final_metrics[key] = default_value
        return final_metrics

    def setup(self, desired_range_width_multiplier: int) -> bool:
        if not super().setup(desired_range_width_multiplier):
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = self.metrics.get('error_message', "Base setup failed")
            if not self.metrics.get('error_message'): self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("web3_utils.w3 not available in PredictiveTest setup after base.setup()")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup"
                return False

            try:
                self.nft_position_manager_address = Web3.to_checksum_address(
                    self.contract.functions.positionManager().call()
                )
                logger.info(f"PredictiveTest.setup: Fetched Uniswap INonfungiblePositionManager address from OUR contract: {self.nft_position_manager_address}")
            except Exception as e_fetch_nfpm:
                logger.error(f"PredictiveTest.setup: Failed to fetch INonfungiblePositionManager address from our contract: {e_fetch_nfpm}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Failed to get NFPM address: {str(e_fetch_nfpm)}"
                return False

            if self.nft_position_manager_address and self.nft_position_manager_address != ('0x' + '0'*40):
                nft_manager_abi_collect_event = [{"anonymous": False, "inputs": [{"indexed": True, "internalType": "uint256", "name": "tokenId", "type": "uint256"},{"indexed": False, "internalType": "address", "name": "recipient", "type": "address"},{"indexed": False, "internalType": "uint256", "name": "amount0", "type": "uint256"},{"indexed": False, "internalType": "uint256", "name": "amount1", "type": "uint256"}],"name": "Collect","type": "event"}]
                try:
                    self.nft_manager_contract_for_events = web3_utils.w3.eth.contract(
                        address=self.nft_position_manager_address,
                        abi=nft_manager_abi_collect_event
                    )
                    logger.info(f"PredictiveTest.setup: Initialized contract instance for INonfungiblePositionManager Collect events at {self.nft_position_manager_address}.")
                except Exception as e_init_nft_events:
                    logger.error(f"PredictiveTest.setup: Failed to create contract instance for NFPM Collect events using address '{self.nft_position_manager_address}': {e_init_nft_events}")
                    self.nft_manager_contract_for_events = None
            else:
                logger.warning(f"PredictiveTest.setup: self.nft_position_manager_address is not validly set (value: {self.nft_position_manager_address}). Cannot initialize for NFPM Collect events.")
                self.nft_manager_contract_for_events = None

            factory_address = self.contract.functions.factory().call()
            factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            if not factory_contract:
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "Factory contract not found"
                return False

            fee = self.contract.functions.fee().call()
            self.pool_address = factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Predictive pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
                
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            if not self.pool_contract:
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "Pool contract object could not be created"
                return False

            logger.info(f"Predictive Pool contract initialized at {self.pool_address}")
            
            logger.info(f"Setting rangeWidthMultiplier to {desired_range_width_multiplier} for Predictive contract...")
            self.metrics['range_width_multiplier_setting'] = desired_range_width_multiplier
            
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                logger.error("PRIVATE_KEY not set for setting rangeWidthMultiplier.")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "PRIVATE_KEY missing for RWM setup"
                return False
            
            tx_account = Account.from_key(private_key)
            tx_params = {
                'from': tx_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(tx_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }
            try:
                gas_estimate = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).estimate_gas({'from': tx_account.address})
                tx_params['gas'] = int(gas_estimate * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for setRangeWidthMultiplier: {e}. Using default 200000.")
                tx_params['gas'] = 200000

            tx_set_rwm_build = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier)
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm_build.build_transaction(tx_params), private_key)
            
            if not receipt_rwm or receipt_rwm.status == 0:
                tx_hash_rwm = receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'
                logger.error(f"Failed to set rangeWidthMultiplier for Predictive contract. TxHash: {tx_hash_rwm}")
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";Failed to set RWM (tx: {tx_hash_rwm})").strip(';')
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Predictive contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Predictive setup failed: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}"
            return False
    
    def get_predicted_price_from_api(self) -> float | None:
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        try:
            logger.info(f"Querying LSTM API at {LSTM_API_URL}...")
            response = requests.get(LSTM_API_URL, timeout=25)
            response.raise_for_status()
            data = response.json()
            
            predicted_price_str = data.get('predicted_price') or data.get('price') or data.get('prediction')
            if predicted_price_str is None:
                logger.error(f"Predicted price key not found in API response. Data: {data}")
                raise ValueError("Predicted price not in API response")

            if isinstance(predicted_price_str, str):
                predicted_price_str = predicted_price_str.replace("USD", "").strip()

            predicted_price = float(predicted_price_str)
            logger.info(f"Received predicted ETH price from API: {predicted_price:.4f} USD")
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
            data_for_log = 'N/A'
            if 'data' in locals(): data_for_log = data
            logger.exception(f"Error processing API response from {LSTM_API_URL}. Data: {data_for_log}. Error: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Response Processing Error: {str(e)}"
            return None

    def calculate_tick_from_price(self, price: float) -> int | None:
        """Calculate Uniswap V3 tick from a given price."""
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if self.token0_decimals is None or self.token1_decimals is None:
            logger.error("Token decimals not available for tick calculation. Ensure LiquidityTestBase.setup() was successful.")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Token decimals missing for tick calc"
            return None
            
        try:
            price_decimal = Decimal(str(price))
            effective_sqrt_price_arg = (Decimal(1) / price_decimal) * (Decimal(10)**(self.token1_decimals - self.token0_decimals))
            
            if effective_sqrt_price_arg <= 0:
                logger.error(f"Argument for sqrt in tick calculation is non-positive: {effective_sqrt_price_arg} (price: {price}, dec0: {self.token0_decimals}, dec1: {self.token1_decimals})")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Invalid arg for sqrt in tick calc"
                return None
            
            effective_sqrt_price = effective_sqrt_price_arg.sqrt()
            
            if effective_sqrt_price <= 0:
                logger.error(f"Effective sqrt price for tick calculation is non-positive: {effective_sqrt_price}")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = "Invalid effective_sqrt_price for tick calc"
                return None

            tick = math.floor(math.log(float(effective_sqrt_price)) / math.log(math.sqrt(1.0001)))
            tick = max(MIN_TICK_CONST, min(MAX_TICK_CONST, tick))
            
            logger.info(f"Tick calculation details (Original Predictive Logic):")
            logger.info(f"  Input price (assumed T1/T0, e.g. WETH/USDC): {price}")
            logger.info(f"  Effective_sqrt_price_arg ((1/price) * 10^(dec1-dec0)): {effective_sqrt_price_arg}")
            logger.info(f"  Effective_sqrt_price: {effective_sqrt_price}")
            logger.info(f"  Final tick: {tick}")
            
            self.metrics['predictedTick_calculated'] = tick
            return tick
            
        except Exception as e:
            logger.exception(f"Failed to calculate predicted tick from price {price}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Tick Calculation Error: {str(e)}"
            return None

    def update_pool_and_position_metrics(self, final_update=False):
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        try:
            if self.pool_contract:
                slot0 = self.pool_contract.functions.slot0().call()
                sqrt_price_x96_pool, current_tick_pool = slot0[0], slot0[1]
                self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96_pool
                self.metrics['currentTick_pool'] = current_tick_pool
                self.metrics['actualPrice_pool'] = self.sqrt_price_x96_to_price_token0_in_token1(str(sqrt_price_x96_pool))
            else:
                logger.warning("Pool contract not available for metrics update (pool_and_position).")

            position_info = self.get_position_info()
            if position_info:
                is_active = position_info.get('active', False)
                if 'active' not in position_info and len(position_info) == 5 :
                    is_active = position_info[4]

                # Update currentXXX_contract fields if they exist in metrics dictionary
                if 'currentTickLower_contract' in self.metrics:
                        self.metrics['currentTickLower_contract'] = position_info.get('tickLower', self.metrics.get('currentTickLower_contract',0))
                if 'currentTickUpper_contract' in self.metrics:
                        self.metrics['currentTickUpper_contract'] = position_info.get('tickUpper', self.metrics.get('currentTickUpper_contract',0))
                if 'currentLiquidity_contract' in self.metrics:
                        self.metrics['currentLiquidity_contract'] = position_info.get('liquidity', self.metrics.get('currentLiquidity_contract',0))

                if final_update or (is_active and position_info.get('liquidity', 0) > 0):
                    self.metrics['finalTickLower_contract'] = position_info.get('tickLower', 0)
                    self.metrics['finalTickUpper_contract'] = position_info.get('tickUpper', 0)
                    if self.metrics.get('liquidity_contract', 0) == 0 or final_update:
                        self.metrics['liquidity_contract'] = position_info.get('liquidity', 0)
            else:
                logger.warning("Could not get position info from contract for metrics update.")

        except Exception as e:
            logger.exception(f"Error updating pool/position metrics: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["METRICS_UPDATE_FAILED"]
            if not self.metrics.get('error_message'): self.metrics['error_message'] = f"Metrics Update Error: {str(e)}"

    def sqrt_price_x96_to_price_token0_in_token1(self, sqrt_price_x96_str: str) -> Decimal:
        """Convert sqrtPriceX96 to price of token0 in terms of token1 (USDC/WETH) - YOUR ORIGINAL FUNCTION"""
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        sqrt_price_x96 = Decimal(sqrt_price_x96_str)
        if sqrt_price_x96 == Decimal(0): return Decimal(0)
        
        if self.token0_decimals is None or self.token1_decimals is None:
            logger.error("Token decimals not available for sqrt_price_x96_to_price_token0_in_token1 (Predictive)")
            return Decimal("-1")

        price_t1_in_t0 = (sqrt_price_x96 / TWO_POW_96)**2
        if price_t1_in_t0 == Decimal(0):
            return Decimal("inf") if sqrt_price_x96 != Decimal(0) else Decimal(0)
        
        price_t0_in_t1 = Decimal(1) / price_t1_in_t0
        decimals_adjustment = Decimal(10)**(self.token0_decimals - self.token1_decimals)
        if decimals_adjustment == Decimal(0):
            return Decimal("-1")

        return price_t0_in_t1 / decimals_adjustment

    # ADDED THIS METHOD - Ensure it is present in your class
    def _get_contract_token_balances_individually(self):
        """Helper to get token balances individually for the contract."""
        balance0_wei, balance1_wei = 0, 0
        try:
            if self.token0:
                token0_contract_instance = web3_utils.get_contract(self.token0, "IERC20")
                if token0_contract_instance:
                    balance0_wei = token0_contract_instance.functions.balanceOf(self.contract_address).call()
            if self.token1:
                token1_contract_instance = web3_utils.get_contract(self.token1, "IERC20")
                if token1_contract_instance:
                    balance1_wei = token1_contract_instance.functions.balanceOf(self.contract_address).call()
        except Exception as e:
            logger.warning(f"Could not read contract token balances individually: {e}")
        return balance0_wei, balance1_wei

    def _perform_swap_for_fees(self, funding_account, private_key_env, swap_token_in_addr: str, swap_token_out_addr: str, swap_amount_readable: Decimal, token_in_decimals: int, token_out_decimals: int):
        """Helper function to perform a swap using TokenOperationsManagerOptimized."""
        # YOUR ORIGINAL FUNCTION - UNCHANGED
        if not self.token_manager_optimized_address:
            logger.warning("TokenManagerOptimized address not set for Predictive. Skipping swap for fees.")
            return False
        
        if not self.token_manager_contract:
            self.token_manager_contract = web3_utils.get_contract(self.token_manager_optimized_address, "TokenOperationsManagerOptimized")
            if not self.token_manager_contract:
                logger.error("Failed to get TokenOperationsManagerOptimized contract instance for Predictive.")
                return False

        logger.info(f"Attempting swap for Predictive: {swap_amount_readable} of {swap_token_in_addr} for {swap_token_out_addr} via TokenManager.")

        try:
            pool_fee_for_swap = self.contract.functions.fee().call()
            amount_to_swap_wei = int(swap_amount_readable * (Decimal(10) ** token_in_decimals))

            token_in_contract_for_approve = web3_utils.get_contract(swap_token_in_addr, "IERC20")
            
            logger.info(f"Approving TokenManagerOptimized ({self.token_manager_optimized_address}) to spend {swap_amount_readable} of token {swap_token_in_addr} from deployer {funding_account.address}...")
            current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            approve_tx_params = {'from': funding_account.address, 'nonce': current_nonce}
            
            try:
                gas_est_approve = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).estimate_gas({'from': funding_account.address})
                approve_tx_params['gas'] = int(gas_est_approve * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation for TokenManager approval (Predictive) failed: {e}. Using default 100,000.")
                approve_tx_params['gas'] = 100000

            built_approve_tx = token_in_contract_for_approve.functions.approve(self.token_manager_optimized_address, amount_to_swap_wei).build_transaction(approve_tx_params)
            receipt_approve = web3_utils.send_transaction(built_approve_tx, private_key_env)

            if not receipt_approve or receipt_approve.status == 0:
                logger.error(f"Failed to approve TokenManagerOptimized (Predictive). Receipt: {receipt_approve}")
                return False
            logger.info(f"TokenManagerOptimized approved for Predictive. Tx: {receipt_approve.transactionHash.hex()}")
            
            current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            logger.info(f"Calling swap on TokenManagerOptimized for Predictive: {swap_amount_readable} {swap_token_in_addr} -> {swap_token_out_addr}, fee {pool_fee_for_swap}...")
            swap_tx_params = {'from': funding_account.address, 'nonce': current_nonce }
            
            try:
                checksum_token_in = Web3.to_checksum_address(swap_token_in_addr)
                checksum_token_out = Web3.to_checksum_address(swap_token_out_addr)
                gas_est_swap = self.token_manager_contract.functions.swap(
                    checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0
                ).estimate_gas({'from': funding_account.address})
                swap_tx_params['gas'] = int(gas_est_swap * 1.30)
            except ContractLogicError as cle:
                logger.error(f"Gas estimation for TokenManager swap (Predictive) failed due to contract logic: {cle}.")
                return False
            except Exception as e:
                logger.warning(f"Gas estimation for TokenManager swap (Predictive) failed: {e}. Using default 700,000.")
                swap_tx_params['gas'] = 700000

            built_swap_tx = self.token_manager_contract.functions.swap(
                checksum_token_in, checksum_token_out, pool_fee_for_swap, amount_to_swap_wei, 0
            ).build_transaction(swap_tx_params)
            
            receipt_swap = web3_utils.send_transaction(built_swap_tx, private_key_env)

            if receipt_swap and receipt_swap.status == 1:
                logger.info(f"Swap via TokenManagerOptimized for Predictive successful. Tx: {receipt_swap.transactionHash.hex()}")
                swap_logs = self.token_manager_contract.events.Operation().process_receipt(receipt_swap, errors=DISCARD)
                for log_entry in swap_logs:
                    op_type_bytes32 = log_entry.args.opType
                    is_swap_op = (web3_utils.w3.to_hex(op_type_bytes32) == web3_utils.w3.to_hex(web3_utils.w3.solidity_keccak(['string'],['SWAP'])))

                    if is_swap_op:
                        decimals_for_amount_out = self.token0_decimals if Web3.to_checksum_address(log_entry.args.tokenB) == Web3.to_checksum_address(self.token0) else self.token1_decimals
                        amount_out_readable = Decimal(log_entry.args.amount) / (Decimal(10) ** decimals_for_amount_out)
                        logger.info(f"TokenManager Swap Event (Predictive context): TokenIn={log_entry.args.tokenA}, TokenOut={log_entry.args.tokenB}, AmountOut={amount_out_readable:.6f}")
                return True
            else:
                logger.error(f"Swap via TokenManagerOptimized for Predictive failed. Receipt: {receipt_swap}")
                return False
        except Exception as e:
            logger.exception(f"Error during _perform_swap_for_fees for Predictive: {e}")
            return False

    # NEW METHOD to call collectCurrentFeesOnly
    def _call_collect_fees_only(self, funding_account, private_key_env) -> bool:
        logger.info("Attempting to call collectCurrentFeesOnly() on PredictiveLiquidityManager...")
        if not self.contract:
            logger.error("PredictiveLiquidityManager contract instance not available for collectCurrentFeesOnly.")
            return False
        
        # Initialize fee metrics if they are None (first time being set)
        if self.metrics.get('fees_collected_token0') is None: self.metrics['fees_collected_token0'] = 0
        if self.metrics.get('fees_collected_token1') is None: self.metrics['fees_collected_token1'] = 0
        if self.metrics.get('fees_collected_token0_via_collect_only') is None: self.metrics['fees_collected_token0_via_collect_only'] = 0
        if self.metrics.get('fees_collected_token1_via_collect_only') is None: self.metrics['fees_collected_token1_via_collect_only'] = 0

        tx_call = self.contract.functions.collectCurrentFeesOnly()
        current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
        tx_params = {'from': funding_account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
        try:
            tx_params['gas'] = int(tx_call.estimate_gas({'from': funding_account.address}) * 1.25)
        except ContractLogicError as cle:
            logger.error(f"Gas estimation for collectCurrentFeesOnly failed (Logic Error): {cle}. This might indicate an issue like no active position or other require failure.")
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";CollectFeesGasEstLogicError: {str(cle)[:100]}").strip(';')
            return False
        except Exception as e:
            tx_params['gas'] = 300000
            logger.warning(f"Gas estimation for collectCurrentFeesOnly failed: {e}. Using default {tx_params['gas']}.")

        try:
            built_tx = tx_call.build_transaction(tx_params)
            receipt = web3_utils.send_transaction(built_tx, private_key_env)

            if receipt and receipt.status == 1:
                logger.info(f"collectCurrentFeesOnly() transaction successful. Tx: {receipt.transactionHash.hex()}")
                self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_SUCCESS"]
                
                amount0_from_our_event, amount1_from_our_event = 0, 0 # Default if event not found/failed
                our_fee_event_logs = self.contract.events.FeesOnlyCollected().process_receipt(receipt, errors=DISCARD)
                if our_fee_event_logs:
                    log_args = our_fee_event_logs[0].args
                    logger.info(f"PredictiveContract FeesOnlyCollected Event: TokenId={log_args.tokenId}, Amount0={log_args.amount0Fees}, Amount1={log_args.amount1Fees}, Success={log_args.success}")
                    if log_args.success:
                        amount0_from_our_event = log_args.amount0Fees
                        amount1_from_our_event = log_args.amount1Fees
                    else:
                        logger.warning("FeesOnlyCollected event reported success=false from contract.")
                else:
                    logger.warning("No FeesOnlyCollected event found from our contract after collectCurrentFeesOnly call.")

                # Populate the specific _via_collect_only fields
                self.metrics['fees_collected_token0_via_collect_only'] = amount0_from_our_event
                self.metrics['fees_collected_token1_via_collect_only'] = amount1_from_our_event
                # MODIFICATION: Also populate the main fee fields with these values
                self.metrics['fees_collected_token0'] = amount0_from_our_event
                self.metrics['fees_collected_token1'] = amount1_from_our_event
                logger.info(f"Main fee metrics updated: T0={self.metrics['fees_collected_token0']}, T1={self.metrics['fees_collected_token1']}")


                if self.nft_manager_contract_for_events:
                    logger.info(f"Checking Uniswap NFPM Collect events in Tx: {receipt.transactionHash.hex()} after collectCurrentFeesOnly call...")
                    collect_logs_nfpm = self.nft_manager_contract_for_events.events.Collect().process_receipt(receipt, errors=DISCARD)
                    found_nfpm_collect = False
                    for nfpm_log_entry in collect_logs_nfpm:
                        if nfpm_log_entry.args.recipient.lower() == self.contract_address.lower():
                            found_nfpm_collect = True
                            logger.info(f"Uniswap NFPM Collect Event (Post collectCurrentFeesOnly, TokenId={nfpm_log_entry.args.tokenId}): Amount0={nfpm_log_entry.args.amount0}, Amount1={nfpm_log_entry.args.amount1}")
                            # If our contract's event was problematic, or for stronger verification, use NFPM event data
                            if not our_fee_event_logs or not our_fee_event_logs[0].args.success :
                                self.metrics['fees_collected_token0_via_collect_only'] = nfpm_log_entry.args.amount0
                                self.metrics['fees_collected_token1_via_collect_only'] = nfpm_log_entry.args.amount1
                                # Also update main fee fields if our event failed
                                self.metrics['fees_collected_token0'] = nfpm_log_entry.args.amount0
                                self.metrics['fees_collected_token1'] = nfpm_log_entry.args.amount1
                                logger.info("Updated ALL fee metrics from NFPM event due to issue/absence of contract's FeesOnlyCollected event.")
                            elif log_args.amount0Fees != nfpm_log_entry.args.amount0 or log_args.amount1Fees != nfpm_log_entry.args.amount1:
                                logger.warning(f"Discrepancy: OurEvent(0:{log_args.amount0Fees}, 1:{log_args.amount1Fees}) vs NFPMEvent(0:{nfpm_log_entry.args.amount0}, 1:{nfpm_log_entry.args.amount1}). Prioritizing our contract's event if successful.")
                            break
                    if not found_nfpm_collect:
                        logger.info("No direct Uniswap NFPM Collect event found with our contract as recipient in this transaction.")
                return True
            else:
                tx_hash_str = receipt.transactionHash.hex() if receipt else "N/A"
                logger.error(f"collectCurrentFeesOnly() transaction failed. Tx: {tx_hash_str}")
                self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";CollectFeesTxFailed: {tx_hash_str}").strip(';')
                return False
        except Exception as e:
            logger.exception(f"Exception during _call_collect_fees_only: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";CollectFeesException: {str(e)[:100]}").strip(';')
            return False

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool:
        self.metrics = self._reset_metrics()
        try:
            self.get_current_eth_price()
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from predictive contract for metrics.")

        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position.")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx"
            self.save_metrics()
            return False
        
        funding_account = Account.from_key(private_key_env)
        stage_results = {'initial_adjustment': False, 'swap': False, 'collect_only': False, 'final_adjustment': False}
        
        pos_info_before_all = self.get_position_info()
        if pos_info_before_all:
            if 'currentTickLower_contract' in self.metrics: self.metrics['currentTickLower_contract'] = pos_info_before_all.get('tickLower')
            if 'currentTickUpper_contract' in self.metrics: self.metrics['currentTickUpper_contract'] = pos_info_before_all.get('tickUpper')
            if 'currentLiquidity_contract' in self.metrics: self.metrics['currentLiquidity_contract'] = pos_info_before_all.get('liquidity', 0)

        # --- STAGE 1: Initial Position Adjustment by Strategy Contract ---
        logger.info("\n--- STAGE 1: Predictive Strategy - Initial position adjustment ---")
        try:
            predicted_price = self.get_predicted_price_from_api()
            if predicted_price is None: self.save_metrics(); return False
            predicted_tick = self.calculate_tick_from_price(predicted_price)
            if predicted_tick is None: self.save_metrics(); return False
            self.update_pool_and_position_metrics(final_update=False)
            logger.info("Ensuring precise token balances for Predictive contract (initial)...")
            if not contract_funder.ensure_precise_token_balances(
                self.contract_address, self.token0, self.token0_decimals, target_usdc_balance,
                self.token1, self.token1_decimals, target_weth_balance, private_key_env
            ):
                logger.error("Precise funding for Predictive contract failed (initial).")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Precise contract funding failed (initial)"
                if 'currentTickLower_contract' in self.metrics: self.metrics['finalTickLower_contract'] = self.metrics.get('currentTickLower_contract', 0)
                if 'currentTickUpper_contract' in self.metrics: self.metrics['finalTickUpper_contract'] = self.metrics.get('currentTickUpper_contract', 0)
                if 'currentLiquidity_contract' in self.metrics: self.metrics['finalLiquidity_contract'] = self.metrics.get('currentLiquidity_contract', 0)
                self.save_metrics(); return False
            
            balance0_val, balance1_val = self._get_contract_token_balances_individually() # Corrected usage
            self.metrics['initial_contract_balance_token0'] = balance0_val
            self.metrics['initial_contract_balance_token1'] = balance1_val
            logger.info(f"Initial contract balances for metrics (after funding): Token0={balance0_val}, Token1={balance1_val}")

            logger.info(f"Calling updatePredictionAndAdjust (initial) with predictedTick: {predicted_tick}")
            tx_function_call_initial = self.contract.functions.updatePredictionAndAdjust(predicted_tick)
            current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            tx_params_initial = {'from': funding_account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
            try:
                tx_params_initial['gas'] = int(tx_function_call_initial.estimate_gas({'from': funding_account.address}) * 1.25)
            except Exception as est_err:
                logger.warning(f"Gas estimation for 'updatePredictionAndAdjust' (initial) failed: {est_err}. Using default 1,500,000")
                tx_params_initial['gas'] = 1500000
            built_tx_initial = tx_function_call_initial.build_transaction(tx_params_initial)
            receipt_initial = web3_utils.send_transaction(built_tx_initial, private_key_env)
            self.metrics['tx_hash'] = receipt_initial.transactionHash.hex() if receipt_initial else None
            
            if receipt_initial and receipt_initial.status == 1:
                logger.info(f"Initial adjustment transaction successful. Tx: {self.metrics['tx_hash']}")
                self.metrics['gas_used'] = receipt_initial.get('gasUsed', 0)
                effective_gas_price_initial = receipt_initial.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * effective_gas_price_initial, 'ether'))
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"]
                stage_results['initial_adjustment'] = True
                try:
                    adj_metrics_logs_initial = self.contract.events.PredictionAdjustmentMetrics().process_receipt(receipt_initial, errors=DISCARD)
                    if adj_metrics_logs_initial:
                        self.metrics['targetTickLower_calculated'] = adj_metrics_logs_initial[0].args.finalTickLower
                        self.metrics['targetTickUpper_calculated'] = adj_metrics_logs_initial[0].args.finalTickUpper
                        liq_attr = 'liquidity' if hasattr(adj_metrics_logs_initial[0].args, 'liquidity') else 'finalLiquidity' if hasattr(adj_metrics_logs_initial[0].args, 'finalLiquidity') else None
                        if liq_attr: self.metrics['liquidity_contract'] = adj_metrics_logs_initial[0].args.get(liq_attr)
                        self.metrics['finalTickLower_contract'] = adj_metrics_logs_initial[0].args.finalTickLower
                        self.metrics['finalTickUpper_contract'] = adj_metrics_logs_initial[0].args.finalTickUpper
                        if liq_attr: self.metrics['finalLiquidity_contract'] = adj_metrics_logs_initial[0].args.get(liq_attr)
                    op_logs_initial = self.contract.events.LiquidityOperation().process_receipt(receipt_initial, errors=DISCARD)
                    for log_entry in op_logs_initial:
                        if log_entry.args.operationType == "MINT":
                            self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0
                            self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1
                            if not adj_metrics_logs_initial:
                                self.metrics['finalTickLower_contract'] = log_entry.args.tickLower
                                self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper
                                self.metrics['finalLiquidity_contract'] = log_entry.args.liquidity
                        elif log_entry.args.operationType == "REMOVE":
                            logger.info(f"PredictiveContract LiquidityOperation REMOVE (Stage 1): tokenId={log_entry.args.tokenId}, amount0(Total)={log_entry.args.amount0}, amount1(Total)={log_entry.args.amount1}")
                except Exception as log_err_initial: logger.exception(f"Error processing logs for initial predictive transaction: {log_err_initial}")
            elif receipt_initial:
                logger.error(f"Initial adjustment transaction reverted. Tx: {self.metrics['tx_hash']}")
                self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                self.metrics['error_message'] = "tx_reverted_onchain (initial)"
                self.metrics['gas_used'] = receipt_initial.get('gasUsed', 0)
                if 'currentTickLower_contract' in self.metrics: self.metrics['finalTickLower_contract'] = self.metrics.get('currentTickLower_contract', 0)
                if 'currentTickUpper_contract' in self.metrics: self.metrics['finalTickUpper_contract'] = self.metrics.get('currentTickUpper_contract', 0)
                if 'currentLiquidity_contract' in self.metrics: self.metrics['finalLiquidity_contract'] = self.metrics.get('currentLiquidity_contract', 0)
                self.save_metrics(); return False
            else:
                logger.error("Initial adjustment transaction sending/receipt failed.")
                self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for initial adjustment failed"
                if 'currentTickLower_contract' in self.metrics: self.metrics['finalTickLower_contract'] = self.metrics.get('currentTickLower_contract', 0)
                if 'currentTickUpper_contract' in self.metrics: self.metrics['finalTickUpper_contract'] = self.metrics.get('currentTickUpper_contract', 0)
                if 'currentLiquidity_contract' in self.metrics: self.metrics['finalLiquidity_contract'] = self.metrics.get('currentLiquidity_contract', 0)
                self.save_metrics(); return False
        except Exception as tx_err_initial:
            logger.exception(f"Error during initial adjustment transaction: {tx_err_initial}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"TxError (initial):{str(tx_err_initial)}"
            self.save_metrics(); return False
        
        # --- Parameters for Swap Simulation ---
        NUM_SWAPS = 50
        SWAP_AMOUNT_WETH = Decimal("2.5")

        # --- STAGE 2: Perform Multiple Swaps via TokenManager to Generate Fees ---
        if stage_results['initial_adjustment']:
            logger.info(f"\n--- STAGE 2: Predictive Strategy - Simulating {NUM_SWAPS} swaps to generate fees ---")
            all_swaps_successful = True  # A flag to track the success of all swaps
            for i in range(NUM_SWAPS):
                logger.info(f"--- Performing Swap {i + 1}/{NUM_SWAPS} ---")
        
                swap_token_in_addr = self.token1
                swap_token_out_addr = self.token0
                token_in_decimals_for_swap = self.token1_decimals
                token_out_decimals_for_swap = self.token0_decimals

                # Call your existing swap function inside the loop
                swap_successful_this_iteration = self._perform_swap_for_fees(
                    funding_account, private_key_env,
                    swap_token_in_addr, swap_token_out_addr,
                    SWAP_AMOUNT_WETH,  # Use the parameter here
                    token_in_decimals_for_swap, token_out_decimals_for_swap
                )
                # If any swap fails, we stop the loop and mark the whole stage as failed
                if not swap_successful_this_iteration:
                    logger.error(f"Swap {i + 1} failed, stopping the swap simulation.")
                    all_swaps_successful = False
                    break
            # After the loop, update the main stage result based on the outcome of all swaps
            stage_results['swap'] = all_swaps_successful

            if all_swaps_successful:
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]
                logger.info(f"All {NUM_SWAPS} swaps completed successfully.")
            else:
                self.metrics['action_taken'] = self.ACTION_STATES["SWAP_FOR_FEES_FAILED"]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";Swap simulation failed at iteration {i + 1}").strip(';')

        # --- STAGE 2.5: Explicitly Collect Fees ---
        # This part remains unchanged. It will now run only if ALL swaps were successful.
        if stage_results['initial_adjustment'] and stage_results['swap']:
        #----------------------------------
            logger.info("\n--- STAGE 2.5: Predictive Strategy - Explicit Fee Collection ---")
            stage_results['collect_only'] = self._call_collect_fees_only(funding_account, private_key_env)
        else:
            logger.info("Skipping Stage 2.5 (Explicit Fee Collection) due to previous stage failure(s).")
            stage_results['collect_only'] = True

        # --- STAGE 3: Second Position Adjustment by Strategy Contract ---
        if stage_results['initial_adjustment'] and stage_results['swap'] and stage_results['collect_only']:
            logger.info("\n--- STAGE 3: Predictive Strategy - Second position adjustment ---")
            try:
                predicted_price_final = self.get_predicted_price_from_api()
                predicted_tick_final = None
                if predicted_price_final is None:
                    logger.warning("Could not get new prediction for final adjustment. Using last known pool tick.")
                    if self.pool_contract: slot0_final = self.pool_contract.functions.slot0().call(); predicted_tick_final = slot0_final[1]
                    else: logger.error("Pool contract not available for fallback tick during final adjustment.")
                else: predicted_tick_final = self.calculate_tick_from_price(predicted_price_final)
                
                if predicted_tick_final is None:
                    logger.error("Could not determine a target tick for final adjustment. Skipping final adjustment.")
                    stage_results['final_adjustment'] = False
                else:
                    logger.info("Ensuring precise token balances for Predictive contract (final)...")
                    if not contract_funder.ensure_precise_token_balances(
                        self.contract_address, self.token0, self.token0_decimals, target_usdc_balance,
                        self.token1, self.token1_decimals, target_weth_balance, private_key_env
                    ):
                        logger.error("Precise funding for Predictive contract failed (final).")
                        self.metrics['error_message'] = (self.metrics.get('error_message', "") + ";Funding failed (final)").strip(';')
                    
                    balance0_before_final_adj, balance1_before_final_adj = self._get_contract_token_balances_individually() # Corrected usage
                    logger.info(f"Balances before final updatePredictionAndAdjust: Token0={balance0_before_final_adj}, Token1={balance1_before_final_adj}")

                    logger.info(f"Calling updatePredictionAndAdjust (final) with predictedTick: {predicted_tick_final}")
                    tx_function_call_final = self.contract.functions.updatePredictionAndAdjust(predicted_tick_final)
                    current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
                    tx_params_final = {'from': funding_account.address, 'nonce': current_nonce, 'chainId': int(web3_utils.w3.net.version)}
                    try: tx_params_final['gas'] = int(tx_function_call_final.estimate_gas({'from': funding_account.address}) * 1.25)
                    except Exception as est_err_final: logger.warning(f"Gas estimation failed for 'updatePredictionAndAdjust' (final): {est_err_final}. Using default 1,500,000"); tx_params_final['gas'] = 1500000
                    built_tx_final = tx_function_call_final.build_transaction(tx_params_final)
                    receipt_final = web3_utils.send_transaction(built_tx_final, private_key_env)
                    self.metrics['tx_hash'] = receipt_final.transactionHash.hex() if receipt_final else self.metrics.get('tx_hash')

                    if receipt_final and receipt_final.status == 1:
                        logger.info(f"Final adjustment transaction successful. Tx: {receipt_final.transactionHash.hex()}")
                        self.metrics['gas_used'] += receipt_final.get('gasUsed', 0)
                        eff_gas_price_final = receipt_final.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                        self.metrics['gas_cost_eth'] += float(Web3.from_wei(receipt_final.get('gasUsed', 0) * eff_gas_price_final, 'ether'))
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_FINAL"]
                        stage_results['final_adjustment'] = True
                        
                        # If a full remove/add happened in Stage 3, it might have collected more fees.
                        # The self.metrics['fees_collected_token0/1'] would be updated by NFPM Collect event if one is found for this tx.
                        # If no NFPM Collect event here, the values from Stage 2.5 remain.
                        
                        adj_metrics_logs_final = self.contract.events.PredictionAdjustmentMetrics().process_receipt(receipt_final, errors=DISCARD)
                        if adj_metrics_logs_final:
                            self.metrics['finalTickLower_contract'] = adj_metrics_logs_final[0].args.finalTickLower
                            self.metrics['finalTickUpper_contract'] = adj_metrics_logs_final[0].args.finalTickUpper
                            self.metrics['targetTickLower_calculated'] = adj_metrics_logs_final[0].args.finalTickLower
                            self.metrics['targetTickUpper_calculated'] = adj_metrics_logs_final[0].args.finalTickUpper
                            liq_attr_final = 'liquidity' if hasattr(adj_metrics_logs_final[0].args, 'liquidity') else 'finalLiquidity' if hasattr(adj_metrics_logs_final[0].args, 'finalLiquidity') else None
                            if liq_attr_final: self.metrics['liquidity_contract'] = adj_metrics_logs_final[0].args.get(liq_attr_final)
                        
                        final_op_logs = self.contract.events.LiquidityOperation().process_receipt(receipt_final, errors=DISCARD)
                        for log_entry in final_op_logs:
                            if log_entry.args.operationType == "MINT":
                                self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0
                                self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1
                                self.metrics['liquidity_contract'] = log_entry.args.liquidity
                                if not adj_metrics_logs_final:
                                    self.metrics['finalTickLower_contract'] = log_entry.args.tickLower
                                    self.metrics['finalTickUpper_contract'] = log_entry.args.tickUpper
                            elif log_entry.args.operationType == "REMOVE":
                                logger.info(f"PredictiveContract LiquidityOperation REMOVE (Stage 3) TX: {receipt_final.transactionHash.hex()}, TokenId={log_entry.args.tokenId}, amount0(Total)={log_entry.args.amount0}, amount1(Total)={log_entry.args.amount1}, SuccessFlag={log_entry.args.success}")

                        if self.nft_manager_contract_for_events:
                            logger.info(f"Checking Uniswap NFPM Collect events for fees in FINAL Tx: {receipt_final.transactionHash.hex()} (NFPM: {self.nft_manager_contract_for_events.address})...")
                            collect_logs_nfpm_final = self.nft_manager_contract_for_events.events.Collect().process_receipt(receipt_final, errors=DISCARD)
                            found_nfpm_collect_final = False
                            if not collect_logs_nfpm_final: logger.info(f"No NFPM Collect events found in FINAL Tx {receipt_final.transactionHash.hex()}.")
                            for nfpm_log_entry in collect_logs_nfpm_final:
                                if nfpm_log_entry.args.recipient.lower() == self.contract_address.lower():
                                    found_nfpm_collect_final = True
                                    logger.info(f"Uniswap NFPM Collect Event (FINAL Tx, TokenId={nfpm_log_entry.args.tokenId}): Amount0={nfpm_log_entry.args.amount0}, Amount1={nfpm_log_entry.args.amount1}")
                                    # If a full remove/collect cycle happened, these are the total collected values (principal+fees)
                                    # Potentially overwrite the main fee fields if this collect is more comprehensive
                                    self.metrics['fees_collected_token0'] = nfpm_log_entry.args.amount0
                                    self.metrics['fees_collected_token1'] = nfpm_log_entry.args.amount1
                                    logger.info(f"Main fee metrics (fees_collected_token0/1) updated from FINAL Tx Uniswap NFPM Collect: ({self.metrics['fees_collected_token0']}, {self.metrics['fees_collected_token1']})")
                                    break
                        self.update_pool_and_position_metrics(final_update=True)
                    
                    elif receipt_final:
                        logger.error(f"Final adjustment transaction reverted. Tx: {receipt_final.transactionHash.hex()}")
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                        self.metrics['error_message'] = (self.metrics.get('error_message', "") + ";Final adjustment reverted").strip(';')
                        self.metrics['gas_used'] += receipt_final.get('gasUsed', 0)
                        stage_results['final_adjustment'] = False
                    else:
                        logger.error("Final adjustment transaction sending/receipt failed.")
                        if self.metrics['action_taken'] != self.ACTION_STATES["TX_REVERTED"]: self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                        self.metrics['error_message'] = (self.metrics.get('error_message', "") + ";Final adjustment send failed").strip(';')
                        stage_results['final_adjustment'] = False
            except Exception as tx_err_final:
                logger.exception(f"Error during final adjustment transaction: {tx_err_final}")
                if self.metrics.get('action_taken') not in [self.ACTION_STATES["SWAP_FOR_FEES_FAILED"], self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]]: self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";TxError (final):{str(tx_err_final)}").strip(';')
                stage_results['final_adjustment'] = False
        else:
            logger.info("Skipping Stage 3 (Final Position Adjustment) due to previous stage failure(s).")
            if stage_results['initial_adjustment'] and stage_results['swap'] and not stage_results['collect_only']: stage_results['final_adjustment'] = False
            else: stage_results['final_adjustment'] = True

        logger.info("Updating final pool and position metrics in 'finally' block for Predictive...")
        self.update_pool_and_position_metrics(final_update=True)
        self.save_metrics()
        
        if not stage_results['initial_adjustment']: logger.error("Overall Test Failed: Initial adjustment did not succeed."); return False
        if self.token_manager_optimized_address and not stage_results['swap']: logger.error("Overall Test Failed: Swap stage (intended) did not succeed."); return False
        if self.token_manager_optimized_address and not stage_results['collect_only']: logger.error("Overall Test Failed: Explicit fee collection (intended) did not succeed."); return False
        if not stage_results['final_adjustment']: logger.error("Overall Test Failed: Final adjustment did not succeed."); return False
        
        logger.info("All intended stages reported success.")
        return True

    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_value = self.metrics.get('actualPrice_pool')
        if isinstance(metric_value, Decimal): self.metrics['actualPrice_pool'] = f"{metric_value}"
        elif metric_value is None: self.metrics['actualPrice_pool'] = ""
        
        # Preserve your original column order and add new fields at the end
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
        
        row_data = {}
        for col in columns:
            val = self.metrics.get(col)
            if val is None:
                if col in ['sqrtPriceX96_pool', 'currentTick_pool', 'targetTickLower_calculated',
                           'targetTickUpper_calculated',
                           'initial_contract_balance_token0',
                           'initial_contract_balance_token1', 'finalTickLower_contract',
                           'finalTickUpper_contract', 'liquidity_contract',
                           'amount0_provided_to_mint', 'amount1_provided_to_mint',
                           'fees_collected_token0', 'fees_collected_token1',
                           'fees_collected_token0_via_collect_only',
                           'fees_collected_token1_via_collect_only',
                           'gas_used', 'gas_cost_eth', 'range_width_multiplier_setting',
                           'predictedTick_calculated']:
                    row_data[col] = 0 if col not in ['predictedPrice_api', 'external_api_eth_price'] else ""
                elif col not in ['tx_hash', 'error_message', 'action_taken', 'contract_type', 'timestamp', 'actualPrice_pool', 'external_api_eth_price']:
                    row_data[col] = ""
                elif self.metrics.get(col) is None :
                    row_data[col] = ""
            else:
                if col == 'external_api_eth_price' and isinstance(val, float): row_data[col] = f"{val:.2f}"
                elif col == 'gas_cost_eth' and isinstance(val, float): row_data[col] = f"{val:.18f}"
                else: row_data[col] = val
        
        try:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_exists = RESULTS_FILE.is_file()

            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Predictive metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save predictive metrics: {e}")

# --- Main Function ---
def main():
    # YOUR ORIGINAL main FUNCTION - UNCHANGED
    logger.info("=" * 50)
    logger.info("Starting Predictive Liquidity Manager Test on Fork")
    logger.info("=" * 50)

    if not web3_utils.init_web3():
        logger.critical("Web3 initialization failed. Exiting predictive test.")
        return
    
    if not web3_utils.w3 or not web3_utils.w3.is_connected():
        logger.critical("web3_utils.w3 instance not available or not connected after init. Exiting predictive test.")
        return

    predictive_address = None
    test_instance = None
    try:
        predictive_address_env = os.getenv('PREDICTIVE_MANAGER_ADDRESS')
        if not predictive_address_env :
            logger.error(f"PREDICTIVE_MANAGER_ADDRESS environment variable not found.")
            temp_test_for_error_log = PredictiveTest(contract_address="0x"+"0"*40)
            temp_test_for_error_log.metrics['action_taken'] = temp_test_for_error_log.ACTION_STATES["SETUP_FAILED"]
            temp_test_for_error_log.metrics['error_message'] = "PREDICTIVE_MANAGER_ADDRESS env var not found"
            temp_test_for_error_log.save_metrics()
            raise ValueError("PREDICTIVE_MANAGER_ADDRESS env var not found")
        
        predictive_address = Web3.to_checksum_address(predictive_address_env)
        logger.info(f"Loaded Predictive Manager Address from ENV: {predictive_address}")

        test_instance = PredictiveTest(predictive_address)
        desired_rwm = int(os.getenv('PREDICTIVE_RWM', '100'))
        target_weth = float(os.getenv('PREDICTIVE_TARGET_WETH', '5.0'))
        target_usdc = float(os.getenv('PREDICTIVE_TARGET_USDC', '10000.0'))

        test_instance.execute_test_steps(
            desired_range_width_multiplier=desired_rwm,
            target_weth_balance=target_weth,
            target_usdc_balance=target_usdc
        )

    except FileNotFoundError as e:
        logger.error(f"Setup Error - Address file not found: {e}")
        if test_instance is None: addr_for_dummy = predictive_address if predictive_address and Web3.is_address(str(predictive_address)) else "0x"+"0"*40; test_instance = PredictiveTest(addr_for_dummy)
        if hasattr(test_instance, 'metrics'): test_instance.metrics['action_taken'] = test_instance.ACTION_STATES["SETUP_FAILED"]; test_instance.metrics['error_message'] = str(e); test_instance.save_metrics()
        
    except ValueError as e:
        logger.error(f"Configuration Error - Problem reading address or address key missing: {e}")
        if test_instance is None: addr_for_dummy = predictive_address if predictive_address and Web3.is_address(str(predictive_address)) else "0x"+"0"*40; test_instance = PredictiveTest(addr_for_dummy)
        if hasattr(test_instance, 'metrics'): test_instance.metrics['action_taken'] = test_instance.ACTION_STATES["SETUP_FAILED"]; test_instance.metrics['error_message'] = str(e); test_instance.save_metrics()
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred during predictive main execution:")
        
        if test_instance is None:
            if predictive_address and Web3.is_address(predictive_address):
                test_instance = PredictiveTest(predictive_address)
            else:
                test_instance = PredictiveTest("0x"+"0"*40)

        if hasattr(test_instance, 'metrics'):
            test_instance.metrics['action_taken'] = test_instance.ACTION_STATES["UNEXPECTED_ERROR"]
            test_instance.metrics['error_message'] = (test_instance.metrics.get('error_message',"") + f"; MainException: {str(e)}").strip(";")
            test_instance.save_metrics()
    finally:
        logger.info("=" * 50)
        logger.info("Predictive test run finished.")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()