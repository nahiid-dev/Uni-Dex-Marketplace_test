# baseline_test.py (FINAL-FIXED with "Set and Forget" Logic and Safer Market Maker)
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
from web3.exceptions import ContractLogicError

# --- Adjust path imports ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import test.utils.web3_utils as web3_utils
import test.utils.contract_funder as contract_funder

getcontext().prec = 78

try:
    from test.utils.test_base import LiquidityTestBase
    from test.utils.web3_utils import send_transaction, get_contract
except ImportError as e:
    print(f"ERROR importing from test.utils in baseline_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272
MAX_TICK_CONST = 887272
Q96 = Decimal(2**96)
TOKEN0_DECIMALS = 6
TOKEN1_DECIMALS = 18
UNISWAP_V3_ROUTER_ADDRESS = "0xE592427A0AEce92De3Edee1F18E0157C05861564" # Mainnet Router
LSTM_API_URL = os.getenv('LSTM_API_URL', 'http://95.216.156.73:5000/predict_price?symbol=ETHUSDT&interval=1h')

TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV = os.getenv('TOKEN_MANAGER_OPTIMIZED_ADDRESS')

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

    def __init__(self, contract_address: str):
        super().__init__(contract_address, "BaselineMinimal")
        self.ACTION_STATES = {
            "INIT": "init", "SETUP_FAILED": "setup_failed",
            "POOL_READ_FAILED": "pool_read_failed", "CALCULATION_FAILED": "calculation_failed",
            "SKIPPED_PROXIMITY": "skipped_proximity", "FUNDING_FAILED": "funding_failed",
            "TX_SENT": "tx_sent", "TX_SUCCESS_ADJUSTED": "tx_success_adjusted",
            "TX_SUCCESS_SKIPPED_ONCHAIN": "tx_success_skipped_onchain",
            "TX_REVERTED": "tx_reverted", "TX_WAIT_FAILED": "tx_wait_failed",
            "METRICS_UPDATE_FAILED": "metrics_update_failed", "UNEXPECTED_ERROR": "unexpected_error",
            "TX_SUCCESS_ADJUSTED_INITIAL": "tx_success_adjusted_initial",
            "TX_SUCCESS_SWAP_FEES": "tx_success_swap_fees",
            "SWAP_FOR_FEES_FAILED": "swap_for_fees_failed",
            "FEES_COLLECT_ONLY_SUCCESS": "fees_collect_only_success",
            "FEES_COLLECT_ONLY_FAILED": "fees_collect_only_failed",
            "API_FAILED": "api_failed"
        }
        self.metrics = self._reset_metrics()
        self.factory_contract = None
        self.tick_spacing = None
        self.pool_address = None
        self.pool_contract = None
        self.token0_contract_instance = None
        self.token1_contract_instance = None
        
        self.nft_position_manager_address = None
        self.nft_manager_contract_for_events = None

        self.token_manager_optimized_address = TOKEN_MANAGER_OPTIMIZED_ADDRESS_ENV

    def _reset_metrics(self):
        base_metrics = super()._reset_metrics()
        baseline_specific_metrics = {
            'contract_type': 'Baseline',
            'action_taken': self.ACTION_STATES["INIT"] if hasattr(self, 'ACTION_STATES') else "init",
            'predictedPrice_api': None,
            'predictedTick_calculated': None,
            'num_swaps_executed': 0,
            'targetTickLower_offchain': None,
            'targetTickUpper_offchain': None,
            'currentTickLower_contract': None,
            'currentTickUpper_contract': None,
            'currentLiquidity_contract': None,
            'finalTickLower_contract': None,
            'finalTickUpper_contract': None,
            'finalLiquidity_contract': None,
            'fees_collected_token0_via_collect_only': None,
            'fees_collected_token1_via_collect_only': None
        }
        final_metrics = {**base_metrics, **baseline_specific_metrics}
        original_keys_defaults = {
            'timestamp': None, 'tx_hash': None, 'range_width_multiplier_setting': None,
            'external_api_eth_price': None, 'actualPrice_pool': None,
            'sqrtPriceX96_pool': None, 'currentTick_pool': None,
            'initial_contract_balance_token0': None, 'initial_contract_balance_token1': None,
            'amount0_provided_to_mint': None, 'amount1_provided_to_mint': None,
            'fees_collected_token0': None, 'fees_collected_token1': None,
            'gas_used': None, 'gas_cost_eth': None, 'error_message': ""
        }
        for key, default_value in original_keys_defaults.items():
            if key not in final_metrics:
                final_metrics[key] = default_value
        return final_metrics

    def setup(self, desired_range_width_multiplier: int) -> bool:
        if not super().setup(desired_range_width_multiplier):
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = "Base setup failed"
            return False
        try:
            if not web3_utils.w3:
                logger.error("web3_utils.w3 not available in BaselineTest setup after base.setup()")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "web3_utils.w3 unavailable post base setup"
                return False

            try:
                self.nft_position_manager_address = Web3.to_checksum_address(
                    self.contract.functions.positionManager().call()
                )
                logger.info(f"BaselineTest.setup: Uniswap INonfungiblePositionManager address from OUR contract: {self.nft_position_manager_address}")
            except Exception as e_fetch_nfpm:
                logger.error(f"BaselineTest.setup: Failed to fetch INonfungiblePositionManager address from our contract: {e_fetch_nfpm}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Failed to get NFPM address: {str(e_fetch_nfpm)}"
                return False

            if self.nft_position_manager_address and self.nft_position_manager_address != ('0x' + '0'*40):
                nft_manager_abi = json.loads('[{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"},{"indexed":false,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1","type":"uint256"}],"name":"Collect","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"},{"indexed":false,"internalType":"uint128","name":"liquidity","type":"uint128"},{"indexed":false,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1","type":"uint256"}],"name":"IncreaseLiquidity","type":"event"}]')
                try:
                    self.nft_manager_contract_for_events = web3_utils.w3.eth.contract(
                        address=self.nft_position_manager_address,
                        abi=nft_manager_abi
                    )
                    logger.info(f"BaselineTest.setup: Initialized contract instance for INonfungiblePositionManager events at {self.nft_position_manager_address}.")
                except Exception as e_init_nft_events:
                    logger.error(f"BaselineTest.setup: Failed to create contract instance for NFPM events using address '{self.nft_position_manager_address}': {e_init_nft_events}")
                    self.nft_manager_contract_for_events = None
            else:
                logger.warning(f"BaselineTest.setup: self.nft_position_manager_address is not validly set. Cannot initialize for NFPM events.")
                self.nft_manager_contract_for_events = None

            factory_address = self.contract.functions.factory().call()
            self.factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            if not self.factory_contract:
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "Factory contract not found"
                return False

            fee = self.contract.functions.fee().call()
            self.pool_address = self.factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Baseline pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
            
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            if not self.pool_contract:
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "Pool contract object could not be created"
                return False
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
                'chainId': web3_utils.w3.eth.chain_id
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
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";Failed to set RWM for Baseline (tx: {tx_hash_rwm_str})").strip(";")
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Baseline contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}"
            return False

    def get_position_info(self) -> dict | None:
        if not self.contract:
            logger.error(f"({self.contract_name}) Contract not initialized. Run setup first (Baseline Override).")
            return None
        if not web3_utils.w3:
            logger.error(f"({self.contract_name}) Web3 not connected in get_position_info (Baseline Override).")
            return None
        
        try:
            pos_data = None
            if hasattr(self.contract.functions, 'currentPosition'):
                pos_data = self.contract.functions.currentPosition().call()
            elif hasattr(self.contract.functions, 'getCurrentPosition'):
                pos_data = self.contract.functions.getCurrentPosition().call()
            else:
                logger.error(f"({self.contract_name}) No known position info method (currentPosition, getCurrentPosition) found on contract (Baseline Override).")
                return None

            if pos_data and len(pos_data) == 5:
                position = {
                    'tokenId': pos_data[0],
                    'active': pos_data[1],
                    'tickLower': pos_data[2],
                    'tickUpper': pos_data[3],
                    'liquidity': pos_data[4]
                }
                logger.debug(f"({self.contract_name}) Fetched Position Info (Baseline Override): {position}")
                return position
            else:
                logger.error(f"({self.contract_name}) Position data not found or empty for BaselineMinimal (Baseline Override).")
                return None

        except Exception as e:
            logger.exception(f"({self.contract_name}) Failed to get position info from contract (Baseline Override): {e}")
            return None

    def sqrt_price_x96_to_price_token0_in_token1(self, sqrt_price_x96_str: str) -> Decimal:
        sqrt_price_x96 = Decimal(sqrt_price_x96_str)
        price_t1_in_t0 = (sqrt_price_x96 / Q96) ** 2
        if price_t1_in_t0 == Decimal(0): return Decimal("inf") if sqrt_price_x96 != Decimal(0) else Decimal(0)
        price_t0_in_t1 = Decimal(1) / price_t1_in_t0
        decimals_adjustment = Decimal(10) ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS)
        if decimals_adjustment == Decimal(0): return Decimal("-1")
        return price_t0_in_t1 / decimals_adjustment

    def get_pool_state(self) -> tuple[int | None, int | None]:
        if not self.pool_contract:
            logger.error("Pool contract not initialized for get_pool_state (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = "Pool contract missing for get_pool_state"
            return None, None
        if not web3_utils.w3:
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
    
    def get_predicted_price_from_api(self) -> float | None:
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
            if 'data' in locals():
                data_for_log = data
            logger.exception(f"Error processing API response from {LSTM_API_URL}. Data: {data_for_log}. Error: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Response Processing Error: {str(e)}"
            return None

    def calculate_tick_from_price(self, price: float) -> int | None:
        if self.token0_decimals is None or self.token1_decimals is None:
            logger.error("Token decimals not available for tick calculation. Ensure LiquidityTestBase.setup() was successful.")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Token decimals missing for tick calc"
            return None
            
        try:
            price_decimal = Decimal(str(price))
            effective_sqrt_price_arg = (Decimal(1) / price_decimal) * (Decimal(10)**(self.token1_decimals - self.token0_decimals))
            
            if effective_sqrt_price_arg <= 0:
                logger.error(f"Argument for sqrt in tick calculation is non-positive: {effective_sqrt_price_arg}")
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
            
            logger.info(f"Tick calculation details (for Market Sim):")
            logger.info(f"  Input price (assumed T1/T0, e.g. WETH/USDC): {price}")
            logger.info(f"  Final tick: {tick}")
            
            self.metrics['predictedTick_calculated'] = tick
            return tick
        except Exception as e:
            logger.exception(f"Failed to calculate predicted tick from price {price}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Tick Calculation Error: {str(e)}"
            return None

    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]:
        if self.tick_spacing is None or current_tick is None:
            logger.error("Tick spacing or current_tick not available for target tick calculation (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Missing data for target tick calc (Baseline)"
            return None, None
        try:
            width_multiplier = self.metrics.get('range_width_multiplier_setting')
            if width_multiplier is None or width_multiplier <= 0: width_multiplier = self.contract.functions.rangeWidthMultiplier().call()
            if width_multiplier is None or width_multiplier <= 0:
                logger.error(f"Invalid rangeWidthMultiplier ({width_multiplier}) for tick calculation.")
                self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
                self.metrics['error_message'] = f"Invalid RWM ({width_multiplier}) for tick calc"
                return None, None
            
            half_total_tick_width = (self.tick_spacing * width_multiplier) // 2
            
            if half_total_tick_width < self.tick_spacing:
                logger.warning(f"Calculated half width {half_total_tick_width} is less than tick spacing {self.tick_spacing}. Using minimum width.")
                half_total_tick_width = self.tick_spacing

            target_lower_tick = ((current_tick - half_total_tick_width) // self.tick_spacing) * self.tick_spacing
            target_upper_tick = ((current_tick + half_total_tick_width) // self.tick_spacing) * self.tick_spacing

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

    def _get_contract_token_balances_individually(self):
        balance0_wei, balance1_wei = 0, 0
        try:
            if self.token0:
                token0_contract_instance = web3_utils.get_contract(self.token0, "IERC20")
                if token0_contract_instance: balance0_wei = token0_contract_instance.functions.balanceOf(self.contract_address).call()
            if self.token1:
                token1_contract_instance = web3_utils.get_contract(self.token1, "IERC20")
                if token1_contract_instance: balance1_wei = token1_contract_instance.functions.balanceOf(self.contract_address).call()
        except Exception as e:
            logger.warning(f"Could not read contract token balances individually (Baseline): {e}")
        return balance0_wei, balance1_wei

    def _perform_one_swap(self, funding_account, private_key_env, token_in_addr: str, token_out_addr: str, amount_in_readable: Decimal, token_in_decimals: int) -> bool:
        """Performs a single exact-input swap via the Uniswap V3 Router."""
        try:
            router_contract = web3_utils.get_contract(UNISWAP_V3_ROUTER_ADDRESS, "ISwapRouter")
            if not router_contract:
                logger.error("Failed to get Router contract instance for a single swap.")
                return False

            amount_in_wei = int(amount_in_readable * (Decimal(10) ** token_in_decimals))
            pool_fee = self.contract.functions.fee().call()
            
            current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            swap_params = {
                'tokenIn': Web3.to_checksum_address(token_in_addr),
                'tokenOut': Web3.to_checksum_address(token_out_addr),
                'fee': pool_fee,
                'recipient': funding_account.address,
                'deadline': int(time.time()) + 600,
                'amountIn': amount_in_wei,
                'amountOutMinimum': 0,
                'sqrtPriceLimitX96': 0
            }

            swap_tx_params = {
                'from': funding_account.address,
                'nonce': current_nonce,
                'gas': 250000
            }
            
            built_swap_tx = router_contract.functions.exactInputSingle(swap_params).build_transaction(swap_tx_params)
            receipt_swap = web3_utils.send_transaction(built_swap_tx, private_key_env)

            if receipt_swap and receipt_swap.status == 1:
                logger.info(f"Single swap successful. Tx: {receipt_swap.transactionHash.hex()}")
                return True
            else:
                logger.error(f"Single swap failed. Receipt: {receipt_swap}")
                return False

        except Exception as e:
            logger.exception(f"Error during _perform_one_swap: {e}")
            return False

    def _call_collect_fees_only(self, funding_account, private_key_env) -> bool:
        logger.info("Attempting to call collectCurrentFeesOnly() on BaselineMinimal contract...")
        if not self.contract:
            logger.error("BaselineMinimal contract instance not available for collectCurrentFeesOnly.")
            return False
        
        if self.metrics.get('fees_collected_token0') is None: self.metrics['fees_collected_token0'] = 0
        if self.metrics.get('fees_collected_token1') is None: self.metrics['fees_collected_token1'] = 0
        if 'fees_collected_token0_via_collect_only' in self.metrics and self.metrics.get('fees_collected_token0_via_collect_only') is None:
            self.metrics['fees_collected_token0_via_collect_only'] = 0
        if 'fees_collected_token1_via_collect_only' in self.metrics and self.metrics.get('fees_collected_token1_via_collect_only') is None:
            self.metrics['fees_collected_token1_via_collect_only'] = 0

        tx_call = self.contract.functions.collectCurrentFeesOnly()
        current_nonce = web3_utils.w3.eth.get_transaction_count(funding_account.address)
        tx_params = {'from': funding_account.address, 'nonce': current_nonce, 'chainId': web3_utils.w3.eth.chain_id}
        try:
            tx_params['gas'] = int(tx_call.estimate_gas({'from': funding_account.address}) * 1.25)
        except ContractLogicError as cle:
            logger.error(f"Gas estimation for collectCurrentFeesOnly (Baseline) failed (Logic Error): {cle}. This might indicate an issue like no active position or other require failure.")
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";CollectFeesGasEstLogicError(Baseline): {str(cle)[:100]}").strip(';')
            return False
        except Exception as e:
            tx_params['gas'] = 300000
            logger.warning(f"Gas estimation for collectCurrentFeesOnly (Baseline) failed: {e}. Using default {tx_params['gas']}.")

        try:
            built_tx = tx_call.build_transaction(tx_params)
            receipt = web3_utils.send_transaction(built_tx, private_key_env)

            if receipt and receipt.status == 1:
                logger.info(f"Baseline collectCurrentFeesOnly() transaction successful. Tx: {receipt.transactionHash.hex()}")
                if "FEES_COLLECT_ONLY_SUCCESS" in self.ACTION_STATES:
                    self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_SUCCESS"]
                
                amount0_from_our_event, amount1_from_our_event = 0,0
                
                our_fee_event_logs = self.contract.events.FeesOnlyCollected().process_receipt(receipt, errors=DISCARD)
                
                if our_fee_event_logs:
                    log_args = our_fee_event_logs[0].args
                    logger.info(f"BaselineContract FeesOnlyCollected Event: TokenId={log_args.tokenId}, Amount0={log_args.amount0Collected}, Amount1={log_args.amount1Collected}, Success={log_args.success}")
                    if log_args.success:
                        amount0_from_our_event = log_args.amount0Collected
                        amount1_from_our_event = log_args.amount1Collected
                    else:
                        logger.warning("Baseline FeesOnlyCollected event reported success=false from contract.")
                else:
                    logger.warning("No FeesOnlyCollected event found from Baseline contract after collectCurrentFeesOnly call.")

                if 'fees_collected_token0_via_collect_only' in self.metrics:
                    self.metrics['fees_collected_token0_via_collect_only'] = amount0_from_our_event
                if 'fees_collected_token1_via_collect_only' in self.metrics:
                    self.metrics['fees_collected_token1_via_collect_only'] = amount1_from_our_event
                
                if self.metrics.get('fees_collected_token0') is None: self.metrics['fees_collected_token0'] = 0
                if self.metrics.get('fees_collected_token1') is None: self.metrics['fees_collected_token1'] = 0
                self.metrics['fees_collected_token0'] += amount0_from_our_event
                self.metrics['fees_collected_token1'] += amount1_from_our_event
                
                logger.info(f"Baseline Main fee metrics updated: T0={self.metrics['fees_collected_token0']}, T1={self.metrics['fees_collected_token1']}")

                if self.nft_manager_contract_for_events:
                    logger.info(f"Checking Uniswap NFPM Collect events in Tx: {receipt.transactionHash.hex()} after Baseline collectCurrentFeesOnly call...")
                    
                    collect_logs_nfpm = self.nft_manager_contract_for_events.events.Collect().process_receipt(receipt, errors=DISCARD)
                    
                    found_nfpm_collect = False
                    for nfpm_log_entry in collect_logs_nfpm:
                        if nfpm_log_entry.args.recipient.lower() == self.contract_address.lower():
                            found_nfpm_collect = True
                            logger.info(f"Uniswap NFPM Collect Event (Post Baseline collectCurrentFeesOnly, TokenId={nfpm_log_entry.args.tokenId}): Amount0={nfpm_log_entry.args.amount0}, Amount1={nfpm_log_entry.args.amount1}")
                            if not our_fee_event_logs or not our_fee_event_logs[0].args.success :
                                if 'fees_collected_token0_via_collect_only' in self.metrics:
                                    self.metrics['fees_collected_token0_via_collect_only'] = nfpm_log_entry.args.amount0
                                if 'fees_collected_token1_via_collect_only' in self.metrics:
                                    self.metrics['fees_collected_token1_via_collect_only'] = nfpm_log_entry.args.amount1
                                logger.info("Updated via_collect_only fee metrics from NFPM event due to issue/absence of contract's event.")
                            elif log_args.amount0Collected != nfpm_log_entry.args.amount0 or log_args.amount1Collected != nfpm_log_entry.args.amount1:
                                logger.warning(f"Discrepancy (Baseline): OurEvent(0:{log_args.amount0Collected}, 1:{log_args.amount1Collected}) vs NFPMEvent(0:{nfpm_log_entry.args.amount0}, 1:{nfpm_log_entry.args.amount1}).")
                            break
                    if not found_nfpm_collect:
                        logger.info("No direct Uniswap NFPM Collect event found with Baseline contract as recipient in this transaction.")
                return True
            else:
                tx_hash_str = receipt.transactionHash.hex() if receipt else "N/A"
                logger.error(f"Baseline collectCurrentFeesOnly() transaction failed. Tx: {tx_hash_str}")
                if "FEES_COLLECT_ONLY_FAILED" in self.ACTION_STATES:
                    self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
                self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";BaselineCollectFeesTxFailed: {tx_hash_str}").strip(';')
                return False
        except Exception as e:
            logger.exception(f"Exception during Baseline _call_collect_fees_only: {e}")
            if "FEES_COLLECT_ONLY_FAILED" in self.ACTION_STATES:
                self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]
            self.metrics['error_message'] = (self.metrics.get('error_message', "") + f";BaselineCollectFeesException: {str(e)[:100]}").strip(';')
            return False

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float):
        self.metrics = self._reset_metrics()
        try:
            self.get_current_eth_price()
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from baseline contract for metrics.")

        stage_results = {
            'initial_adjustment': False,
            'swap': False,
            'collect_only': False
        }
        
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position (Baseline).")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx (Baseline)"
            self.save_metrics()
            sys.exit(1)
        funding_account = Account.from_key(private_key_env)
        
        try:
            pos_info_before_all = self.get_position_info()
            if pos_info_before_all:
                self.metrics['currentTickLower_contract'] = pos_info_before_all.get('tickLower',0)
                self.metrics['currentTickUpper_contract'] = pos_info_before_all.get('tickUpper',0)
                self.metrics['currentLiquidity_contract'] = pos_info_before_all.get('liquidity', 0)

            # --- STAGE 1: Initial Position Adjustment ("Set and Forget") ---
            logger.info("\n--- STAGE 1: Baseline Strategy - Initial 'Set and Forget' position adjustment ---")
            
            logger.info("Ensuring precise token balances for Baseline contract (initial)...")
            if not contract_funder.ensure_precise_token_balances(
                self.contract_address, self.token0, self.token0_decimals, target_usdc_balance,
                self.token1, self.token1_decimals, target_weth_balance, private_key_env
            ):
                raise Exception("Precise funding for Baseline contract failed (initial).")
            
            balance0_val, balance1_val = self._get_contract_token_balances_individually()
            self.metrics['initial_contract_balance_token0'] = balance0_val
            self.metrics['initial_contract_balance_token1'] = balance1_val
            logger.info(f"Initial contract balances for baseline metrics (after funding): Token0={balance0_val}, Token1={balance1_val}")

            logger.info(f"Calling adjustLiquidityWithCurrentPrice for Baseline contract (initial)...")
            tx_function_call_base_initial = self.contract.functions.adjustLiquidityWithCurrentPrice()
            current_nonce_initial_adjust = web3_utils.w3.eth.get_transaction_count(funding_account.address)
            tx_params_base_initial = {'from': funding_account.address, 'nonce': current_nonce_initial_adjust, 'chainId': web3_utils.w3.eth.chain_id}
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
                stage_results['initial_adjustment'] = True
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED_INITIAL"]
                self.metrics['gas_used'] = receipt_base_initial.get('gasUsed', 0)
                effective_gas_price_initial = receipt_base_initial.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(web3_utils.w3.from_wei(self.metrics['gas_used'] * effective_gas_price_initial, 'ether'))

                logger.info("Processing PositionMinted event to capture mint amounts for Baseline...")
                try:
                    mint_logs = self.contract.events.PositionMinted().process_receipt(receipt_base_initial, errors=DISCARD)
                    if mint_logs:
                        log_args = mint_logs[0].args
                        self.metrics['amount0_provided_to_mint'] = log_args.amount0Actual
                        self.metrics['amount1_provided_to_mint'] = log_args.amount1Actual
                        logger.info(f"Captured Mint Amounts from PositionMinted Event: Token0={log_args.amount0Actual}, Token1={log_args.amount1Actual}")
                    else:
                        logger.warning("No PositionMinted event found from Baseline contract.")
                except Exception as e:
                    logger.warning(f"Could not process PositionMinted event for Baseline initial adjustment: {e}")
            else:
                self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"] if receipt_base_initial else self.ACTION_STATES["TX_WAIT_FAILED"]
                raise Exception("Baseline initial adjustment transaction failed")

            # --- STAGE 2: Intelligent Market Simulation (Position is now fixed) ---
            if stage_results['initial_adjustment']:
                logger.info(f"\n--- STAGE 2: Simulating Intelligent Market Activity (for Baseline Test) ---")

                predicted_price = self.get_predicted_price_from_api()
                if predicted_price is None:
                    raise Exception("Failed to get predicted price for market simulation.")
                predicted_tick = self.calculate_tick_from_price(predicted_price)
                if predicted_tick is None:
                    raise Exception("Failed to calculate predicted tick for market simulation.")

                realignment_swap_amount_eth = Decimal(os.getenv('REALIGNMENT_SWAP_AMOUNT_ETH', '5.0'))
                realignment_swap_amount_usdc = Decimal(os.getenv('REALIGNMENT_SWAP_AMOUNT_USDC', '15000'))
                volatility_swap_amount_eth = Decimal(os.getenv('VOLATILITY_SWAP_AMOUNT_ETH', '15.0'))
                volatility_swap_amount_usdc = Decimal(os.getenv('VOLATILITY_SWAP_AMOUNT_USDC', '45000'))
                num_volatility_swaps = int(os.getenv('NUM_VOLATILITY_SWAPS', '10'))
                num_realignment_swaps = 15

                logger.info("Pre-approving router for all swaps in simulation...")
                token0_contract = web3_utils.get_contract(self.token0, "IERC20")
                token1_contract = web3_utils.get_contract(self.token1, "IERC20")
                
                total_usdc_to_approve = (realignment_swap_amount_usdc * num_realignment_swaps) + (volatility_swap_amount_usdc * num_volatility_swaps)
                total_weth_to_approve = (realignment_swap_amount_eth * num_realignment_swaps) + (volatility_swap_amount_eth * num_volatility_swaps)

                approve_tx0 = token0_contract.functions.approve(UNISWAP_V3_ROUTER_ADDRESS, int(total_usdc_to_approve * 10**self.token0_decimals)).build_transaction({'from': funding_account.address, 'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address), 'gas': 100000})
                web3_utils.send_transaction(approve_tx0, private_key_env)
                
                approve_tx1 = token1_contract.functions.approve(UNISWAP_V3_ROUTER_ADDRESS, int(total_weth_to_approve * 10**self.token1_decimals)).build_transaction({'from': funding_account.address, 'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address), 'gas': 100000})
                web3_utils.send_transaction(approve_tx1, private_key_env)
                logger.info("Router pre-approved for both USDC and WETH.")

                all_swaps_successful = True
                realignment_swaps_performed = 0
                volatility_swaps_performed = 0
                
                logger.info(f"--- Phase 1: Realigning price towards target tick: {predicted_tick} ---")
                _, current_tick = self.get_pool_state()

                for i in range(num_realignment_swaps):
                    if current_tick > predicted_tick:
                        logger.info(f"Realign Swap {i+1}/{num_realignment_swaps}: Current tick {current_tick} > target {predicted_tick}. Buying WETH.")
                        token_in, token_out = self.token0, self.token1
                        amount, decimals = realignment_swap_amount_usdc, self.token0_decimals
                    else:
                        logger.info(f"Realign Swap {i+1}/{num_realignment_swaps}: Current tick {current_tick} < target {predicted_tick}. Selling WETH.")
                        token_in, token_out = self.token1, self.token0
                        amount, decimals = realignment_swap_amount_eth, self.token1_decimals
                        
                    if not self._perform_one_swap(funding_account, private_key_env, token_in, token_out, amount, decimals):
                        all_swaps_successful = False
                        break
                    realignment_swaps_performed +=1
                    time.sleep(1)
                    _, current_tick = self.get_pool_state()

                if not all_swaps_successful:
                    raise Exception("Market realignment (Phase 1) failed.")
                logger.info(f"Phase 1 finished after {realignment_swaps_performed} swaps. Current tick: {current_tick}")

                logger.info(f"--- Phase 2: Generating focused volatility around tick {current_tick} ---")
                for i in range(num_volatility_swaps):
                    logger.info(f"Volatility Swap Pair {i + 1}/{num_volatility_swaps}")
                    
                    if not self._perform_one_swap(funding_account, private_key_env, self.token1, self.token0, volatility_swap_amount_eth, self.token1_decimals):
                        all_swaps_successful = False
                        break
                    volatility_swaps_performed += 1
                    time.sleep(1)
                    
                    if not self._perform_one_swap(funding_account, private_key_env, self.token0, self.token1, volatility_swap_amount_usdc, self.token0_decimals):
                        all_swaps_successful = False
                        break
                    volatility_swaps_performed += 1
                    time.sleep(1)

                stage_results['swap'] = all_swaps_successful
                self.metrics['num_swaps_executed'] = realignment_swaps_performed + volatility_swaps_performed

                if not all_swaps_successful:
                    self.metrics['action_taken'] = self.ACTION_STATES["SWAP_FOR_FEES_FAILED"]
                    raise Exception("Focused volatility simulation (Phase 2) failed for Baseline test.")
                else:
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_SWAP_FEES"]

            if stage_results['initial_adjustment'] and stage_results['swap']:
                logger.info("\n--- STAGE 2.5: Baseline Strategy - Explicit Fee Collection ---")
                stage_results['collect_only'] = self._call_collect_fees_only(funding_account, private_key_env)
                if not stage_results['collect_only']:
                    logger.warning("Fee Collection failed, but continuing test.")
            
            logger.info("\n--- STAGE 3: Baseline Strategy - Final state check ---")
            self.metrics['action_taken'] = self.ACTION_STATES["FEES_COLLECT_ONLY_SUCCESS"] if stage_results.get('collect_only') else self.ACTION_STATES["FEES_COLLECT_ONLY_FAILED"]

        except Exception as e:
            logger.exception(f"An exception occurred during Baseline adjust_position: {e}")
            if self.metrics.get("action_taken") == "init":
                self.metrics["action_taken"] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            if not self.metrics.get("error_message"):
                self.metrics["error_message"] = str(e)
            
        finally:
            logger.info("Updating final position info in 'finally' block for Baseline (absolute final state)...")
            final_pos_info = self.get_position_info()
            if final_pos_info:
                self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower',0)
                self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper',0)
                self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity',0)
            else:
                logger.warning("Baseline: Could not get final position info from contract in finally block.")
            
            _, final_tick = self.get_pool_state()
            if final_tick is not None:
                self.calculate_target_ticks_offchain(final_tick)

            self.save_metrics()

    def save_metrics(self):
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_value = self.metrics.get('actualPrice_pool')
        if isinstance(metric_value, Decimal):
            self.metrics['actualPrice_pool'] = f"{metric_value}"
        elif metric_value is None:
            self.metrics['actualPrice_pool'] = ""
        
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'range_width_multiplier_setting',
            'predictedPrice_api', 'predictedTick_calculated',
            'external_api_eth_price',
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_offchain', 'targetTickUpper_offchain',
            'initial_contract_balance_token0', 'initial_contract_balance_token1',
            'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
            'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract',
            'amount0_provided_to_mint', 'amount1_provided_to_mint',
            'fees_collected_token0', 'fees_collected_token1',
            'num_swaps_executed',
            'gas_used', 'gas_cost_eth', 'error_message'
        ]
        
        row_data = {}
        for col in columns:
            val = self.metrics.get(col)
            if val is None:
                if col in ['sqrtPriceX96_pool', 'currentTick_pool', 'targetTickLower_offchain', 'targetTickUpper_offchain',
                           'initial_contract_balance_token0', 'initial_contract_balance_token1',
                           'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
                           'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract',
                           'amount0_provided_to_mint', 'amount1_provided_to_mint',
                           'fees_collected_token0', 'fees_collected_token1',
                           'fees_collected_token0_via_collect_only', 'fees_collected_token1_via_collect_only',
                           'gas_used', 'gas_cost_eth', 'range_width_multiplier_setting',
                           'predictedTick_calculated', 'num_swaps_executed']:
                    row_data[col] = 0 if col not in ['predictedPrice_api', 'external_api_eth_price'] else ""
                elif col not in ['tx_hash', 'error_message', 'action_taken', 'contract_type', 'timestamp', 'actualPrice_pool', 'external_api_eth_price']:
                    row_data[col] = ""
                elif self.metrics.get(col) is None :
                    row_data[col] = ""
            else:
                if col == 'external_api_eth_price' and isinstance(val, float):
                    row_data[col] = f"{val:.2f}"
                elif col == 'gas_cost_eth' and isinstance(val, float):
                    row_data[col] = f"{val:.18f}"
                else:
                    row_data[col] = val
        
        try:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_exists = RESULTS_FILE.is_file()
            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Baseline metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save baseline metrics: {e}")

def main():
    logger.info("="*50)
    logger.info("Starting Baseline Minimal Liquidity Manager Test on Fork")
    logger.info("="*50)

    if not web3_utils.init_web3():
        logger.critical("Web3 initialization failed. Exiting baseline test.")
        sys.exit(1)
        
    if not web3_utils.w3:
        logger.critical("web3_utils.w3 instance not available after init. Exiting baseline test.")
        sys.exit(1)

    baseline_address_val = None
    test_baseline = None
    try:
        baseline_address_env = os.getenv('BASELINE_MINIMAL_ADDRESS')
        if not baseline_address_env:
            logger.error(f"BASELINE_MINIMAL_ADDRESS not found in environment variables.")
            temp_test_b_error = BaselineTest("0x" + "0" * 40)
            temp_test_b_error.metrics['action_taken'] = temp_test_b_error.ACTION_STATES["SETUP_FAILED"]
            temp_test_b_error.metrics['error_message'] = "BASELINE_MINIMAL_ADDRESS env var not found"
            temp_test_b_error.save_metrics()
            raise ValueError("BASELINE_MINIMAL_ADDRESS env var not found")
        
        baseline_address_val = Web3.to_checksum_address(baseline_address_env)
        logger.info(f"Loaded Baseline Minimal Address from ENV: {baseline_address_val}")
        
        test_baseline = BaselineTest(baseline_address_val)
        
        desired_rwm_base = int(os.getenv('BASELINE_RWM', '20'))
        target_weth_for_test = float(os.getenv('BASELINE_TARGET_WETH', '50.0'))
        target_usdc_for_test = float(os.getenv('BASELINE_TARGET_USDC', '2000000.0'))
        
        setup_success = test_baseline.setup(desired_range_width_multiplier=desired_rwm_base)
        
        if setup_success:
            test_baseline.adjust_position(
                target_weth_balance=target_weth_for_test,
                target_usdc_balance=target_usdc_for_test
            )
        else:
            logger.error("Baseline setup failed. Skipping adjust_position.")
            sys.exit(1)

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration Error for Baseline: {e}")
        if test_baseline:
            test_baseline.metrics['action_taken'] = test_baseline.ACTION_STATES["SETUP_FAILED"]
            test_baseline.metrics['error_message'] = str(e)
            test_baseline.save_metrics()
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during baseline main execution:")
        if test_baseline:
            test_baseline.metrics['action_taken'] = test_baseline.ACTION_STATES["UNEXPECTED_ERROR"]
            test_baseline.metrics['error_message'] = (test_baseline.metrics.get('error_message', "") + f";MainException: {str(e)}").strip(";")
            test_baseline.save_metrics()
        sys.exit(1)
    finally:
        logger.info("=" * 50)
        logger.info("Baseline test run finished.")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()
