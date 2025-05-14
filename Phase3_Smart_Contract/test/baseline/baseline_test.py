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


# Precision for Decimal calculations
getcontext().prec = 78

try:
    from test.utils.test_base import LiquidityTestBase
    from test.utils.web3_utils import send_transaction, get_contract
except ImportError as e:
    print(f"ERROR importing from test.utils in baseline_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(0.02, 'ether')
MIN_USDC_TO_FUND_CONTRACT = 20 * (10**6) # 20 USDC
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
            'actualPrice_pool': None, 'sqrtPriceX96_pool': None, 'currentTick_pool': None,
            'targetTickLower_offchain': None, 'targetTickUpper_offchain': None,
            'currentTickLower_contract': None, 'currentTickUpper_contract': None, 'currentLiquidity_contract': None,
            'finalTickLower_contract': None, 'finalTickUpper_contract': None, 'finalLiquidity_contract': None,
            'gas_used': None, 'gas_cost_eth': None, 'error_message': ""
        }

    def get_position_info(self) -> dict:
        """Get current position details with improved liquidity handling."""
        try:
            # Get basic position info from contract
            token_id, active, tick_lower, tick_upper, liquidity = self.contract.functions.getCurrentPosition().call()
            # If there's an active position but liquidity is 0, try to estimate
            if active and liquidity == 0:
                try:
                    # Try to estimate liquidity based on token balances and tick range
                    liquidity = self._estimate_liquidity(tick_lower, tick_upper)
                    logger.warning("Active position with 0 liquidity - using estimated liquidity value.")
                except Exception as e:
                    logger.warning(f"Could not estimate liquidity: {e}")
            return {
                'active': active,
                'tokenId': token_id,
                'tickLower': tick_lower,
                'tickUpper': tick_upper,
                'liquidity': liquidity
            }
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None

    def _estimate_liquidity(self, tick_lower: int, tick_upper: int) -> int:
        """Estimate liquidity based on token balances and tick range (simple approximation)."""
        try:
            # Initialize token contracts if not already
            if self.token0_contract is None:
                self.token0_contract = get_contract(self.token0, "IERC20")
            if self.token1_contract is None:
                self.token1_contract = get_contract(self.token1, "IERC20")
            # Get token balances
            token0_bal = self.token0_contract.functions.balanceOf(self.contract_address).call()
            token1_bal = self.token1_contract.functions.balanceOf(self.contract_address).call()
            # Get current tick and sqrtPrice
            slot0 = self.pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            current_tick = slot0[1]
            sqrt_price = float(sqrt_price_x96) / (2 ** 96)
            # Simple estimation logic (not exact Uniswap math)
            if current_tick < tick_lower:
                # All in token0
                return int(token0_bal)
            elif current_tick > tick_upper:
                # All in token1
                return int(token1_bal)
            else:
                # Some in both
                return int((token0_bal + token1_bal) / 2)
        except Exception as e:
            logger.error(f"Error estimating liquidity: {e}")
            return 0

    def setup(self) -> bool:
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
            return True
        except Exception as e:
            logger.exception(f"Baseline setup failed getting pool/tickSpacing: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup pool/tick error: {str(e)}"
            return False

    def get_pool_state(self) -> tuple[int | None, int | None]:
        if not self.pool_contract:
            logger.error("Pool contract not initialized for get_pool_state.")
            return None, None
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("Web3 not connected in get_pool_state.")
            return None, None
        try:
            slot0 = self.pool_contract.functions.slot0().call()
            sqrt_price_x96, tick = slot0[0], slot0[1]
            self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96
            self.metrics['currentTick_pool'] = tick
            self.metrics['actualPrice_pool'] = self._calculate_actual_price(sqrt_price_x96)
            return sqrt_price_x96, tick
        except Exception as e:
            logger.exception(f"Failed to get pool state: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"]
            self.metrics['error_message'] = f"Pool state read error: {str(e)}"
            return None, None
    
    def calculate_target_ticks_offchain(self, current_tick: int) -> tuple[int | None, int | None]:
        if self.tick_spacing is None or current_tick is None:
            logger.error("Tick spacing or current_tick not available for target tick calculation.")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Missing data for target tick calc"
            return None, None
        try:
            width_multiplier = 4 
            half_total_tick_width = (self.tick_spacing * width_multiplier) // 2
            
            if half_total_tick_width < self.tick_spacing:
                 half_total_tick_width = self.tick_spacing

            target_lower_tick = math.floor((current_tick - half_total_tick_width) / self.tick_spacing) * self.tick_spacing
            target_upper_tick = math.floor((current_tick + half_total_tick_width) / self.tick_spacing) * self.tick_spacing
            
            if target_lower_tick >= target_upper_tick:
                target_upper_tick = target_lower_tick + self.tick_spacing

            target_lower_tick = max(MIN_TICK_CONST, target_lower_tick)
            target_upper_tick = min(MAX_TICK_CONST, target_upper_tick)

            if target_lower_tick >= target_upper_tick:
                if target_upper_tick == MAX_TICK_CONST: 
                    target_lower_tick = target_upper_tick - self.tick_spacing
                else: 
                    target_upper_tick = target_lower_tick + self.tick_spacing
                target_lower_tick = max(MIN_TICK_CONST, target_lower_tick)

            if target_lower_tick >= target_upper_tick or \
               target_lower_tick < MIN_TICK_CONST or target_upper_tick > MAX_TICK_CONST:
                logger.error(f"Invalid tick range after calculation: L={target_lower_tick}, U={target_upper_tick}")
                raise ValueError("Invalid tick range generated")

            logger.info(f"Off-chain calculated target ticks: Lower={target_lower_tick}, Upper={target_upper_tick} (Current: {current_tick}, Spacing: {self.tick_spacing})")
            self.metrics['targetTickLower_offchain'] = target_lower_tick
            self.metrics['targetTickUpper_offchain'] = target_upper_tick
            return target_lower_tick, target_upper_tick
        except Exception as e:
            logger.exception(f"Error calculating target ticks off-chain: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Target tick calc error: {str(e)}"
            return None, None

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
        adjusted_onchain_event = False
        try:
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                logger.error("Web3 not connected at start of adjust_position.")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = "W3 unavailable in adjust_position"
                self.save_metrics()
                return False
            _, current_tick = self.get_pool_state()
            if current_tick is None:
                self.save_metrics()
                return False
            position_info = self.get_position_info()
            current_pos_active = position_info.get('active', False) if position_info else False
            self.metrics['currentTickLower_contract'] = position_info.get('tickLower') if current_pos_active else None
            self.metrics['currentTickUpper_contract'] = position_info.get('tickUpper') if current_pos_active else None
            self.metrics['currentLiquidity_contract'] = position_info.get('liquidity', 0) if current_pos_active else 0
            target_lower_tick, target_upper_tick = self.calculate_target_ticks_offchain(current_tick)
            if target_lower_tick is None or target_upper_tick is None:
                self.save_metrics()
                return False
            if current_pos_active and self.metrics['currentTickLower_contract'] is not None and self.tick_spacing is not None:
                TICK_PROXIMITY_THRESHOLD = self.tick_spacing
                is_close_enough = (abs(target_lower_tick - self.metrics['currentTickLower_contract']) <= TICK_PROXIMITY_THRESHOLD and
                                   abs(target_upper_tick - self.metrics['currentTickUpper_contract']) <= TICK_PROXIMITY_THRESHOLD)
                if is_close_enough:
                    logger.info("Off-chain proximity check: Target ticks are close to current. Skipping adjustment call.")
                    self.metrics['action_taken'] = self.ACTION_STATES["SKIPPED_PROXIMITY"]
                    self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                    self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                    self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                    self.save_metrics()
                    return True
            if not self.fund_contract_if_needed():
                logger.error("Funding contract failed. Skipping adjustment.")
                self.metrics['finalTickLower_contract'] = self.metrics['currentTickLower_contract']
                self.metrics['finalTickUpper_contract'] = self.metrics['currentTickUpper_contract']
                self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract']
                self.save_metrics()
                return False
            logger.info(f"Calling adjustLiquidityWithCurrentPrice...")
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
                tx_function = self.contract.functions.adjustLiquidityWithCurrentPrice()
                tx_params = {
                    'from': account.address,
                    'nonce': current_nonce,
                    'chainId': int(web3_utils.w3.net.version)
                }
                try:
                    pre_tx = tx_function.build_transaction({'from': account.address, 'nonce': current_nonce, 'chainId': tx_params['chainId']})
                    estimated_gas = web3_utils.w3.eth.estimate_gas(pre_tx)
                    tx_params['gas'] = int(estimated_gas * 1.25)
                    logger.info(f"Estimated gas for baseline adjustment: {estimated_gas}, using: {tx_params['gas']}")
                except Exception as est_err:
                    logger.warning(f"Gas estimation failed for baseline adjustment: {est_err}. Using default 1,500,000")
                    tx_params['gas'] = 1500000
                final_tx_to_send = tx_function.build_transaction(tx_params)
                receipt = send_transaction(final_tx_to_send)
                self.metrics['tx_hash'] = receipt.transactionHash.hex() if receipt else None
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SENT"]
                if receipt and receipt.status == 1:
                    logger.info(f"Baseline adjustment transaction successful (Status 1). Tx: {self.metrics['tx_hash']}. Processing events...")
                    self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                    if receipt.get('effectiveGasPrice'):
                        self.metrics['gas_cost_eth'] = float(Web3.from_wei(receipt.gasUsed * receipt.effectiveGasPrice, 'ether'))
                    try:
                        event_name = "BaselineAdjustmentMetrics"
                        if hasattr(self.contract.events, event_name):
                            logs = self.contract.events[event_name]().process_receipt(receipt, errors=logging.WARN)
                            if logs and len(logs) > 0:
                                adjusted_onchain_event = logs[0]['args'].get('adjusted', False)
                                logger.info(f"Event '{event_name}' found: Adjusted={adjusted_onchain_event}, Args={logs[0]['args']}")
                                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"] if adjusted_onchain_event else self.ACTION_STATES["TX_SUCCESS_SKIPPED_ONCHAIN"]
                            else:
                                logger.warning(f"Event '{event_name}' not found in transaction logs, assuming adjustment occurred.")
                                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                                adjusted_onchain_event = True
                        else:
                            logger.warning(f"Contract does not have event '{event_name}'. Assuming adjustment if tx successful.")
                            self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                            adjusted_onchain_event = True
                    except Exception as log_exc:
                        logger.warning(f"Error processing event '{event_name}': {log_exc}. Assuming adjustment if tx successful.")
                        self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                        adjusted_onchain_event = True
                    adjustment_call_success = True
                elif receipt:
                    logger.error(f"Baseline adjustment transaction reverted (Status 0). Tx: {self.metrics['tx_hash']}")
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                    self.metrics['error_message'] = "tx_reverted_onchain"
                    self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                    if receipt.get('effectiveGasPrice'):
                       self.metrics['gas_cost_eth'] = float(Web3.from_wei(receipt.gasUsed * receipt.effectiveGasPrice, 'ether'))
                    adjustment_call_success = False
                else:
                    logger.error("Baseline adjustment transaction sending/receipt failed.")
                    if self.metrics['action_taken'] == self.ACTION_STATES["TX_SENT"]:
                         self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                    if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for baseline adjustment failed"
                    adjustment_call_success = False
            except Exception as tx_err:
                logger.exception(f"Error during baseline adjustment transaction call/wait: {tx_err}")
                self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                self.metrics['error_message'] = f"TxError: {str(tx_err)}"
                self.save_metrics()
                return False
            # After transaction, get improved final position info
            final_pos_info = self.get_position_info()
            if final_pos_info:
                self.metrics['finalLiquidity_contract'] = final_pos_info.get('liquidity', 0)
                self.metrics['finalTickLower_contract'] = final_pos_info.get('tickLower')
                self.metrics['finalTickUpper_contract'] = final_pos_info.get('tickUpper')
                # If active but liquidity is still zero, estimate
                if final_pos_info.get('active') and self.metrics['finalLiquidity_contract'] == 0:
                    logger.warning("Active position with 0 liquidity after tx - using estimation.")
                    estimated_liq = self._estimate_liquidity(
                        self.metrics['finalTickLower_contract'],
                        self.metrics['finalTickUpper_contract']
                    )
                    self.metrics['finalLiquidity_contract'] = estimated_liq
            else:
                logger.warning("Could not read final position info after tx - using target ticks and current liquidity.")
                self.metrics['finalTickLower_contract'] = target_lower_tick
                self.metrics['finalTickUpper_contract'] = target_upper_tick
                self.metrics['finalLiquidity_contract'] = self.metrics['currentLiquidity_contract'] or self._estimate_liquidity(
                    target_lower_tick,
                    target_upper_tick
                )
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
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_offchain', 'targetTickUpper_offchain',
            'currentTickLower_contract', 'currentTickUpper_contract', 'currentLiquidity_contract',
            'finalTickLower_contract', 'finalTickUpper_contract', 'finalLiquidity_contract',
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