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
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import test.utils.web3_utils as web3_utils
import test.utils.contract_funder as contract_funder

# Precision for Decimal calculations
getcontext().prec = 78 # دقت محاسبات Decimal

try:
    from test.utils.test_base import LiquidityTestBase
    from test.utils.web3_utils import send_transaction, get_contract # اطمینان از وارد کردن get_contract
except ImportError as e:
    print(f"ERROR importing from test.utils in predictive_test.py: {e}. Check sys.path and __init__.py files.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
# مقادیر مورد نیاز برای شارژ قرارداد، از مقادیر جدید استفاده شده
MIN_WETH_TO_FUND_CONTRACT = Web3.to_wei(1, 'ether') # مطابق نسخه جدید فایل شما
MIN_USDC_TO_FUND_CONTRACT = 1000 * (10**6)  # 1000 USDC مطابق نسخه جدید فایل شما
TWO_POW_96 = Decimal(2**96)
MIN_TICK_CONST = -887272
MAX_TICK_CONST = 887272

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
RESULTS_FILE = project_root / 'position_results_predictive.csv' # همانطور که در نسخه جدید است
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
        # token0_decimals and token1_decimals will be set in super().setup()

    def _reset_metrics(self): # مطابق نسخه جدید فایل شما
        return {
            'timestamp': None, 'contract_type': 'Predictive',
            'action_taken': self.ACTION_STATES["INIT"], 'tx_hash': None,
            'range_width_multiplier_setting': None, # اضافه شده در نسخه جدید
            'predictedPrice_api': None, 'predictedTick_calculated': None,
            'external_api_eth_price': None, # اضافه شده در نسخه جدید
            'actualPrice_pool': None, 'sqrtPriceX96_pool': 0, 'currentTick_pool': 0,
            'targetTickLower_calculated': 0, 'targetTickUpper_calculated': 0,
            'initial_contract_balance_token0': None, # اضافه شده در نسخه جدید
            'initial_contract_balance_token1': None, # اضافه شده در نسخه جدید
            'finalTickLower_contract': 0, 'finalTickUpper_contract': 0, 'liquidity_contract': 0,
            'amount0_provided_to_mint': None, # اضافه شده در نسخه جدید
            'amount1_provided_to_mint': None, # اضافه شده در نسخه جدید
            'fees_collected_token0': None, # اضافه شده در نسخه جدید
            'fees_collected_token1': None, # اضافه شده در نسخه جدید
            'gas_used': 0, 'gas_cost_eth': 0.0, 'error_message': ""
        }

    def setup(self, desired_range_width_multiplier: int) -> bool: # پارامتر اضافه شده مطابق نسخه جدید
        if not super().setup(desired_range_width_multiplier): # Pass the multiplier to parent class setup
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
            # get_contract باید از web3_utils وارد شود
            factory_contract = get_contract(factory_address, "IUniswapV3Factory")
            if not factory_contract: return False # بررسی اینکه get_contract موفق بوده

            fee = self.contract.functions.fee().call()
            self.pool_address = factory_contract.functions.getPool(self.token0, self.token1, fee).call()
            
            if not self.pool_address or self.pool_address == '0x' + '0' * 40:
                logger.error(f"Predictive pool address not found for {self.token0}/{self.token1} fee {fee}")
                self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                self.metrics['error_message'] = f"Pool not found {self.token0}/{self.token1} fee {fee}"
                return False
                
            self.pool_contract = get_contract(self.pool_address, "IUniswapV3Pool")
            if not self.pool_contract: return False # بررسی اینکه get_contract موفق بوده

            logger.info(f"Predictive Pool contract initialized at {self.pool_address}")

            # تنظیم rangeWidthMultiplier مطابق نسخه جدید
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
            # افزودن تخمین گاز
            try:
                gas_estimate = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier).estimate_gas({'from': tx_account.address})
                tx_params['gas'] = int(gas_estimate * 1.2)
            except Exception as e:
                logger.warning(f"Gas estimation failed for setRangeWidthMultiplier: {e}. Using default 200000.")
                tx_params['gas'] = 200000


            tx_set_rwm_build = self.contract.functions.setRangeWidthMultiplier(desired_range_width_multiplier)
            # ارسال تراکنش با استفاده از web3_utils.send_transaction که کلید خصوصی را می‌پذیرد
            receipt_rwm = web3_utils.send_transaction(tx_set_rwm_build.build_transaction(tx_params), private_key)
            
            if not receipt_rwm or receipt_rwm.status == 0:
                tx_hash_rwm = receipt_rwm.transactionHash.hex() if receipt_rwm else 'N/A'
                logger.error(f"Failed to set rangeWidthMultiplier for Predictive contract. TxHash: {tx_hash_rwm}")
                self.metrics['error_message'] += f";Failed to set RWM (tx: {tx_hash_rwm})"
                return False
            logger.info(f"rangeWidthMultiplier set successfully for Predictive contract. TxHash: {receipt_rwm.transactionHash.hex()}")
            return True
        except Exception as e:
            logger.exception(f"Predictive setup failed: {e}") # تغییر پیام لاگ
            self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
            self.metrics['error_message'] = f"Setup error: {str(e)}" # تغییر پیام لاگ
            return False

    # ---- متدهای بازگردانده شده از نسخه قدیمی ----
    def get_predicted_price_from_api(self) -> float | None: # مطابق نسخه قدیمی [92-97]
        try:
            logger.info(f"Querying LSTM API at {LSTM_API_URL}...")
            response = requests.get(LSTM_API_URL, timeout=15) # افزایش timeout
            response.raise_for_status() # بررسی خطاهای HTTP
            data = response.json()
            
            # تلاش برای خواندن کلیدهای مختلف قیمت از پاسخ API
            predicted_price_str = data.get('predicted_price') or data.get('price') or data.get('prediction')

            if predicted_price_str is None:
                logger.error(f"Predicted price key not found in API response. Data: {data}")
                raise ValueError("Predicted price not in API response")

            if isinstance(predicted_price_str, str):
                predicted_price_str = predicted_price_str.replace("USD", "").strip() # حذف "USD" اگر وجود دارد

            predicted_price = float(predicted_price_str)
            logger.info(f"Received predicted ETH price from API: {predicted_price:.4f} USD") # افزایش دقت نمایش
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
        except (ValueError, KeyError) as e: # ValueError برای float()، KeyError برای .get()
            logger.exception(f"Error processing API response from {LSTM_API_URL}. Data: {data if 'data' in locals() else 'N/A'}. Error: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["API_FAILED"]
            self.metrics['error_message'] = f"API Response Processing Error: {str(e)}"
            return None

    def calculate_tick_from_price(self, price: float) -> int | None: # مطابق نسخه قدیمی [98-103]
        if self.token0_decimals is None or self.token1_decimals is None:
            logger.error("Token decimals not available for tick calculation. Ensure LiquidityTestBase.setup() was successful.")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = "Token decimals missing for tick calc"
            return None
        try:
            price_decimal = Decimal(str(price))
            # فرمول محاسبه تیک از قیمت برای Uniswap V3
            # tick = floor(log_sqrt(1.0001)(sqrt(price_token1/token0 * 10^(decimals_token0 - decimals_token1))))
            # ما قیمت ETH به USD را داریم (token1/token0). اگر token0=USDC و token1=WETH باشد:
            # price_decimal همان قیمت WETH/USDC است.
            effective_sqrt_price_arg = price_decimal * (Decimal(10)**(self.token0_decimals - self.token1_decimals))
            
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

            # tick = floor(ln(effective_sqrt_price) / ln(sqrt(1.0001)))
            tick = math.floor(math.log(float(effective_sqrt_price)) / math.log(math.sqrt(1.0001)))
            tick = max(MIN_TICK_CONST, min(MAX_TICK_CONST, tick)) # اطمینان از اینکه تیک در محدوده مجاز است
            
            logger.info(f"Calculated tick {tick} from price {price:.4f} (dec0: {self.token0_decimals}, dec1: {self.token1_decimals})")
            self.metrics['predictedTick_calculated'] = tick
            return tick
        except Exception as e:
            logger.exception(f"Failed to calculate predicted tick from price {price}: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["CALCULATION_FAILED"]
            self.metrics['error_message'] = f"Tick Calculation Error: {str(e)}"
            return None

    def update_pool_and_position_metrics(self, final_update=False): # مطابق نسخه قدیمی [104-107]
        try:
            if self.pool_contract:
                slot0 = self.pool_contract.functions.slot0().call()
                sqrt_price_x96_pool, current_tick_pool = slot0[0], slot0[1]
                self.metrics['sqrtPriceX96_pool'] = sqrt_price_x96_pool
                self.metrics['currentTick_pool'] = current_tick_pool
                # _calculate_actual_price باید در LiquidityTestBase تعریف شده باشد
                self.metrics['actualPrice_pool'] = self._calculate_actual_price(sqrt_price_x96_pool) 
            else:
                logger.warning("Pool contract not available for metrics update (pool_and_position).")
                # self.metrics['action_taken'] = self.ACTION_STATES["POOL_READ_FAILED"] # ممکن است این حالت را نخواهیم خطا در نظر بگیریم
                # self.metrics['error_message'] += ";Pool contract missing for metrics"


            # get_position_info باید در LiquidityTestBase تعریف شده باشد
            position_info = self.get_position_info() 
            if position_info:
                # در نسخه جدید، فیلدهای initial_contract_balance و ... اضافه شده‌اند
                # فیلدهای finalTickLower_contract و ... باید اینجا به‌روز شوند
                # اگر پوزیشنی وجود داشته باشد
                if final_update or position_info.get('liquidity', 0) > 0 : # فقط اگر پوزیشن معتبر است
                    self.metrics['finalTickLower_contract'] = position_info.get('tickLower', 0)
                    self.metrics['finalTickUpper_contract'] = position_info.get('tickUpper', 0)
                    self.metrics['liquidity_contract'] = position_info.get('liquidity', 0)
            else:
                logger.warning("Could not get position info from contract for metrics update.")
                # اگر پوزیشنی وجود ندارد، مقادیر صفر باقی می‌مانند (که در _reset_metrics تنظیم شده)

        except Exception as e:
            logger.exception(f"Error updating pool/position metrics: {e}")
            self.metrics['action_taken'] = self.ACTION_STATES["METRICS_UPDATE_FAILED"]
            if not self.metrics.get('error_message'): self.metrics['error_message'] = f"Metrics Update Error: {str(e)}"
    # ---- پایان متدهای بازگردانده شده ----

    def adjust_position(self, target_weth_balance: float, target_usdc_balance: float) -> bool: # پارامترها مطابق نسخه جدید
        self.metrics = self._reset_metrics() # ریست کردن متریک‌ها در ابتدای هر فراخوانی
        try:
            # خواندن range_width_multiplier فعلی از قرارداد برای لاگ کردن
            current_rwm = self.contract.functions.rangeWidthMultiplier().call()
            self.metrics['range_width_multiplier_setting'] = current_rwm
        except Exception:
            logger.warning("Could not read current rangeWidthMultiplier from predictive contract for metrics.")
            # اگر در setup مقداردهی شده باشد، همان استفاده می‌شود.

        adjustment_call_success = False
        private_key_env = os.getenv('PRIVATE_KEY')
        if not private_key_env:
            logger.error("PRIVATE_KEY not found for adjust_position.")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = "PRIVATE_KEY missing for adjustment tx"
            self.save_metrics()
            return False
        
        funding_account = Account.from_key(private_key_env) # نام متغیر به funding_account تغییر یافت

        try:
            # 1. دریافت قیمت پیش‌بینی شده
            predicted_price = self.get_predicted_price_from_api() # فراخوانی متد بازگردانده شده
            if predicted_price is None:
                # self.metrics قبلاً در get_predicted_price_from_api به‌روز شده
                self.save_metrics()
                return False

            # 2. محاسبه تیک از قیمت پیش‌بینی شده
            predicted_tick = self.calculate_tick_from_price(predicted_price) # فراخوانی متد بازگردانده شده
            if predicted_tick is None:
                # self.metrics قبلاً در calculate_tick_from_price به‌روز شده
                self.save_metrics()
                return False
            
            # 3. به‌روزرسانی اطلاعات اولیه استخر (قبل از تراکنش)
            self.update_pool_and_position_metrics(final_update=False) # فراخوانی متد بازگردانده شده

            # 4. شارژ دقیق قرارداد (مطابق نسخه جدید)
            logger.info("Ensuring precise token balances for Predictive contract...")
            if not contract_funder.ensure_precise_token_balances(
                contract_address=self.contract_address,
                token0_address=self.token0,
                token0_decimals=self.token0_decimals,
                target_token0_amount_readable=target_usdc_balance, # USDC معمولا token0 است
                token1_address=self.token1,
                token1_decimals=self.token1_decimals,
                target_token1_amount_readable=target_weth_balance, # WETH معمولا token1 است
                funding_account_private_key=private_key_env
            ):
                logger.error("Precise funding for Predictive contract failed.")
                self.metrics['action_taken'] = self.ACTION_STATES["FUNDING_FAILED"]
                self.metrics['error_message'] = "Precise contract funding failed"
                self.save_metrics()
                return False

            # 5. خواندن موجودی اولیه قرارداد (بعد از شارژ و قبل از تراکنش اصلی) - مطابق نسخه جدید
            try:
                token0_contract = get_contract(self.token0, "IERC20")
                token1_contract = get_contract(self.token1, "IERC20")
                if token0_contract and token1_contract:
                    self.metrics['initial_contract_balance_token0'] = token0_contract.functions.balanceOf(self.contract_address).call()
                    self.metrics['initial_contract_balance_token1'] = token1_contract.functions.balanceOf(self.contract_address).call()
                else:
                    logger.warning("Could not get token contracts to read initial balances.")
            except Exception as bal_err:
                logger.warning(f"Could not read initial contract balances for metrics: {bal_err}")

            # 6. فراخوانی تابع updatePredictionAndAdjust در قرارداد هوشمند
            logger.info(f"Calling updatePredictionAndAdjust with predictedTick: {predicted_tick}")
            
            tx_function_call = self.contract.functions.updatePredictionAndAdjust(predicted_tick)
            tx_parameters = {
                'from': funding_account.address,
                'nonce': web3_utils.w3.eth.get_transaction_count(funding_account.address),
                'chainId': int(web3_utils.w3.net.version)
            }
            
            try: # تخمین گاز
                gas_estimate = tx_function_call.estimate_gas({'from': funding_account.address})
                tx_parameters['gas'] = int(gas_estimate * 1.25) # افزایش ضریب بافر
                logger.info(f"Estimated gas for 'updatePredictionAndAdjust': {gas_estimate}, using: {tx_parameters['gas']}")
            except Exception as est_err:
                logger.warning(f"Gas estimation failed for 'updatePredictionAndAdjust': {est_err}. Using default 1,500,000")
                tx_parameters['gas'] = 1500000

            built_transaction = tx_function_call.build_transaction(tx_parameters)
            receipt = web3_utils.send_transaction(built_transaction, private_key_env) # ارسال کلید خصوصی

            self.metrics['tx_hash'] = receipt.transactionHash.hex() if receipt else None
            self.metrics['action_taken'] = self.ACTION_STATES["TX_SENT"]

            if receipt and receipt.status == 1:
                logger.info(f"Adjustment transaction successful (Status 1). Tx: {self.metrics['tx_hash']}")
                self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                effective_gas_price = receipt.get('effectiveGasPrice', web3_utils.w3.eth.gas_price) # در صورت نبود، از gas_price فعلی استفاده کن
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * effective_gas_price, 'ether'))
                self.metrics['action_taken'] = self.ACTION_STATES["TX_SUCCESS_ADJUSTED"]
                adjustment_call_success = True

                # پردازش رویدادها برای استخراج مقادیر (مطابق نسخه جدید)
                try:
                    # پردازش رویداد LiquidityOperation
                    mint_logs = self.contract.events.LiquidityOperation().process_receipt(receipt, errors=logging.WARN)
                    for log_entry in mint_logs: # تغییر نام متغیر برای جلوگیری از تداخل با ماژول logging
                        if log_entry.args.operationType == "MINT": # مطمئن شوید operationType در رویداد شما وجود دارد
                            self.metrics['amount0_provided_to_mint'] = log_entry.args.amount0Actual # یا amount0
                            self.metrics['amount1_provided_to_mint'] = log_entry.args.amount1Actual # یا amount1
                        elif log_entry.args.operationType == "REMOVE": # یا هر نام دیگری برای نوع عملیات
                            self.metrics['fees_collected_token0'] = log_entry.args.amount0Collected # یا amount0
                            self.metrics['fees_collected_token1'] = log_entry.args.amount1Collected # یا amount1
                    
                    # پردازش رویداد PredictionAdjustmentMetrics (یا نام مشابه در قرارداد شما)
                    # اگر این رویداد شامل finalTickLower/Upper است
                    adj_metrics_logs = self.contract.events.PredictionAdjustmentMetrics().process_receipt(receipt, errors=logging.WARN)
                    if adj_metrics_logs:
                        # این بخش به جای خواندن مستقیم از get_position_info برای تیک‌های نهایی استفاده می‌شود
                        self.metrics['finalTickLower_contract'] = adj_metrics_logs[0].args.finalTickLower
                        self.metrics['finalTickUpper_contract'] = adj_metrics_logs[0].args.finalTickUpper
                        # self.metrics['liquidity_contract'] = adj_metrics_logs[0].args.finalLiquidity # اگر در رویداد باشد
                except Exception as log_processing_error:
                    logger.error(f"Error processing logs for predictive transaction: {log_processing_error}")

            elif receipt: # تراکنش ناموفق بود (status 0)
                logger.error(f"Adjustment transaction reverted (Status 0). Tx: {self.metrics['tx_hash']}")
                self.metrics['action_taken'] = self.ACTION_STATES["TX_REVERTED"]
                self.metrics['error_message'] = "tx_reverted_onchain"
                self.metrics['gas_used'] = receipt.get('gasUsed', 0)
                effective_gas_price = receipt.get('effectiveGasPrice', web3_utils.w3.eth.gas_price)
                self.metrics['gas_cost_eth'] = float(Web3.from_wei(self.metrics['gas_used'] * effective_gas_price, 'ether'))
                adjustment_call_success = False
            else: # ارسال تراکنش به طور کلی ناموفق بود (رسید دریافت نشد)
                logger.error("Adjustment transaction sending/receipt failed.")
                if self.metrics['action_taken'] == self.ACTION_STATES["TX_SENT"]: # اگر قبلاً به TX_SENT رفته بود
                    self.metrics['action_taken'] = self.ACTION_STATES["TX_WAIT_FAILED"]
                if not self.metrics['error_message']: self.metrics['error_message'] = "send_transaction for adjustment failed"
                adjustment_call_success = False

        except Exception as tx_err:
            logger.exception(f"Error during adjustment transaction call/wait: {tx_err}")
            self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
            self.metrics['error_message'] = f"TxError: {str(tx_err)}"
            # self.save_metrics() # در finally ذخیره می‌شود
            return False # در اینجا return می‌کنیم چون یک خطای غیرمنتظره در ارسال تراکنش رخ داده

        finally: # اطمینان از اینکه متریک‌ها همیشه ذخیره می‌شوند
            # 7. به‌روزرسانی نهایی اطلاعات استخر و پوزیشن (بعد از تراکنش)
            # اگر رویدادها شامل تیک‌های نهایی و نقدینگی نبودند، این بخش آنها را پر می‌کند
            if not self.metrics['finalTickLower_contract'] and not self.metrics['finalTickUpper_contract']:
                 self.update_pool_and_position_metrics(final_update=True) # فراخوانی متد بازگردانده شده
            self.save_metrics()
        
        return adjustment_call_success
    
    def save_metrics(self): # مطابق نسخه جدید فایل شما
        self.metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # ستون‌ها مطابق با _reset_metrics در نسخه جدید
        columns = [
            'timestamp', 'contract_type', 'action_taken', 'tx_hash',
            'range_width_multiplier_setting',
            'predictedPrice_api', 'predictedTick_calculated',
            'external_api_eth_price', # اطمینان از اینکه این مقدار جایی پر می‌شود
            'actualPrice_pool', 'sqrtPriceX96_pool', 'currentTick_pool',
            'targetTickLower_calculated', 'targetTickUpper_calculated', # اینها باید توسط قرارداد محاسبه و از رویداد خوانده شوند یا جداگانه محاسبه شوند
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
            # اطمینان از اینکه همه ستون‌ها در self.metrics وجود دارند یا مقدار پیش‌فرض می‌گیرند
            row_data = {col: self.metrics.get(col) for col in columns}

            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore') # extrasaction='ignore' برای نادیده گرفتن کلیدهای اضافی در row_data
                if not file_exists or os.path.getsize(RESULTS_FILE) == 0: # بررسی خالی بودن فایل
                    writer.writeheader()
                writer.writerow(row_data)
            logger.info(f"Predictive metrics saved to {RESULTS_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save predictive metrics: {e}")

# --- Main Function ---
def main(): # مطابق نسخه جدید فایل شما
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
            # در صورت عدم وجود فایل، یک ردیف خطا در CSV ذخیره کن
            temp_test_for_error_log = PredictiveTest(contract_address="0x0") # آدرس موقت
            temp_test_for_error_log.metrics['action_taken'] = temp_test_for_error_log.ACTION_STATES["SETUP_FAILED"]
            temp_test_for_error_log.metrics['error_message'] = f"Address file not found: {ADDRESS_FILE_PREDICTIVE}"
            temp_test_for_error_log.save_metrics()
            raise FileNotFoundError(f"File not found: {ADDRESS_FILE_PREDICTIVE}")

        logger.info(f"Reading predictive address from: {ADDRESS_FILE_PREDICTIVE}")
        with open(ADDRESS_FILE_PREDICTIVE, 'r') as f:
            content = f.read()
            logger.debug(f"Predictive address file content: {content}")
            addresses_data = json.loads(content)
            predictive_address = addresses_data.get('address')
            if not predictive_address:
                logger.error(f"Key 'address' not found in {ADDRESS_FILE_PREDICTIVE}")
                # ذخیره خطا در CSV
                temp_test_for_error_log = PredictiveTest(contract_address="0x0")
                temp_test_for_error_log.metrics['action_taken'] = temp_test_for_error_log.ACTION_STATES["SETUP_FAILED"]
                temp_test_for_error_log.metrics['error_message'] = f"Key 'address' not found in {ADDRESS_FILE_PREDICTIVE}"
                temp_test_for_error_log.save_metrics()
                raise ValueError(f"Key 'address' not found in {ADDRESS_FILE_PREDICTIVE}")
        logger.info(f"Loaded Predictive Manager Address: {predictive_address}")

        test = PredictiveTest(predictive_address)
        # execute_test_steps باید پارامترهای لازم برای setup و adjust_position را بپذیرد
        # این مقادیر باید از یک منبع خارجی (مانند یک فایل کانفیگ یا آرگومان‌های خط فرمان) بیایند
        # برای مثال:
        desired_rwm = int(os.getenv('PREDICTIVE_RWM', '50')) # خواندن از متغیر محیطی یا مقدار پیش‌فرض
        target_weth = float(os.getenv('PREDICTIVE_TARGET_WETH', '1.0'))
        target_usdc = float(os.getenv('PREDICTIVE_TARGET_USDC', '2000.0'))

        test.execute_test_steps(
            desired_range_width_multiplier=desired_rwm,
            target_weth_balance=target_weth,
            target_usdc_balance=target_usdc
        )


    except FileNotFoundError as e:
        logger.error(f"Setup Error - Address file not found: {e}")
        # خطا قبلاً در CSV ذخیره شده است
    except ValueError as e:
        logger.error(f"Configuration Error - Problem reading address file or address key missing: {e}")
        # خطا قبلاً در CSV ذخیره شده است
    except Exception as e:
        logger.exception(f"An unexpected error occurred during predictive main execution:")
        # ذخیره خطای عمومی در CSV
        # این بخش ممکن است نیاز به تعریف یک نمونه موقت از PredictiveTest داشته باشد اگر test تعریف نشده باشد
        if 'test' not in locals() and predictive_address: # اگر test تعریف نشده ولی آدرس داریم
            test = PredictiveTest(predictive_address) # یا یک آدرس موقت
        elif 'test' not in locals():
            test = PredictiveTest("0x0")

        test.metrics['action_taken'] = test.ACTION_STATES["UNEXPECTED_ERROR"]
        test.metrics['error_message'] = str(e)
        test.save_metrics()
    finally:
        logger.info("="*50)
        logger.info("Predictive test run finished.")
        logger.info("="*50)

if __name__ == "__main__":
    main()