import os
import logging
from abc import ABC, abstractmethod
from web3 import Web3
from decimal import Decimal
# Import the web3_utils module itself to access its w3 instance and functions
import test.utils.web3_utils as web3_utils

logger = logging.getLogger('test_base')

class LiquidityTestBase(ABC):
    """Base class for liquidity position testing."""

    def __init__(self, contract_address: str, contract_name: str):
        """Initialize test with contract info."""
        if not contract_address:
            raise ValueError("Contract address cannot be empty")
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract_name = contract_name
        self.contract = None
        self.token0 = None
        self.token1 = None
        self.token0_decimals = None
        self.token1_decimals = None
        # self.w3 will now be web3_utils.w3

    def setup(self, desired_range_width_multiplier: int) -> bool:
        """Initialize connection and contracts."""
        try:
            # Ensure Web3 is initialized using the function from web3_utils
            if not web3_utils.init_web3():
                logger.error("Web3 initialization failed in LiquidityTestBase setup")
                return False
            
            # Ensure the w3 instance in web3_utils is usable
            if not web3_utils.w3 or not web3_utils.w3.is_connected():
                 logger.error("web3_utils.w3 is not available or not connected after init_web3 call.")
                 return False


            logger.info(f"Setting up test for {self.contract_name} at {self.contract_address}")
            # Load main contract using get_contract from web3_utils (which uses web3_utils.w3)
            self.contract = web3_utils.get_contract(self.contract_address, self.contract_name)
            if not self.contract:
                logger.error(f"Failed to load contract {self.contract_name}")
                return False

            self.token0 = Web3.to_checksum_address(self.contract.functions.token0().call())
            self.token1 = Web3.to_checksum_address(self.contract.functions.token1().call())

            # Get decimals
            token0_contract = web3_utils.get_contract(self.token0, "IERC20")
            self.token0_decimals = token0_contract.functions.decimals().call()

            token1_contract = web3_utils.get_contract(self.token1, "IERC20")
            self.token1_decimals = token1_contract.functions.decimals().call()

            logger.info(f"Token0: {self.token0} (Decimals: {self.token0_decimals})")
            logger.info(f"Token1: {self.token1} (Decimals: {self.token1_decimals})")
            
            logger.info(f"Setup completed for {self.contract_name}")
            return True

        except Exception as e:
            logger.exception(f"Setup failed for {self.contract_name} at {self.contract_address}: {e}")
            return False

    def check_balances(self) -> bool:
        """Step 2: Check contract's token balances."""
        if not self.contract or not self.token0 or not self.token1:
            logger.error("Contract or tokens not initialized. Run setup first.")
            return False
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("Web3 not connected in check_balances.")
            return False
        try:
            token0_contract = web3_utils.get_contract(self.token0, "IERC20")
            token1_contract = web3_utils.get_contract(self.token1, "IERC20")

            balance0_wei = token0_contract.functions.balanceOf(self.contract_address).call()
            balance1_wei = token1_contract.functions.balanceOf(self.contract_address).call()

            readable_balance0 = Decimal(balance0_wei) / (10 ** self.token0_decimals)
            readable_balance1 = Decimal(balance1_wei) / (10 ** self.token1_decimals)

            logger.info(f"Contract Token0 ({self.token0[-6:]}) balance: {readable_balance0:.6f}")
            logger.info(f"Contract Token1 ({self.token1[-6:]}) balance: {readable_balance1:.6f}")
            return True
        except Exception as e:
            logger.exception(f"Balance check failed: {e}")
            return False

    @abstractmethod
    def adjust_position(self) -> bool:
        """Abstract method for adjusting the position. Implement in derived class."""
        pass

    @abstractmethod
    def save_metrics(self):
        """Abstract method for saving metrics. Implement in derived class."""
        pass

    def execute_test_steps(self, desired_range_width_multiplier: int = 50, target_weth_balance: float = 1.0, target_usdc_balance: float = 1000.0) -> bool:
        """Execute all test steps sequentially with required arguments."""
        try:
            logger.info("--- Test Step 1: Setup ---")
            if not self.setup(desired_range_width_multiplier):
                logger.error("Setup failed. Aborting test.")
                if hasattr(self, 'metrics') and self.metrics and hasattr(self, 'ACTION_STATES') and "SETUP_FAILED" in self.ACTION_STATES:
                    self.metrics['action_taken'] = self.ACTION_STATES["SETUP_FAILED"]
                    self.metrics['error_message'] = self.metrics.get('error_message', "Base setup failed in execute_test_steps")
                    self.save_metrics()
                return False

            logger.info("--- Test Step 2: Balance Check ---")
            self.check_balances()

            logger.info("--- Test Step 3: Position Adjustment ---")
            if not self.adjust_position(target_weth_balance, target_usdc_balance):
                logger.error("Position adjustment failed.")
                return False

            logger.info("--- All test steps completed successfully ---")
            return True
        except Exception as e:
            logger.exception(f"Test execution failed during steps: {e}")
            try:
                if hasattr(self, 'metrics') and self.metrics:
                    if hasattr(self, 'ACTION_STATES') and "UNEXPECTED_ERROR" in self.ACTION_STATES:
                        self.metrics['action_taken'] = self.ACTION_STATES["UNEXPECTED_ERROR"]
                    self.metrics['error_message'] = self.metrics.get('error_message', f"Test Execution Aborted: {str(e)}")
                    self.save_metrics()
            except Exception as save_err:
                logger.error(f"Also failed to save metrics during exception handling: {save_err}")
            return False

    def get_position_info(self) -> dict | None:
        """Get current position info from the contract."""
        if not self.contract:
            logger.error("Contract not initialized. Run setup first.")
            return None
        if not web3_utils.w3 or not web3_utils.w3.is_connected():
            logger.error("Web3 not connected in get_position_info.")
            return None
        try:
            pos_data = None
            if hasattr(self.contract.functions, 'getPositionState'):
                 pass 
            
            if hasattr(self.contract.functions, 'getCurrentPosition'):
                pos_data = self.contract.functions.getCurrentPosition().call()
            elif hasattr(self.contract.functions, 'currentPosition'):
                pos_data = self.contract.functions.currentPosition().call()
            else:
                logger.error(f"No known position info method (getCurrentPosition, currentPosition) found on contract {self.contract_name}")
                return None

            if pos_data and len(pos_data) == 5:
                 position = {
                    'tokenId': pos_data[0],
                    'liquidity': pos_data[1],
                    'tickLower': pos_data[2],
                    'tickUpper': pos_data[3],
                    'active': pos_data[4]
                }
                 logger.debug(f"Fetched Position Info: {position}")
                 return position
            else:
                logger.error(f"Position data format unexpected or not found. Data: {pos_data}")
                return None

        except Exception as e:
            logger.exception(f"Failed to get position info from contract {self.contract_name}: {e}")
            return None

    def _calculate_actual_price(self, sqrt_price_x96: int) -> float:
        """
        Calculates the human-readable price from a sqrtPriceX96 value.
        Assumes T1/T0 price (e.g., WETH/USDC if Token1 is WETH and Token0 is USDC).
        """
        if not sqrt_price_x96 or sqrt_price_x96 == 0:
            return 0.0
        if self.token0_decimals is None or self.token1_decimals is None:
            logger.error("Token decimals not set, cannot calculate actual price.")
            return 0.0

        TWO_POW_96 = Decimal(2**96)
        try:
            sqrt_price_x96_dec = Decimal(sqrt_price_x96)
            price_ratio_token1_token0 = (sqrt_price_x96_dec / TWO_POW_96)**2
            adjusted_price = price_ratio_token1_token0 * (Decimal(10)**(self.token1_decimals - self.token0_decimals))
            return float(adjusted_price)
        except Exception as e:
            logger.exception(f"Error calculating actual price from sqrtPriceX96={sqrt_price_x96}: {e}")
            return 0.0