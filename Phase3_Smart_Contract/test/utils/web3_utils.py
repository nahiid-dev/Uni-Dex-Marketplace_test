import os
import json
import logging
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
from pathlib import Path
import time

# --- Logging Setup ---
logger = logging.getLogger('web3_utils')
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
utils_dir = Path(__file__).resolve().parent
PROJECT_ROOT = utils_dir.parent.parent  # Should resolve to Phase3_Smart_Contract
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    # logger.debug(f"Loaded .env file from: {env_path}")
else:
    logger.warning(f".env file not found at expected location: {env_path}")

PRIVATE_KEY = os.getenv('PRIVATE_KEY')
RPC_URL = os.getenv('MAINNET_FORK_RPC_URL', 'http://127.0.0.1:8545')

# --- Web3 Initialization ---
w3 = None

def init_web3(retries=3, delay=2):
    """Initialize Web3 connection and validate environment."""
    global w3
    if w3 and w3.is_connected():
        # logger.debug("Web3 already initialized and connected.")
        return True

    if not RPC_URL:
        logger.error("RPC_URL (for fork) not found or set.")
        return False
    # PRIVATE_KEY check moved to functions that require it for sending transactions

    for attempt in range(retries):
        try:
            logger.info(f"Attempting to connect to Web3 provider at {RPC_URL} (Attempt {attempt + 1}/{retries})...")
            w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={'timeout': 60}))
            if w3.is_connected():
                chain_id = w3.net.version
                logger.info(f"Successfully connected to network via {RPC_URL} - Chain ID: {chain_id}")
                return True
            else:
                logger.warning(f"Connection attempt {attempt + 1} failed (is_connected() is false).")
        except Exception as e:
            logger.error(f"Error connecting to Web3 provider on attempt {attempt + 1}: {e}")

        if attempt < retries - 1:
            logger.info(f"Retrying connection after {delay} seconds...")
            time.sleep(delay)

    logger.critical("Failed to connect to Web3 provider after multiple retries.")
    w3 = None # Ensure w3 is None if all retries fail
    return False

# --- Standard IERC20 ABI ---
IERC20_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function", "stateMutability": "view"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function", "stateMutability": "view"},
    {"constant": False, "inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "type": "function", "stateMutability": "nonpayable"},
    {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function", "stateMutability": "nonpayable"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}
]

# --- WETH ABI ---
WETH_ABI = IERC20_ABI + [
    {"type": "fallback", "stateMutability": "payable"},
    {"type": "receive", "stateMutability": "payable"}, # For receiving ETH
    {"constant": False, "inputs": [], "name": "deposit", "outputs": [], "type": "function", "stateMutability": "payable"},
    {"constant": False, "inputs": [{"name": "wad", "type": "uint256"}], "name": "withdraw", "outputs": [], "type": "function", "stateMutability": "nonpayable"}
]


# --- Mainnet WETH Address ---
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

def load_contract_abi(contract_name):
    """Load a contract ABI from artifacts directory."""
    if contract_name == "IERC20": return IERC20_ABI
    if contract_name == "WETH": return WETH_ABI

    base_path = PROJECT_ROOT # Assumes artifacts is in PROJECT_ROOT

    possible_paths = [
        base_path / f'artifacts/contracts/{contract_name}.sol/{contract_name}.json',
        base_path / f'artifacts/contracts/interfaces/{contract_name}.sol/{contract_name}.json',
        base_path / f'artifacts/@uniswap/v3-core/contracts/interfaces/{contract_name}.sol/{contract_name}.json',
        base_path / f'artifacts/@uniswap/v3-periphery/contracts/interfaces/{contract_name}.sol/{contract_name}.json',
        # Add other potential paths if your project structure is different
    ]

    for path in possible_paths:
        if path.exists():
            # logger.debug(f"Loading ABI for {contract_name} from: {path}")
            with open(path) as f:
                try:
                    contract_json = json.load(f)
                    if 'abi' not in contract_json:
                        logger.error(f"ABI key not found in {path}")
                        continue
                    return contract_json['abi']
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {path}")
                    continue
    raise FileNotFoundError(f"ABI not found for {contract_name} in any expected path relative to {base_path}. Searched paths: {possible_paths}")

def get_contract(address, contract_name):
    """Get contract instance with loaded ABI."""
    global w3
    if not w3 or not w3.is_connected():
        if not init_web3(): # Attempt to initialize if not already
            raise ConnectionError("Web3 connection failed or could not be established in get_contract.")

    if not address: raise ValueError(f"Address for {contract_name} not provided")

    try:
        checksum_address = Web3.to_checksum_address(address)
        abi = load_contract_abi(contract_name)
        return w3.eth.contract(address=checksum_address, abi=abi)
    except Exception as e:
        logger.exception(f"Error getting contract {contract_name} at {address}: {e}")
        raise

def send_transaction(tx_params_dict): # Renamed to avoid conflict if tx_params is a function argument elsewhere
    """Builds necessary fields, signs, sends a transaction and waits for receipt."""
    global w3
    if not w3 or not w3.is_connected():
        if not init_web3():
            raise ConnectionError("Web3 connection failed or could not be established in send_transaction.")

    if 'from' not in tx_params_dict: raise ValueError("Transaction 'from' address missing")
    if not PRIVATE_KEY:
        logger.error("PRIVATE_KEY not found in environment variables for send_transaction.")
        return None

    try:
        # Ensure Chain ID
        if 'chainId' not in tx_params_dict: tx_params_dict['chainId'] = int(w3.net.version)

        # Ensure Nonce
        if 'nonce' not in tx_params_dict: tx_params_dict['nonce'] = w3.eth.get_transaction_count(tx_params_dict['from'])

        # Gas Estimation (if not provided)
        if 'gas' not in tx_params_dict:
            try:
                tx_params_dict['gas'] = int(w3.eth.estimate_gas(tx_params_dict) * 1.25) # Add 25% buffer
                # logger.debug(f"Estimated gas: {tx_params_dict['gas']}")
            except Exception as gas_err:
                logger.error(f"Gas estimation failed: {gas_err}. Using default 1,000,000.")
                tx_params_dict['gas'] = 1000000

        # Gas Price Strategy (Handle EIP-1559 vs Legacy)
        if 'gasPrice' not in tx_params_dict and 'maxFeePerGas' not in tx_params_dict:
            try:
                fee_history = w3.eth.fee_history(1, 'latest', [10])
                base_fee = fee_history['baseFeePerGas'][-1]
                tip = fee_history['reward'][-1][0] if fee_history['reward'] and fee_history['reward'][-1] else w3.to_wei(1, 'gwei') # Fallback tip
                tx_params_dict['maxPriorityFeePerGas'] = tip
                tx_params_dict['maxFeePerGas'] = base_fee * 2 + tip
                # logger.debug(f"Using EIP-1559 gas: maxFeePerGas={tx_params_dict['maxFeePerGas']}, maxPriorityFeePerGas={tx_params_dict['maxPriorityFeePerGas']}")
            except:
                tx_params_dict['gasPrice'] = int(w3.eth.gas_price * 1.1)
                # logger.debug(f"Using legacy gasPrice: {tx_params_dict['gasPrice']}")
        # elif 'gasPrice' in tx_params_dict:
            # logger.debug(f"Using provided legacy gasPrice: {tx_params_dict['gasPrice']}")
        # elif 'maxFeePerGas' in tx_params_dict:
            # logger.debug(f"Using provided EIP-1559 gas: maxFeePerGas={tx_params_dict['maxFeePerGas']}, maxPriorityFeePerGas={tx_params_dict.get('maxPriorityFeePerGas')}")

        # logger.debug(f"Final TX params before signing: {tx_params_dict}")
        signed_tx = w3.eth.account.sign_transaction(tx_params_dict, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        logger.info(f"Transaction confirmed in block: {receipt.blockNumber}, Status: {receipt.status}")
        return receipt
    except Exception as e:
        logger.exception(f"Transaction processing failed: {e}")
        return None

def wrap_eth_to_weth(amount_wei) -> bool:
    """
    Wrap ETH to WETH by sending ETH to the WETH contract address.
    """
    global w3
    if not w3 or not w3.is_connected():
        if not init_web3():
             raise ConnectionError("Web3 connection failed or could not be established in wrap_eth_to_weth.")

    if not PRIVATE_KEY:
        logger.error("PRIVATE_KEY not found for wrap_eth_to_weth.")
        return False

    try:
        account = Account.from_key(PRIVATE_KEY)
        checksum_weth_address = Web3.to_checksum_address(WETH_ADDRESS)

        logger.info(f"Attempting to wrap {Web3.from_wei(amount_wei, 'ether')} ETH for {account.address} by sending to {checksum_weth_address}")

        tx_dict = {
            'from': account.address,
            'to': checksum_weth_address,
            'value': amount_wei,
            'nonce': w3.eth.get_transaction_count(account.address),
            'chainId': int(w3.net.version),
        }

        try:
            tx_dict['gas'] = int(w3.eth.estimate_gas(tx_dict) * 1.25)
        except Exception as e:
            logger.warning(f"Gas estimation for wrap_eth_to_weth failed: {e}. Using default gas limit 100000.")
            tx_dict['gas'] = 100000 # WETH deposit is usually low gas

        # Set gas price (EIP-1559 preferred)
        try:
            fee_history = w3.eth.fee_history(1, 'latest', [10])
            base_fee = fee_history['baseFeePerGas'][-1]
            tip = fee_history['reward'][-1][0] if fee_history['reward'] and fee_history['reward'][-1] else w3.to_wei(1, 'gwei')
            tx_dict['maxPriorityFeePerGas'] = tip
            tx_dict['maxFeePerGas'] = base_fee * 2 + tip
        except:
            tx_dict['gasPrice'] = int(w3.eth.gas_price * 1.1)

        receipt = send_transaction(tx_dict) # Use the main send_transaction helper

        if receipt and receipt.status == 1:
            logger.info(f"WETH wrap confirmed successfully.")
            return True
        else:
            logger.error(f"WETH wrap transaction failed or receipt not obtained. Receipt: {receipt}")
            return False
    except Exception as e:
        logger.exception(f"Wrapping ETH to WETH failed: {e}")
        return False