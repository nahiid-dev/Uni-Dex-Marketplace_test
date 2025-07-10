#!/bin/bash

# ==============================================================================
#                 Optimized DEX Fork Test Automation Script
#
# Version: 4.3 (Corrected with all helper functions)
# Description:
# Automates end-to-end testing of DEX strategies on a local Hardhat Mainnet fork.
# This version runs each test in a completely isolated fork environment and
# includes all necessary helper functions for execution.
# ==============================================================================

# --- Script Configuration ---
set -e -u -o pipefail

# --- PATH & Directory Setup ---
PROJECT_DIR_DEFAULT="/root/Uni-Dex-Marketplace_test/Phase3_Smart_Contract"
PROJECT_DIR="${FORK_TEST_PROJECT_DIR:-$PROJECT_DIR_DEFAULT}"
LOG_FILE_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_FILE_DIR"
LOG_FILE="$LOG_FILE_DIR/fork_test_run_$(date +%Y%m%d_%H%M%S).log"
HARDHAT_NODE_LOG_FILE_PRED="$LOG_FILE_DIR/hardhat_node_pred_$(date +%Y%m%d_%H%M%S).log"
HARDHAT_NODE_LOG_FILE_BASE="$LOG_FILE_DIR/hardhat_node_base_$(date +%Y%m%d_%H%M%S).log"

# --- Script & File Paths ---
DEPLOY_SCRIPT_PREDICTIVE="$PROJECT_DIR/scripts/deployPredictiveManager.js"
DEPLOY_SCRIPT_BASELINE="$PROJECT_DIR/scripts/deployMinimal.js"
PYTHON_SCRIPT_PREDICTIVE="$PROJECT_DIR/test/predictive/predictive_test.py"
PYTHON_SCRIPT_BASELINE="$PROJECT_DIR/test/baseline/baseline_test.py"
FUNDING_SCRIPT="$PROJECT_DIR/test/utils/fund_my_wallet.py"
ENV_FILE="$PROJECT_DIR/.env"

# --- Address Files (Generated during deployment) ---
ADDRESS_FILE_PREDICTIVE="$PROJECT_DIR/predictiveManager_address.json"
ADDRESS_FILE_BASELINE="$PROJECT_DIR/baselineMinimal_address.json"

# --- Network & Retry Configuration ---
LOCAL_RPC_URL="http://127.0.0.1:8545"
HARDHAT_PORT=8545
HARDHAT_HOST="127.0.0.1"
MAX_RETRIES=3
RETRY_DELAY=10 # Seconds

# --- Python Environment ---
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"

# --- Helper Functions ---

function log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

function validate_environment() {
    log "Validating environment prerequisites..."
    local all_ok=true
    local required_commands=(node npm python3 curl jq git)
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR: Required command '$cmd' not found."
            all_ok=false
        fi
    done

    local required_files=("$ENV_FILE" "$DEPLOY_SCRIPT_PREDICTIVE" "$DEPLOY_SCRIPT_BASELINE" \
                          "$PYTHON_SCRIPT_PREDICTIVE" "$PYTHON_SCRIPT_BASELINE" "$FUNDING_SCRIPT")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log "ERROR: Required file '$file' not found."
            all_ok=false
        fi
    done
    
    if [ -f "$ENV_FILE" ]; then
        local temp_env_vars
        temp_env_vars=$(grep -v '^#' "$ENV_FILE" | grep -v '^$')
        local required_env_vars=("MAINNET_RPC_URL" "PRIVATE_KEY" "DEPLOYER_ADDRESS")
        for var_name in "${required_env_vars[@]}"; do
            if ! echo "$temp_env_vars" | grep -q "^${var_name}="; then
                 log "ERROR: Required environment variable '$var_name' is not set in $ENV_FILE."
                 all_ok=false
            fi
        done
    fi

    if [ "$all_ok" = false ]; then
        log "CRITICAL: Environment validation failed. Please fix the errors above."
        exit 1
    fi
    log "Environment validation successful."
}

function kill_hardhat_node() {
    log "Attempting to stop any existing Hardhat node on port $HARDHAT_PORT..."
    PIDS=$(pgrep -f "hardhat node.*--port $HARDHAT_PORT" || true)
    if [ -n "$PIDS" ]; then
        kill $PIDS &>/dev/null || true
        sleep 2
        if pgrep -f "hardhat node.*--port $HARDHAT_PORT" &>/dev/null; then
            log "Node still running. Force killing (kill -9)..."
            kill -9 $PIDS &>/dev/null || true
        fi
    fi
    log "Hardhat node stopped or was not running."
}

function check_rpc_ready() {
    log "Checking if RPC at $LOCAL_RPC_URL is ready..."
    for attempt in $(seq 1 20); do
        if curl -s -X POST --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
           -H "Content-Type: application/json" "$LOCAL_RPC_URL" --connect-timeout 5 --max-time 10 2>/dev/null | jq -e '.result' > /dev/null; then
            log "RPC is ready."
            return 0
        fi
        log "RPC not ready yet (attempt $attempt/20). Retrying in 5 seconds..."
        sleep 5
    done
    log "ERROR: RPC did not become ready."
    return 1
}

function deploy_contract() {
    local script_path="$1"
    local address_file="$2"
    local contract_name="$3"

    log "Deploying $contract_name..."
    rm -f "$address_file"

    for attempt in $(seq 1 $MAX_RETRIES); do
        log "Attempt $attempt of $MAX_RETRIES to deploy $contract_name..."
        if npx hardhat run "$script_path" --network localhost >> "$LOG_FILE" 2>&1; then
            if [ -f "$address_file" ] && [[ "$(jq -r '.address' "$address_file" 2>/dev/null)" =~ ^0x[a-fA-F0-9]{40}$ ]]; then
                log "✅ $contract_name deployed successfully!"
                return 0
            else
                log "ERROR: Deployment script ran, but address file '$address_file' is missing or its content is invalid."
            fi
        else
            log "ERROR: Hardhat deployment script failed for $contract_name on attempt $attempt."
        fi

        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done

    log "FATAL: Failed to deploy $contract_name after $MAX_RETRIES attempts."
    return 1
}

# <<< START: MISSING FUNCTION ADDED HERE >>>
function run_python_test() {
    local script_path="$1"
    local test_name="$2"
    log "Running Python test: $test_name..."
    if python3 -u "$script_path" >> "$LOG_FILE" 2>&1; then
        log "✅ $test_name test completed successfully."
        return 0
    else
        log "❌ ERROR: $test_name test failed. Check $LOG_FILE for traceback."
        return 1
    fi
}
# <<< END: MISSING FUNCTION ADDED HERE >>>


# --- MAIN SCRIPT EXECUTION ---
exec > >(tee -a "$LOG_FILE") 2>&1

log "=============================================="
log "🚀 Starting DEX Fork Test Automation Script 🚀"
log "=============================================="

# --- 1. GLOBAL SETUP ---
log "--- [1/5] Performing Global Setup ---"
validate_environment
cd "$PROJECT_DIR" || exit 1
log "Changed directory to $(pwd)"

log "Cleaning and compiling contracts..."
npx hardhat clean
npx hardhat compile

if [ -f "$VENV_ACTIVATE" ]; then
    log "Activating Python virtual environment..."
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
    log "Python virtual environment activated: $(command -v python3)"
else
    log "WARNING: Python venv not found. Using system Python."
fi

log "Loading environment variables from $ENV_FILE..."
set -o allexport
# shellcheck source=/dev/null
source "$ENV_FILE"
set +o allexport
log "Environment variables loaded."

# --- 2. PREPARE FORK CONFIG ---
log "--- [2/5] Preparing Fork Configuration ---"
log "Fetching latest block number from Mainnet..."
LATEST_BLOCK_HEX=$(curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' "$MAINNET_RPC_URL" | jq -r '.result')
if [[ "$LATEST_BLOCK_HEX" =~ ^0x[a-fA-F0-9]+$ ]]; then
    LATEST_BLOCK_DEC=$((LATEST_BLOCK_HEX))
    log "Forking from block number: $LATEST_BLOCK_DEC"
else
    log "CRITICAL: Failed to fetch valid block number. Response: $LATEST_BLOCK_HEX"
    exit 1
fi
export NODE_OPTIONS=--max-old-space-size=4096

# Initialize test status variables
predictive_test_status=1
baseline_test_status=1

# --- 3. RUN PREDICTIVE TEST ---
log ""
log "====================================================="
log "--- [3/5] RUNNING PREDICTIVE STRATEGY TEST ---"
log "====================================================="
kill_hardhat_node
log "Starting Hardhat fork for Predictive test..."
nohup npx hardhat node --hostname "$HARDHAT_HOST" --port "$HARDHAT_PORT" \
    --fork "$MAINNET_RPC_URL" --fork-block-number "$LATEST_BLOCK_DEC" > "$HARDHAT_NODE_LOG_FILE_PRED" 2>&1 &
HARDHAT_PID=$!
log "Hardhat node started with PID: $HARDHAT_PID. Waiting for RPC..."
if ! check_rpc_ready; then
    log "CRITICAL: Hardhat node failed for Predictive test. Check $HARDHAT_NODE_LOG_FILE_PRED."
    exit 1
fi
export MAINNET_FORK_RPC_URL="$LOCAL_RPC_URL"

log "Funding wallet for Predictive test..."
if ! python3 -u "$FUNDING_SCRIPT"; then
    log "CRITICAL: Wallet funding failed for Predictive test."
    kill "$HARDHAT_PID"
    exit 1
fi

if ! deploy_contract "$DEPLOY_SCRIPT_PREDICTIVE" "$ADDRESS_FILE_PREDICTIVE" "PredictiveManager"; then kill "$HARDHAT_PID"; exit 1; fi

log "Exporting settings for Predictive test..."
export PREDICTIVE_MANAGER_ADDRESS=$(jq -r '.address' "$ADDRESS_FILE_PREDICTIVE")
# === SETTINGS FOR PREDICTIVE TEST ===
export PREDICTIVE_TARGET_WETH="50.0"
export PREDICTIVE_TARGET_USDC="2000000.0"
export PREDICTIVE_NUM_SWAPS="20"
export PREDICTIVE_SWAP_AMOUNT_ETH="11.5"
export PREDICTIVE_SWAP_AMOUNT_USDC="32200" # 11.5 WETH * 2800 $/WETH
# ====================================

log "Running Python test script for Predictive Strategy..."
run_python_test "$PYTHON_SCRIPT_PREDICTIVE" "Predictive Strategy"
predictive_test_status=$?

log "Stopping Hardhat fork after Predictive test..."
kill "$HARDHAT_PID"

# --- 4. RUN BASELINE TEST ---
log ""
log "====================================================="
log "--- [4/5] RUNNING BASELINE STRATEGY TEST ---"
log "====================================================="
kill_hardhat_node
log "Starting Hardhat fork for Baseline test..."
nohup npx hardhat node --hostname "$HARDHAT_HOST" --port "$HARDHAT_PORT" \
    --fork "$MAINNET_RPC_URL" --fork-block-number "$LATEST_BLOCK_DEC" > "$HARDHAT_NODE_LOG_FILE_BASE" 2>&1 &
HARDHAT_PID=$!
log "Hardhat node started with PID: $HARDHAT_PID. Waiting for RPC..."
if ! check_rpc_ready; then
    log "CRITICAL: Hardhat node failed for Baseline test. Check $HARDHAT_NODE_LOG_FILE_BASE."
    exit 1
fi
export MAINNET_FORK_RPC_URL="$LOCAL_RPC_URL"

log "Funding wallet for Baseline test..."
if ! python3 -u "$FUNDING_SCRIPT"; then
    log "CRITICAL: Wallet funding failed for Baseline test."
    kill "$HARDHAT_PID"
    exit 1
fi

if ! deploy_contract "$DEPLOY_SCRIPT_BASELINE" "$ADDRESS_FILE_BASELINE" "BaselineMinimal"; then kill "$HARDHAT_PID"; exit 1; fi

log "Exporting settings for Baseline test..."
export BASELINE_MINIMAL_ADDRESS=$(jq -r '.address' "$ADDRESS_FILE_BASELINE")
# === SETTINGS FOR BASELINE TEST ===
export BASELINE_TARGET_WETH="50.0"
export BASELINE_TARGET_USDC="2000000.0"
export BASELINE_RWM="20"
# NOTE: Edit baseline_test.py to use these variables for swap count and amount.
# ==================================

log "Running Python test script for Baseline Strategy..."
run_python_test "$PYTHON_SCRIPT_BASELINE" "Baseline Strategy"
baseline_test_status=$?

log "Stopping Hardhat fork after Baseline test..."
kill "$HARDHAT_PID"

# --- 5. TEARDOWN & SUMMARY ---
log ""
log "=============================================="
log "--- [5/5] Performing Teardown & Summary ---"
log "=============================================="
kill_hardhat_node # Final cleanup check

if type deactivate &> /dev/null && [[ -n "${VIRTUAL_ENV-}" ]]; then
    log "Deactivating Python virtual environment..."
    deactivate
fi

log ""
log "=============================================="
log "📊 Test Automation Script Completed 📊"
log "=============================================="
log "Predictive Test Result: $( [ $predictive_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )"
log "Baseline Test Result:   $( [ $baseline_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )"
log "Detailed logs available in: $LOG_FILE"
log "Hardhat node logs are in separate files in: $LOG_FILE_DIR"

if [ $predictive_test_status -ne 0 ] || [ $baseline_test_status -ne 0 ]; then
    exit 1
fi
exit 0