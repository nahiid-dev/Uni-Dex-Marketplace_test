#!/bin/bash

# ==============================================
# Enhanced DEX Fork Test Automation Script
# Version: 2.0
# Author: Your Name
# Date: $(date +%Y-%m-%d)
# ==============================================

# --- Configuration Section ---
PROJECT_DIR="/root/Uniswap-Decentralized-Marketplace/Phase3_Smart_Contract"
LOG_FILE="$PROJECT_DIR/fork_test_run_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES=3
RETRY_DELAY=10

# Script Paths
DEPLOY_SCRIPT_PREDICTIVE="$PROJECT_DIR/scripts/deployPredictiveManager.js"
DEPLOY_SCRIPT_BASELINE="$PROJECT_DIR/scripts/deployMinimal.js"

# Test Script Paths
PYTHON_SCRIPT_PREDICTIVE="$PROJECT_DIR/test/predictive/predictive_test.py"
PYTHON_SCRIPT_BASELINE="$PROJECT_DIR/test/baseline/baseline_test.py"

# Address Files
ADDRESS_FILE_PREDICTIVE="$PROJECT_DIR/predictiveManager_address.json"
ADDRESS_FILE_BASELINE="$PROJECT_DIR/baselineMinimal_address.json"

# Utility Scripts
FUNDING_SCRIPT="$PROJECT_DIR/test/utils/fund_my_wallet.py"
ENV_FILE="$PROJECT_DIR/.env"

# Network Configuration
LOCAL_RPC_URL="http://127.0.0.1:8545"
HARDHAT_PORT=8545
HARDHAT_HOST="0.0.0.0"
CHAIN_ID=31337

# Python Environment
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"

# --- Helper Functions ---
function log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

function validate_environment() {
    # Check required commands exist
    local required_commands=(node npm python3 curl jq)
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR: Required command '$cmd' not found"
            exit 1
        fi
    done

    # Check project directory structure
    local required_dirs=(contracts scripts test)
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$PROJECT_DIR/$dir" ]; then
            log "ERROR: Directory $dir not found in project"
            exit 1
        fi
    done
}

function kill_hardhat_node() {
    log "Stopping any existing Hardhat node..."
    pkill -f "hardhat node" || true
    sleep 2
    if pgrep -f "hardhat node" > /dev/null; then
        log "Force killing Hardhat node..."
        pkill -9 -f "hardhat node"
    fi
}

function check_rpc_ready() {
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s -X POST --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
           -H "Content-Type: application/json" "$LOCAL_RPC_URL" &> /dev/null; then
            return 0
        fi
        sleep $RETRY_DELAY
        ((retries++))
    done
    return 1
}

function deploy_contract() {
    local script_path=$1
    local address_file=$2
    local contract_name=$3
    
    log "Deploying $contract_name contract..."
    for attempt in $(seq 1 $MAX_RETRIES); do
        if npx hardhat run "$script_path" --network localhost >> "$LOG_FILE" 2>&1; then
            if [ -f "$address_file" ]; then
                local contract_address=$(jq -r '.address' "$address_file")
                log "$contract_name deployed successfully at $contract_address"
                return 0
            else
                log "ERROR: Address file not created for $contract_name"
            fi
        else
            log "Attempt $attempt failed to deploy $contract_name"
            sleep $RETRY_DELAY
        fi
    done
    
    log "ERROR: Failed to deploy $contract_name after $MAX_RETRIES attempts"
    return 1
}

function run_python_test() {
    local script_path=$1
    local test_name=$2
    
    log "Running $test_name test script..."
    if python3 "$script_path" >> "$LOG_FILE" 2>&1; then
        log "$test_name test completed successfully"
        return 0
    else
        log "ERROR: $test_name test failed"
        return 1
    fi
}

# --- Main Execution ---
{
    log "=============================================="
    log "Starting DEX Fork Test Automation"
    log "=============================================="
    log "Project Directory: $PROJECT_DIR"
    log "Log File: $LOG_FILE"
    
    # Validate environment
    validate_environment
    
    # Change to project directory
    cd "$PROJECT_DIR" || { log "ERROR: Failed to change to project directory"; exit 1; }
    
    # Activate Python virtual environment
    if [ -f "$VENV_ACTIVATE" ]; then
        log "Activating Python virtual environment..."
        source "$VENV_ACTIVATE"
    fi
    
    # Cleanup previous run
    kill_hardhat_node
    rm -f "$ADDRESS_FILE_PREDICTIVE" "$ADDRESS_FILE_BASELINE"
    
    # Load environment variables
    if [ -f "$ENV_FILE" ]; then
        log "Loading environment variables..."
        set -o allexport
        source "$ENV_FILE"
        set +o allexport
        
        # Validate required env vars
        required_vars=(MAINNET_RPC_URL PRIVATE_KEY DEPLOYER_ADDRESS)
        for var in "${required_vars[@]}"; do
            if [ -z "${!var}" ]; then
                log "ERROR: Required environment variable $var not set"
                exit 1
            fi
        done
    else
        log "ERROR: .env file not found"
        exit 1
    fi
    
    # Start Hardhat node
    log "Starting Hardhat node with mainnet fork..."
    nohup npx hardhat node --hostname "$HARDHAT_HOST" --port "$HARDHAT_PORT" \
        --fork "$MAINNET_RPC_URL" > hardhat_node.log 2>&1 &
    HARDHAT_PID=$!
    
    # Wait for node to be ready
    if check_rpc_ready; then
        log "Hardhat node (PID: $HARDHAT_PID) is ready at $LOCAL_RPC_URL"
    else
        log "ERROR: Hardhat node failed to start"
        kill_hardhat_node
        exit 1
    fi
    
    # Fund deployer account
    log "Funding deployer account ($DEPLOYER_ADDRESS)..."
    if python3 "$FUNDING_SCRIPT" >> "$LOG_FILE" 2>&1; then
        log "Deployer account funded successfully"
    else
        log "ERROR: Failed to fund deployer account"
        kill_hardhat_node
        exit 1
    fi
    
    # Deploy contracts
    deploy_contract "$DEPLOY_SCRIPT_PREDICTIVE" "$ADDRESS_FILE_PREDICTIVE" "PredictiveManager" || { kill_hardhat_node; exit 1; }
    deploy_contract "$DEPLOY_SCRIPT_BASELINE" "$ADDRESS_FILE_BASELINE" "BaselineMinimal" || { kill_hardhat_node; exit 1; }
    
    # Run tests
    export MAINNET_FORK_RPC_URL="$LOCAL_RPC_URL"
    run_python_test "$PYTHON_SCRIPT_PREDICTIVE" "Predictive"
    local predictive_result=$?
    run_python_test "$PYTHON_SCRIPT_BASELINE" "Baseline"
    local baseline_result=$?
    
    # Cleanup
    kill_hardhat_node
    if type deactivate &> /dev/null; then
        log "Deactivating Python virtual environment..."
        deactivate
    fi
    
    log "=============================================="
    log "Test Automation Completed"
    log "Predictive Test Result: $( [ $predictive_result -eq 0 ] && echo "SUCCESS" || echo "FAILURE" )"
    log "Baseline Test Result: $( [ $baseline_result -eq 0 ] && echo "SUCCESS" || echo "FAILURE" )"
    log "=============================================="
    
    # Final exit code
    exit $(( predictive_result + baseline_result ))
} | tee -a "$LOG_FILE"