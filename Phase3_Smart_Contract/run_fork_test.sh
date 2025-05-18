#!/bin/bash

# ==============================================
# Enhanced DEX Fork Test Automation Script
# Version: 2.1
# Author: Your Name
# Date: $(date +%Y-%m-%d)
# ==============================================

# --- Configuration Section ---
# Ensure this path is correctly set
PROJECT_DIR_DEFAULT="/root/Uni-Dex-Marketplace_test/Phase3_Smart_Contract"
PROJECT_DIR="${FORK_TEST_PROJECT_DIR:-$PROJECT_DIR_DEFAULT}" # اجازه می‌دهد از طریق متغیر محیطی بازنویسی شود

LOG_FILE_DIR="$PROJECT_DIR/logs" # پوشه‌ای مجزا برای لاگ‌ها
mkdir -p "$LOG_FILE_DIR" # ایجاد پوشه لاگ‌ها اگر وجود نداشته باشد
LOG_FILE="$LOG_FILE_DIR/fork_test_run_$(date +%Y%m%d_%H%M%S).log"
HARDHAT_NODE_LOG_FILE="$LOG_FILE_DIR/hardhat_node_$(date +%Y%m%d_%H%M%S).log"

MAX_RETRIES=3
RETRY_DELAY=10 # ثانیه

# Script Paths
DEPLOY_SCRIPT_PREDICTIVE="$PROJECT_DIR/scripts/deployPredictiveManager.js"
DEPLOY_SCRIPT_BASELINE="$PROJECT_DIR/scripts/deployMinimal.js"

# Test Script Paths
PYTHON_SCRIPT_PREDICTIVE="$PROJECT_DIR/test/predictive/predictive_test.py"
PYTHON_SCRIPT_BASELINE="$PROJECT_DIR/test/baseline/baseline_test.py"

# Address Files (مسیر فایل‌های آدرس نسبت به PROJECT_DIR)
ADDRESS_FILE_PREDICTIVE="$PROJECT_DIR/predictiveManager_address.json"
ADDRESS_FILE_BASELINE="$PROJECT_DIR/baselineMinimal_address.json"

# Utility Scripts
FUNDING_SCRIPT="$PROJECT_DIR/test/utils/fund_my_wallet.py" # اسکریپتی که ارسال کردید
ENV_FILE="$PROJECT_DIR/.env"

# Network Configuration
LOCAL_RPC_URL="http://127.0.0.1:8545"
HARDHAT_PORT=8545
HARDHAT_HOST="0.0.0.0" # یا "127.0.0.1" اگر فقط دسترسی محلی مد نظر است
# CHAIN_ID=31337 # معمولاً توسط Hardhat به طور خودکار برای شبکه محلی تنظیم می‌شود

# Python Environment
VENV_PATH="$PROJECT_DIR/venv" # مسیر پوشه محیط مجازی
VENV_ACTIVATE="$VENV_PATH/bin/activate"

# --- Helper Functions ---
function log() {
    # هم به کنسول و هم به فایل اصلی لاگ می‌نویسد
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

function validate_environment() {
    log "Validating environment prerequisites..."
    local all_ok=true
    local required_commands=(node npm python3 curl jq git) # git اضافه شد برای بررسی‌های احتمالی دیگر
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR: Required command '$cmd' not found in PATH."
            all_ok=false
        fi
    done

    if [ ! -f "$ENV_FILE" ]; then
        log "ERROR: Environment file .env not found at $ENV_FILE"
        all_ok=false
    fi
    
    if [ ! -d "$PROJECT_DIR" ]; then
        log "ERROR: Project directory $PROJECT_DIR does not exist."
        all_ok=false
    fi

    # بررسی وجود اسکریپت‌های اصلی
    local required_scripts=("$DEPLOY_SCRIPT_PREDICTIVE" "$DEPLOY_SCRIPT_BASELINE" "$PYTHON_SCRIPT_PREDICTIVE" "$PYTHON_SCRIPT_BASELINE" "$FUNDING_SCRIPT")
    for script_path in "${required_scripts[@]}"; do
        if [ ! -f "$script_path" ]; then
            log "ERROR: Required script '$script_path' not found."
            all_ok=false
        fi
    done
    
    if [ "$all_ok" = false ]; then
        log "Environment validation failed. Please check messages above."
        exit 1
    fi
    log "Environment validation successful."
}

function kill_hardhat_node() {
    log "Attempting to stop any existing Hardhat node..."
    # استفاده از pkill با دقت بیشتر برای جلوگیری از kill کردن فرآیندهای اشتباه
    pgrep -f "hardhat node.*--port $HARDHAT_PORT" | xargs -r kill
    sleep 2 # زمان برای بسته شدن آرام
    if pgrep -f "hardhat node.*--port $HARDHAT_PORT" > /dev/null; then
        log "Hardhat node still running. Force killing..."
        pgrep -f "hardhat node.*--port $HARDHAT_PORT" | xargs -r kill -9
        sleep 1
    fi
    if ! pgrep -f "hardhat node.*--port $HARDHAT_PORT" > /dev/null; then
        log "Hardhat node stopped successfully or was not running."
    else
        log "WARNING: Could not stop existing Hardhat node. Manual check might be needed."
    fi
}

function check_rpc_ready() {
    local url_to_check=$1
    local max_check_retries=20 # افزایش تعداد تلاش‌ها برای آمادگی RPC
    local check_delay=5    # افزایش زمان تاخیر بین بررسی‌ها
    log "Checking if RPC at $url_to_check is ready..."
    for attempt in $(seq 1 $max_check_retries); do
        # استفاده از یک درخواست ساده net_version
        if curl -s -X POST --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
           -H "Content-Type: application/json" "$url_to_check" 2>/dev/null | jq -e '.result' > /dev/null; then
            log "RPC at $url_to_check is ready."
            return 0 # موفقیت
        fi
        log "RPC not ready yet (attempt $attempt/$max_check_retries). Retrying in $check_delay seconds..."
        sleep $check_delay
    done
    log "ERROR: RPC at $url_to_check did not become ready after $max_check_retries attempts."
    return 1 # شکست
}

function deploy_contract() {
    local script_path="$1"
    local address_file="$2"
    local contract_name="$3"
    
    log "Deploying $contract_name contract using $script_path..."
    # حذف فایل آدرس قدیمی اگر وجود دارد
    rm -f "$address_file"

    for attempt in $(seq 1 $MAX_RETRIES); do
        log "Attempt $attempt of $MAX_RETRIES to deploy $contract_name..."
        # اجرای دستور deploy و هدایت stdout و stderr به فایل لاگ اصلی
        if npx hardhat run "$script_path" --network localhost >> "$LOG_FILE" 2>&1; then
            if [ -f "$address_file" ] && jq -e '.address' "$address_file" > /dev/null 2>&1; then
                local contract_address
                contract_address=$(jq -r '.address' "$address_file")
                if [[ "$contract_address" != "null" && -n "$contract_address" ]]; then
                    log "$contract_name deployed successfully! Address: $contract_address (saved to $address_file)"
                    return 0 # موفقیت
                else
                    log "ERROR: Deployment script for $contract_name ran, but address in $address_file is invalid or null."
                fi
            else
                log "ERROR: Deployment script for $contract_name ran, but address file $address_file was not created or is invalid."
            fi
        else
            log "ERROR: Hardhat deployment script $script_path failed for $contract_name on attempt $attempt."
        fi
        
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log "Retrying deployment of $contract_name in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    log "FATAL: Failed to deploy $contract_name after $MAX_RETRIES attempts. Check logs for details."
    return 1 # شکست
}

function run_python_test() {
    local script_path="$1"
    local test_name="$2"
    
    log "Running Python test script: $test_name ($script_path)..."
    # اجرای اسکریپت پایتون و هدایت خروجی آن
    if python3 "$script_path" >> "$LOG_FILE" 2>&1; then
        log "✅ $test_name test completed successfully."
        return 0 # موفقیت
    else
        log "❌ ERROR: $test_name test failed. Check $LOG_FILE for details."
        return 1 # شکست
    fi
}

# --- Main Execution ---
# تمام خروجی این بلاک به فایل لاگ اصلی هدایت می‌شود و همچنین در کنسول نمایش داده می‌شود
{
    log "=============================================="
    log "🚀 Starting DEX Fork Test Automation Script 🚀"
    log "=============================================="
    log "Project Directory: $PROJECT_DIR"
    log "Main Log File: $LOG_FILE"
    log "Hardhat Node Log File: $HARDHAT_NODE_LOG_FILE"
    
    validate_environment # ابتدا محیط را بررسی کن
    
    cd "$PROJECT_DIR" || { log "CRITICAL: Failed to change to project directory $PROJECT_DIR. Exiting."; exit 1; }
    log "Changed directory to $PROJECT_DIR"
    
    if [ -f "$VENV_ACTIVATE" ]; then
        log "Activating Python virtual environment from $VENV_ACTIVATE..."
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
        log "Python virtual environment activated. Current python: $(command -v python3)"
    else
        log "WARNING: Python virtual environment activation script not found at $VENV_ACTIVATE. Using system Python."
    fi
    
    kill_hardhat_node # پاکسازی اجرای قبلی
    # حذف فایل‌های آدرس قدیمی برای اطمینان از ایجاد مجدد
    log "Removing old address files (if any)..."
    rm -f "$ADDRESS_FILE_PREDICTIVE" "$ADDRESS_FILE_BASELINE"
    
    log "Loading environment variables from $ENV_FILE..."
    if [ -f "$ENV_FILE" ]; then
        set -o allexport # تمام متغیرهای تعریف شده را export کن
        # shellcheck source=/dev/null
        source "$ENV_FILE"
        set +o allexport
        log "Environment variables loaded."

        # بررسی متغیرهای محیطی ضروری
        local env_check_ok=true
        local required_env_vars=("MAINNET_RPC_URL" "PRIVATE_KEY" "DEPLOYER_ADDRESS" "LSTM_API_URL")
        for var_name in "${required_env_vars[@]}"; do
            if [ -z "${!var_name}" ]; then # بررسی اینکه متغیر خالی نباشد
                log "ERROR: Required environment variable '$var_name' is not set in $ENV_FILE."
                env_check_ok=false
            fi
        done
        if [ "$env_check_ok" = false ]; then
            log "CRITICAL: Missing required environment variables. Exiting."
            exit 1
        fi
        log "Required environment variables seem to be set."
    else
        log "CRITICAL: .env file not found at $ENV_FILE. Exiting."
        exit 1
    fi
    
    log "Starting Hardhat node with Mainnet fork from $MAINNET_RPC_URL..."
    # اجرای Hardhat node در پس‌زمینه و ذخیره لاگ آن در فایل مجزا
    nohup npx hardhat node --hostname "$HARDHAT_HOST" --port "$HARDHAT_PORT" \
        --fork "$MAINNET_RPC_URL" > "$HARDHAT_NODE_LOG_FILE" 2>&1 &
    HARDHAT_PID=$!
    log "Hardhat node started with PID: $HARDHAT_PID. Waiting for it to be ready..."
    
    if ! check_rpc_ready "$LOCAL_RPC_URL"; then
        log "CRITICAL: Hardhat node failed to start or become ready. Check $HARDHAT_NODE_LOG_FILE. Exiting."
        kill_hardhat_node # سعی در kill کردن دوباره
        exit 1
    fi
    log "Hardhat node is ready and listening on $LOCAL_RPC_URL"
    
    # صادرات متغیر محیطی برای استفاده در اسکریپت‌های پایتون
    export MAINNET_FORK_RPC_URL="$LOCAL_RPC_URL" 
    log "Exported MAINNET_FORK_RPC_URL as $LOCAL_RPC_URL for Python scripts."

    log "Funding deployer account ($DEPLOYER_ADDRESS)..."
    # اطمینان از اینکه DEPLOYER_ADDRESS برای اسکریپت fund_my_wallet.py در دسترس است
    # این متغیر باید از .env خوانده شده باشد
    if python3 "$FUNDING_SCRIPT"; then # خروجی این اسکریپت به طور خودکار به LOG_FILE می‌رود
        log "Deployer account funding script executed. Check its specific logs if issues persist."
    else
        log "CRITICAL: Failed to execute funding script for deployer account. Check $LOG_FILE. Exiting."
        kill_hardhat_node
        exit 1
    fi
    
    # استقرار قراردادها
    # برای هر قرارداد، یک بلوک if جداگانه برای خروج در صورت شکست
    if ! deploy_contract "$DEPLOY_SCRIPT_PREDICTIVE" "$ADDRESS_FILE_PREDICTIVE" "PredictiveManager"; then
        log "CRITICAL: Deployment of PredictiveManager failed. Exiting."
        kill_hardhat_node
        exit 1
    fi
    
    if ! deploy_contract "$DEPLOY_SCRIPT_BASELINE" "$ADDRESS_FILE_BASELINE" "BaselineMinimal"; then
        log "CRITICAL: Deployment of BaselineMinimal failed. Exiting."
        kill_hardhat_node
        exit 1
    fi
    
    # اجرای تست‌های پایتون
    log "--- Starting Python Tests ---"    predictive_test_status=1 # Default to failure
    baseline_test_status=1   # Default to failure

    run_python_test "$PYTHON_SCRIPT_PREDICTIVE" "Predictive Strategy"
    predictive_test_status=$?
    
    run_python_test "$PYTHON_SCRIPT_BASELINE" "Baseline Strategy"
    baseline_test_status=$?
    
    log "--- Python Tests Finished ---"

    # Final cleanup
    log "Cleaning up: Stopping Hardhat node..."
    kill_hardhat_node
    
    if type deactivate &> /dev/null && [[ -n "$VIRTUAL_ENV" ]]; then # Only if we're in a virtual environment
        log "Deactivating Python virtual environment..."
        deactivate
    fi
    
    log "=============================================="
    log "🏁 Test Automation Script Completed 🏁"
    log "=============================================="
    log "Predictive Test Result: $( [ $predictive_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )"
    log "Baseline Test Result:   $( [ $baseline_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )"
    log "Detailed logs available in: $LOG_FILE"
    log "Hardhat node logs (if any issues): $HARDHAT_NODE_LOG_FILE"
    
    # کد خروج نهایی بر اساس موفقیت هر دو تست
    if [ $predictive_test_status -eq 0 ] && [ $baseline_test_status -eq 0 ]; then
        exit 0
    else
        exit 1 # or use a combined error code if needed
    fi

} # اتمام بلاک اصلی که خروجی‌اش به tee هدایت می‌شود
# tee به خودی خود خروجی را هم به کنسول و هم به فایل می‌فرستد. نیازی به pipe جداگانه نیست.