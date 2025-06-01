#!/bin/bash

# ==============================================
# Enhanced DEX Fork Test Automation Script
# Version: 2.4
# ==============================================

# --- Configuration Section ---
PROJECT_DIR_DEFAULT="/root/Uni-Dex-Marketplace_test/Phase3_Smart_Contract" # مسیر پیش‌فرض پروژه شما
PROJECT_DIR="${FORK_TEST_PROJECT_DIR:-$PROJECT_DIR_DEFAULT}" # اجازه می‌دهد از طریق متغیر محیطی بازنویسی شود

LOG_FILE_DIR="$PROJECT_DIR/logs" # پوشه‌ای مجزا برای لاگ‌ها
mkdir -p "$LOG_FILE_DIR" # ایجاد پوشه لاگ‌ها اگر وجود نداشته باشد
LOG_FILE="$LOG_FILE_DIR/fork_test_run_$(date +%Y%m%d_%H%M%S).log"
HARDHAT_NODE_LOG_FILE="$LOG_FILE_DIR/hardhat_node_$(date +%Y%m%d_%H%M%S).log"

MAX_RETRIES=3 # حداکثر تعداد تلاش مجدد برای عملیات ناموفق
RETRY_DELAY=10 # تاخیر بین تلاش‌های مجدد (ثانیه)

# Script Paths
DEPLOY_SCRIPT_PREDICTIVE="$PROJECT_DIR/scripts/deployPredictiveManager.js"
DEPLOY_SCRIPT_BASELINE="$PROJECT_DIR/scripts/deployMinimal.js"
DEPLOY_SCRIPT_TOKEN_MANAGER="$PROJECT_DIR/scripts/deployTokenManagerOptimized.js"

# Test Script Paths
PYTHON_SCRIPT_PREDICTIVE="$PROJECT_DIR/test/predictive/predictive_test.py"
PYTHON_SCRIPT_BASELINE="$PROJECT_DIR/test/baseline/baseline_test.py"

# Address Files (مسیر فایل‌های آدرس نسبت به PROJECT_DIR)
ADDRESS_FILE_PREDICTIVE="$PROJECT_DIR/predictiveManager_address.json"
ADDRESS_FILE_BASELINE="$PROJECT_DIR/baselineMinimal_address.json"
ADDRESS_FILE_TOKEN_MANAGER="$PROJECT_DIR/tokenManagerOptimized_address.json"

# Utility Scripts
FUNDING_SCRIPT="$PROJECT_DIR/test/utils/fund_my_wallet.py"
ENV_FILE="$PROJECT_DIR/.env" # فایل متغیرهای محیطی

# Network Configuration
LOCAL_RPC_URL="http://127.0.0.1:8545" # آدرس RPC محلی که Hardhat Node روی آن اجرا می‌شود
HARDHAT_PORT=8545 # پورت Hardhat Node
HARDHAT_HOST="0.0.0.0" # هاست Hardhat Node (0.0.0.0 برای دسترسی از همه جا، 127.0.0.1 برای دسترسی فقط محلی)

# Python Environment
VENV_PATH="$PROJECT_DIR/venv" # مسیر پوشه محیط مجازی پایتون
VENV_ACTIVATE="$VENV_PATH/bin/activate" # اسکریپت فعال‌سازی محیط مجازی

# --- Helper Functions ---
function log() {
    # هم به کنسول و هم به فایل اصلی لاگ می‌نویسد
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE" # [cite: 495]
}

function validate_environment() {
    log "Validating environment prerequisites..."
    local all_ok=true
    local required_commands=(node npm python3 curl jq git) # [cite: 496]
    for cmd in "${required_commands[@]}"; do # [cite: 496]
        if ! command -v "$cmd" &> /dev/null; then # [cite: 497]
            log "ERROR: Required command '$cmd' not found in PATH." # [cite: 498]
            all_ok=false # [cite: 498]
        fi
    done

    if [ ! -f "$ENV_FILE" ]; then # [cite: 499]
        log "ERROR: Environment file .env not found at $ENV_FILE"
        all_ok=false
    fi
    
    if [ ! -d "$PROJECT_DIR" ]; then # [cite: 500]
        log "ERROR: Project directory $PROJECT_DIR does not exist."
        all_ok=false # [cite: 501]
    fi

    # بررسی وجود اسکریپت‌های اصلی
    local required_scripts=(
        "$DEPLOY_SCRIPT_PREDICTIVE" 
        "$DEPLOY_SCRIPT_BASELINE" 
        "$DEPLOY_SCRIPT_TOKEN_MANAGER" 
        "$PYTHON_SCRIPT_PREDICTIVE" 
        "$PYTHON_SCRIPT_BASELINE" 
        "$FUNDING_SCRIPT"
    ) # [cite: 502]
    for script_path in "${required_scripts[@]}"; do # [cite: 502]
        if [ ! -f "$script_path" ]; then # [cite: 503]
            log "ERROR: Required script '$script_path' not found." # [cite: 504]
            all_ok=false # [cite: 504]
        fi
    done
    
    if [ "$all_ok" = false ]; then # [cite: 505]
        log "Environment validation failed. Please check messages above." # [cite: 506]
        exit 1 # [cite: 506]
    fi
    log "Environment validation successful." # [cite: 507]
}

function kill_hardhat_node() {
    log "Attempting to stop any existing Hardhat node on port $HARDHAT_PORT..."
    # استفاده از pkill با دقت بیشتر برای جلوگیری از kill کردن فرآیندهای اشتباه
    PIDS=$(pgrep -f "hardhat node.*--port $HARDHAT_PORT") # [cite: 508]
    if [ -n "$PIDS" ]; then # [cite: 508]
        kill $PIDS # [cite: 508]
        sleep 2 # زمان برای بسته شدن آرام # [cite: 508]
        PIDS=$(pgrep -f "hardhat node.*--port $HARDHAT_PORT") # Check again # [cite: 509]
        if [ -n "$PIDS" ]; then # [cite: 509]
            log "Hardhat node still running. Force killing..." # [cite: 509]
            kill -9 $PIDS # [cite: 510]
            sleep 1 # [cite: 510]
        fi
    fi
    if ! pgrep -f "hardhat node.*--port $HARDHAT_PORT" > /dev/null; then # [cite: 511]
        log "Hardhat node stopped successfully or was not running." # [cite: 512]
    else
        log "WARNING: Could not stop existing Hardhat node. Manual check might be needed." # [cite: 513]
    fi
}

function check_rpc_ready() {
    local url_to_check=$1
    local max_check_retries=25 
    local check_delay=6    
    log "Checking if RPC at $url_to_check is ready..."
    for attempt in $(seq 1 $max_check_retries); do # [cite: 514]
        # استفاده از یک درخواست ساده net_version
        if curl -s -X POST --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
           -H "Content-Type: application/json" "$url_to_check" --connect-timeout 5 2>/dev/null | jq -e '.result' > /dev/null; then # [cite: 515]
            log "RPC at $url_to_check is ready." # [cite: 516]
            return 0 # موفقیت # [cite: 516]
        fi
        log "RPC not ready yet (attempt $attempt/$max_check_retries). Retrying in $check_delay seconds..."
        sleep $check_delay
    done
    log "ERROR: RPC at $url_to_check did not become ready after $max_check_retries attempts." # [cite: 517]
    return 1 # شکست # [cite: 517]
}

function deploy_contract() {
    local script_path="$1"
    local address_file="$2"
    local contract_name="$3"
    
    log "Deploying $contract_name contract using $script_path..."
    rm -f "$address_file" # اطمینان از ایجاد فایل آدرس جدید

    for attempt in $(seq 1 $MAX_RETRIES); do # [cite: 518]
        log "Attempt $attempt of $MAX_RETRIES to deploy $contract_name..."
        # اجرای دستور deploy و هدایت stdout و stderr به فایل لاگ اصلی
        if npx hardhat run "$script_path" --network localhost >> "$LOG_FILE" 2>&1; then # [cite: 519]
            if [ -f "$address_file" ] && jq -e '.address' "$address_file" > /dev/null 2>&1; then # [cite: 520]
                local contract_address
                contract_address=$(jq -r '.address' "$address_file") # [cite: 520]
                # بررسی ساده فرمت آدرس (شروع با 0x و طول مناسب)
                if [[ "$contract_address" =~ ^0x[a-fA-F0-9]{40}$ ]]; then # [cite: 521]
                    log "$contract_name deployed successfully! Address: $contract_address (saved to $address_file)"
                    return 0 # موفقیت
                else
                    log "ERROR: Deployment script for $contract_name ran, but address in $address_file ('$contract_address') is invalid." # [cite: 522]
                fi
            else
                log "ERROR: Deployment script for $contract_name ran, but address file $address_file was not created or is invalid." # [cite: 523]
            fi
        else
            log "ERROR: Hardhat deployment script $script_path failed for $contract_name on attempt $attempt." # [cite: 524]
        fi
        
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then # [cite: 525]
            log "Retrying deployment of $contract_name in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    log "FATAL: Failed to deploy $contract_name after $MAX_RETRIES attempts. Check $LOG_FILE and $HARDHAT_NODE_LOG_FILE for details." # [cite: 526]
    return 1 # شکست # [cite: 526]
}

function run_python_test() {
    local script_path="$1"
    local test_name="$2"
    
    log "Running Python test script: $test_name ($script_path)..."
    # اجرای اسکریپت پایتون و هدایت خروجی آن
    # استفاده از python3 -u برای خروجی بدون بافر (بهتر برای لاگ‌گیری زنده)
    if python3 -u "$script_path" >> "$LOG_FILE" 2>&1; then # [cite: 527]
        log "✅ $test_name test completed successfully." # [cite: 528]
        return 0 # موفقیت # [cite: 528]
    else
        log "❌ ERROR: $test_name test failed. Check $LOG_FILE for details." # [cite: 529]
        return 1 # شکست # [cite: 529]
    fi
}

# --- Main Execution ---
{
    log "=============================================="
    log "🚀 Starting DEX Fork Test Automation Script 🚀"
    log "=============================================="
    log "Project Directory: $PROJECT_DIR"
    log "Main Log File: $LOG_FILE"
    log "Hardhat Node Log File: $HARDHAT_NODE_LOG_FILE"
    
    validate_environment # ابتدا محیط را بررسی کن
    
    cd "$PROJECT_DIR" || { log "CRITICAL: Failed to change to project directory $PROJECT_DIR. Exiting."; exit 1; } # [cite: 530, 531]
    log "Changed directory to $PROJECT_DIR" # [cite: 531]

    # --- ADDED CLEAN AND COMPILE ---
    log "Cleaning previous Hardhat artifacts..."
    if npx hardhat clean >> "$LOG_FILE" 2>&1; then
        log "Hardhat clean successful."
    else
        # این یک هشدار است چون ممکن است Hardhat بدون clean هم کار کند، اما clean بهتر است.
        log "WARNING: Hardhat clean failed. Continuing, but this might cause issues."
    fi

    log "Compiling contracts..."
    if npx hardhat compile >> "$LOG_FILE" 2>&1; then
        log "Contracts compiled successfully."
    else
        log "CRITICAL: Contract compilation failed. Check $LOG_FILE. Exiting."
        exit 1
    fi
    # --- END OF ADDED CLEAN AND COMPILE ---
    
    if [ -f "$VENV_ACTIVATE" ]; then # [cite: 532]
        log "Activating Python virtual environment from $VENV_ACTIVATE..."
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE" # [cite: 532]
        log "Python virtual environment activated. Current python: $(command -v python3)"
    else
        log "WARNING: Python virtual environment activation script not found at $VENV_ACTIVATE. Using system Python." # [cite: 533]
    fi
    
    kill_hardhat_node # پاکسازی اجرای قبلی
    # حذف فایل‌های آدرس قدیمی برای اطمینان از ایجاد مجدد
    log "Removing old address files (if any)..."
    rm -f "$ADDRESS_FILE_PREDICTIVE" "$ADDRESS_FILE_BASELINE" "$ADDRESS_FILE_TOKEN_MANAGER"
    
    log "Loading environment variables from $ENV_FILE..."
    if [ -f "$ENV_FILE" ]; then # [cite: 534]
        set -o allexport # تمام متغیرهای تعریف شده را export کن # [cite: 534]
        # shellcheck source=/dev/null
        source "$ENV_FILE" # [cite: 534]
        set +o allexport # [cite: 534]
        log "Environment variables loaded." # [cite: 535]
        # بررسی متغیرهای محیطی ضروری
        env_check_ok=true
        required_env_vars=("MAINNET_RPC_URL" "PRIVATE_KEY" "DEPLOYER_ADDRESS" "LSTM_API_URL") # [cite: 535]
        for var_name in "${required_env_vars[@]}"; do # [cite: 536]
            if [ -z "${!var_name}" ]; then # بررسی اینکه متغیر خالی نباشد # [cite: 537]
                log "ERROR: Required environment variable '$var_name' is not set in $ENV_FILE." # [cite: 538]
                env_check_ok=false # [cite: 538]
            fi
        done
        if [ "$env_check_ok" = false ]; then # [cite: 539]
            log "CRITICAL: Missing required environment variables. Exiting." # [cite: 540]
            exit 1 # [cite: 540]
        fi
        log "Required environment variables seem to be set." # [cite: 541]
    else
        log "CRITICAL: .env file not found at $ENV_FILE. Exiting." # [cite: 542]
        exit 1 # [cite: 542]
    fi
    
    log "Starting Hardhat node with Mainnet fork from $MAINNET_RPC_URL..."
    # اجرای Hardhat node در پس‌زمینه و ذخیره لاگ آن در فایل مجزا
    nohup npx hardhat node --hostname "$HARDHAT_HOST" --port "$HARDHAT_PORT" \
        --fork "$MAINNET_RPC_URL" > "$HARDHAT_NODE_LOG_FILE" 2>&1 & # [cite: 543]
    HARDHAT_PID=$! # [cite: 543]
    log "Hardhat node started with PID: $HARDHAT_PID. Waiting for it to be ready..." # [cite: 543]
    
    if ! check_rpc_ready "$LOCAL_RPC_URL"; then # [cite: 544]
        log "CRITICAL: Hardhat node failed to start or become ready. Check $HARDHAT_NODE_LOG_FILE. Exiting." # [cite: 545]
        kill_hardhat_node # سعی در kill کردن دوباره # [cite: 545]
        exit 1 # [cite: 545]
    fi
    log "Hardhat node is ready and listening on $LOCAL_RPC_URL"
    
    # صادرات متغیر محیطی برای استفاده در اسکریپت‌های پایتون
    export MAINNET_FORK_RPC_URL="$LOCAL_RPC_URL" 
    log "Exported MAINNET_FORK_RPC_URL as $LOCAL_RPC_URL for Python scripts." # [cite: 546]

    log "Funding deployer account ($DEPLOYER_ADDRESS) using $FUNDING_SCRIPT..."
    # اطمینان از اینکه DEPLOYER_ADDRESS برای اسکریپت fund_my_wallet.py در دسترس است
    # این متغیر باید از .env خوانده شده باشد
    if python3 -u "$FUNDING_SCRIPT"; then # خروجی این اسکریپت به طور خودکار به LOG_FILE می‌رود # [cite: 547]
        log "Deployer account funding script executed. Check its specific logs if issues persist." # [cite: 548]
    else
        log "CRITICAL: Failed to execute funding script for deployer account. Check $LOG_FILE. Exiting." # [cite: 549]
        kill_hardhat_node # [cite: 549]
        exit 1 # [cite: 549]
    fi
    
    # استقرار قراردادها
    # برای هر قرارداد، یک بلوک if جداگانه برای خروج در صورت شکست
    if ! deploy_contract "$DEPLOY_SCRIPT_PREDICTIVE" "$ADDRESS_FILE_PREDICTIVE" "PredictiveManager"; then # [cite: 550]
        log "CRITICAL: Deployment of PredictiveManager failed. Exiting." # [cite: 551]
        kill_hardhat_node # [cite: 551]
        exit 1 # [cite: 551]
    fi
    
    if ! deploy_contract "$DEPLOY_SCRIPT_BASELINE" "$ADDRESS_FILE_BASELINE" "BaselineMinimal"; then # [cite: 552]
        log "CRITICAL: Deployment of BaselineMinimal failed. Exiting." # [cite: 553]
        kill_hardhat_node # [cite: 553]
        exit 1 # [cite: 553]
    fi

    if ! deploy_contract "$DEPLOY_SCRIPT_TOKEN_MANAGER" "$ADDRESS_FILE_TOKEN_MANAGER" "TokenManagerOptimized"; then
        log "CRITICAL: Deployment of TokenManagerOptimized failed. Exiting."
        kill_hardhat_node
        exit 1
    fi
    
    # Export contract addresses for Python scripts
    if [ -f "$ADDRESS_FILE_PREDICTIVE" ]; then
        export PREDICTIVE_MANAGER_ADDRESS=$(jq -r '.address' "$ADDRESS_FILE_PREDICTIVE")
        log "Exported PREDICTIVE_MANAGER_ADDRESS=${PREDICTIVE_MANAGER_ADDRESS}"
    else log "ERROR: Predictive address file not found!"; kill_hardhat_node; exit 1; fi

    if [ -f "$ADDRESS_FILE_BASELINE" ]; then
        export BASELINE_MINIMAL_ADDRESS=$(jq -r '.address' "$ADDRESS_FILE_BASELINE")
        log "Exported BASELINE_MINIMAL_ADDRESS=${BASELINE_MINIMAL_ADDRESS}"
    else log "ERROR: Baseline address file not found!"; kill_hardhat_node; exit 1; fi
    
    if [ -f "$ADDRESS_FILE_TOKEN_MANAGER" ]; then
        export TOKEN_MANAGER_OPTIMIZED_ADDRESS=$(jq -r '.address' "$ADDRESS_FILE_TOKEN_MANAGER")
        log "Exported TOKEN_MANAGER_OPTIMIZED_ADDRESS=${TOKEN_MANAGER_OPTIMIZED_ADDRESS}"
    else log "ERROR: Token Manager Optimized address file not found!"; kill_hardhat_node; exit 1; fi

    # اجرای تست‌های پایتون
    log "--- Starting Python Tests ---"
    predictive_test_status=1 # Default to failure
    baseline_test_status=1   # Default to failure

    run_python_test "$PYTHON_SCRIPT_PREDICTIVE" "Predictive Strategy" # [cite: 554]
    predictive_test_status=$? # [cite: 554]

    run_python_test "$PYTHON_SCRIPT_BASELINE" "Baseline Strategy"
    baseline_test_status=$?
    
    log "--- Python Tests Finished ---"

    # Final cleanup
    log "Cleaning up: Stopping Hardhat node (PID: $HARDHAT_PID)..."
    kill_hardhat_node
    
    if type deactivate &> /dev/null && [[ -n "$VIRTUAL_ENV" ]]; then # [cite: 555]
        log "Deactivating Python virtual environment..." # [cite: 555]
        deactivate # [cite: 555]
    fi
    
    log "=============================================="
    log "🏁 Test Automation Script Completed 🏁"
    log "=============================================="
    log "Predictive Test Result: $( [ $predictive_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )" # [cite: 556]
    log "Baseline Test Result:   $( [ $baseline_test_status -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILURE" )" # [cite: 556]
    log "Detailed logs available in: $LOG_FILE"
    log "Hardhat node logs (if any issues): $HARDHAT_NODE_LOG_FILE"
    
    # کد خروج نهایی بر اساس موفقیت هر دو تست
    if [ $predictive_test_status -eq 0 ] && [ $baseline_test_status -eq 0 ]; then # [cite: 557]
        exit 0 # [cite: 557]
    else
        exit 1 # or use a combined error code if needed # [cite: 557]
    fi

} # اتمام بلاک اصلی که خروجی‌اش به tee هدایت می‌شود # [cite: 558]
# tee به خودی خود خروجی را هم به کنسول و هم به فایل می‌فرستد. نیازی به pipe جداگانه نیست. # [cite: 558]