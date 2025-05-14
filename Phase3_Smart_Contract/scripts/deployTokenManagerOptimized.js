const hre = require("hardhat");
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    try {
        const [deployer] = await hre.ethers.getSigners();
        console.log("=========================================");
        console.log("Deploying optimized contract with the account:", deployer.address);
        console.log("Account balance:", (await deployer.getBalance()).toString());
        console.log("=========================================");

        // آدرس‌های Uniswap V3 برای شبکه سپولیا
        const SWAP_ROUTER = "0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E";
        const WETH = "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9";

        console.log("Using the following addresses:");
        console.log("- Swap Router:", SWAP_ROUTER);
        console.log("- WETH:", WETH);
        console.log("=========================================");

        // تنظیمات بهینه برای دیپلوی
        const deploymentOptions = {
            gasLimit: 2000000, // کاهش به 2 میلیون
            gasPrice: ethers.utils.parseUnits("30", "gwei") // استفاده از gasPrice به جای maxFeePerGas
        };

        console.log("Using deployment options:");
        console.log("- Gas Limit:", deploymentOptions.gasLimit.toString());
        console.log("- Gas Price:", ethers.utils.formatUnits(deploymentOptions.gasPrice, "gwei"), "gwei");
        console.log("=========================================");

        console.log("Deploying TokenOperationsManagerOptimized contract...");
        const TokenOperationsManagerOptimized = await hre.ethers.getContractFactory("TokenOperationsManagerOptimized");

        console.log("Creating deployment transaction...");
        const tokenManager = await TokenOperationsManagerOptimized.deploy(
            SWAP_ROUTER,
            WETH,
            deploymentOptions
        );

        console.log("Waiting for TokenOperationsManagerOptimized deployment transaction to be mined...");
        console.log("Transaction hash:", tokenManager.deployTransaction.hash);
        await tokenManager.deployed();
        console.log("✅ TokenOperationsManagerOptimized deployed to:", tokenManager.address);

        // ذخیره آدرس‌های قراردادها در فایل .env
        updateEnvFile({
            TOKEN_MANAGER_OPTIMIZED_ADDRESS: tokenManager.address
        });

        console.log("=========================================");
        console.log("Deployment completed successfully!");
        console.log("TokenOperationsManagerOptimized:", tokenManager.address);
        console.log("=========================================");
        console.log("Next steps:");
        console.log("1. Verify contract on Etherscan:");
        console.log(`   npx hardhat verify --network sepolia ${tokenManager.address} ${SWAP_ROUTER} ${WETH}`);
        console.log("2. Fund contract with USDC and WETH");
        console.log("=========================================");
    } catch (error) {
        console.error("❌ Deployment failed with error:", error);

        // نمایش جزئیات بیشتر در صورت وجود
        if (error.reason) {
            console.error("Error reason:", error.reason);
        }

        if (error.code) {
            console.error("Error code:", error.code);
        }

        if (error.transaction) {
            console.error("Failed transaction details:", {
                hash: error.transaction.hash,
                from: error.transaction.from,
                to: error.transaction.to,
                gasLimit: error.transaction.gasLimit.toString(),
                gasPrice: error.transaction.gasPrice
                    ? ethers.utils.formatUnits(error.transaction.gasPrice, "gwei") + " gwei"
                    : "unknown"
            });
        }

        throw error;
    }
}

// تابع به‌روزرسانی فایل .env
function updateEnvFile(addresses) {
    try {
        const envPath = path.resolve(__dirname, '../.env');
        let envContent = '';

        try {
            envContent = fs.readFileSync(envPath, 'utf8');
        } catch (error) {
            console.log("Creating new .env file");
            envContent = '';
        }

        // به‌روزرسانی یا اضافه کردن آدرس‌های قراردادها
        for (const [key, value] of Object.entries(addresses)) {
            if (envContent.includes(`${key}=`)) {
                envContent = envContent.replace(
                    new RegExp(`${key}=.*`),
                    `${key}="${value}"`
                );
            } else {
                envContent += `\n${key}="${value}"\n`;
            }
        }

        fs.writeFileSync(envPath, envContent);
        console.log(`Contract addresses saved to .env file`);
    } catch (error) {
        console.warn("Could not update .env file:", error.message);
    }
}

main().catch((error) => {
    console.error("Unhandled error:", error);
    process.exit(1);
}); 