// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6; // Keeping the original pragma
pragma abicoder v2; // Keep abicoder v2

// --- Imports ---
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
// TransferHelper is used implicitly by SafeERC20 for approvals etc. No direct calls needed.

// Interface for WETH
interface IWETH {
    function deposit() external payable;
    function withdraw(uint256) external;
}

/**
 * @title BaselineMinimal
 * @notice Minimal version of the baseline liquidity management contract, optimized for gas and tracking.
 * @dev Adds liquidity tracking via state variable and event. Avoids re-minting if ticks are unchanged. Includes sqrtPriceX96 in metrics event.
 */
contract BaselineMinimal is Ownable {
    using SafeERC20 for IERC20;

    // --- Main State Variables ---
    IUniswapV3Factory public immutable factory;
    INonfungiblePositionManager public positionManager;
    ISwapRouter public swapRouter;
    address public immutable token0;
    address public immutable token1;
    uint24 public immutable fee;
    int24 public tickSpacing;

    // --- Variables for position tracking ---
    uint256 public currentTokenId; // NFT ID of the current position
    uint128 public currentLiquidity; // Liquidity amount of the current position
    bool public hasPosition; // Flag indicating if a position is active
    int24 public lowerTick; // Lower tick of the current position
    int24 public upperTick; // Upper tick of the current position

    // --- Event declarations ---

    /**
     * @notice Emitted when the active position's state changes (created, removed, or ticks/liquidity updated).
     * @param tokenId The NFT token ID of the position.
     * @param hasPosition True if a position is now active, false if removed.
     * @param lowerTick The lower tick of the active position (0 if removed).
     * @param upperTick The upper tick of the active position (0 if removed).
     * @param liquidity The liquidity amount of the active position (0 if removed).
     */
    event PositionStateChanged(
        uint256 indexed tokenId,
        bool hasPosition,
        int24 lowerTick,
        int24 upperTick,
        uint128 liquidity
    );

    /**
     * @notice Emitted by the main adjustment function.
     * @param timestamp The block timestamp of the adjustment check.
     * @param sqrtPriceX96 The raw sqrtPriceX96 from the pool at the time of the check.
     * @param currentTick The pool tick corresponding to sqrtPriceX96.
     * @param targetTickLower The calculated lower tick for the target position.
     * @param targetTickUpper The calculated upper tick for the target position.
     * @param adjusted True if an on-chain action (remove/mint) was performed, false if skipped because ticks were unchanged.
     */
    event BaselineAdjustmentMetrics(
        uint256 timestamp,
        uint160 sqrtPriceX96, // Added price info
        int24 currentTick,
        int24 targetTickLower,
        int24 targetTickUpper,
        bool adjusted
    );

    /**
     * @notice Emitted when tokens are swapped within the receive function.
     * @param amountIn Amount of token swapped in.
     * @param amountOut Amount of token received from swap.
     */
    event TokensSwapped(uint256 amountIn, uint256 amountOut);

    /**
     * @notice Emitted specifically upon successful minting of a new position.
     * @param tokenId The NFT token ID of the new position.
     * @param liquidity The liquidity amount created.
     * @param tickLower The lower tick of the new position.
     * @param tickUpper The upper tick of the new position.
     * @param amount0Actual The actual amount of token0 used for minting.
     * @param amount1Actual The actual amount of token1 used for minting.
     */
    event PositionMinted(
        uint256 indexed tokenId,
        uint128 liquidity,
        int24 tickLower,
        int24 tickUpper,
        uint256 amount0Actual,
        uint256 amount1Actual
    );

    // --- Constructor ---
    constructor(
        address _factory,
        address _positionManager,
        address _swapRouter,
        address _token0,
        address _token1,
        uint24 _fee
    ) {
        require(_factory != address(0), "Invalid factory address");
        require(_positionManager != address(0), "Invalid position manager");
        require(_swapRouter != address(0), "Invalid swap router");
        require(_token0 != address(0), "Invalid token0");
        require(_token1 != address(0), "Invalid token1");
        require(_token0 != _token1, "Tokens must be different"); // Added check
        require(_fee > 0, "Invalid fee");

        factory = IUniswapV3Factory(_factory);
        positionManager = INonfungiblePositionManager(_positionManager);
        swapRouter = ISwapRouter(_swapRouter);
        token0 = _token0;
        token1 = _token1;
        fee = _fee;

        // Get and validate tickSpacing
        address poolAddress = IUniswapV3Factory(_factory).getPool(
            _token0,
            _token1,
            _fee
        );
        require(
            poolAddress != address(0),
            "Pool does not exist for specified tokens/fee"
        );
        tickSpacing = IUniswapV3Pool(poolAddress).tickSpacing();
        require(tickSpacing > 0, "Invalid tick spacing from pool");

        // Set infinite approvals for position manager and swap router
        // Consider security implications of infinite approval
        IERC20(_token0).approve(address(positionManager), type(uint256).max);
        IERC20(_token1).approve(address(positionManager), type(uint256).max);
        IERC20(_token0).approve(address(swapRouter), type(uint256).max);
        IERC20(_token1).approve(address(swapRouter), type(uint256).max);
    }

    // --- Function to receive ETH ---
    /**
     * @notice Allows the contract to receive ETH, wraps it to WETH (token1),
     * swaps half for token0, and triggers a liquidity adjustment.
     * @dev Assumes token1 is WETH. Consider adding access control.
     */
    receive() external payable {
        require(msg.value > 0, "Must send ETH");
        // Assumes token1 is WETH address
        IWETH(token1).deposit{value: msg.value}();

        // Swap half of the received WETH for token0
        uint256 halfAmountWETH = msg.value / 2;
        if (halfAmountWETH > 0) {
            _swapExactInputSingle(token1, token0, halfAmountWETH);
            // Note: swap failure is not explicitly handled, execution continues.
        }

        // Adjust liquidity using the contract's current token balances
        adjustLiquidityWithCurrentPrice();
    }

    // --- Internal Swap function ---
    /**
     * @notice Internal function to swap tokens using the Uniswap V3 router.
     * @param tokenIn Address of the input token.
     * @param tokenOut Address of the output token.
     * @param amountIn Amount of tokenIn to swap.
     * @return amountOut Amount of tokenOut received.
     */
    function _swapExactInputSingle(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter
            .ExactInputSingleParams({
                tokenIn: tokenIn,
                tokenOut: tokenOut,
                fee: fee,
                recipient: address(this), // Output tokens sent to this contract
                deadline: block.timestamp,
                amountIn: amountIn,
                amountOutMinimum: 0, // No minimum output, consider risk
                sqrtPriceLimitX96: 0 // No price limit
            });

        amountOut = swapRouter.exactInputSingle(params);
        emit TokensSwapped(amountIn, amountOut);
    }

    // --- Main liquidity adjustment function (OPTIMIZED) ---
    /**
     * @notice Reads the current pool tick, calculates a target range, and adjusts
     * the liquidity position if the target range differs from the current one.
     * @dev Skips on-chain actions (remove/mint) if target ticks match current ticks.
     */
    function adjustLiquidityWithCurrentPrice() public {
        // 1. Get current price (sqrtPriceX96) and tick information
        address pool = factory.getPool(token0, token1, fee);
        uint160 sqrtPriceX96;
        int24 currentTick;
        (sqrtPriceX96, currentTick, , , , , ) = IUniswapV3Pool(pool).slot0();

        // 2. Calculate target tick range based on currentTick
        // Uses a fixed width multiplier of 4 * tickSpacing.
        int24 halfWidth = (tickSpacing * 4) / 2;
        if (halfWidth < tickSpacing) {
            // Ensure minimum width
            halfWidth = tickSpacing;
        }
        // Calculate boundaries based on current tick, aligned to tickSpacing
        int24 targetLowerTick = ((currentTick - halfWidth) / tickSpacing) *
            tickSpacing;
        int24 targetUpperTick = ((currentTick + halfWidth) / tickSpacing) *
            tickSpacing;

        // Ensure upper > lower and handle potential crossing after alignment
        if (targetLowerTick >= targetUpperTick) {
            targetUpperTick = targetLowerTick + tickSpacing;
        }
        // Basic TickMath boundary checks (approximated for int24)
        int24 minTick = -887272;
        int24 maxTick = 887272;
        if (targetLowerTick < minTick) targetLowerTick = minTick;
        if (targetUpperTick > maxTick) targetUpperTick = maxTick;
        // Final check after boundary adjustment
        if (targetLowerTick >= targetUpperTick) {
            // If maxTick was hit, adjust lower tick downwards
            if (targetUpperTick == maxTick) {
                targetLowerTick = targetUpperTick - tickSpacing;
            } else {
                targetUpperTick = targetLowerTick + tickSpacing;
            }
        }
        // Ensure ticks are still valid after potential adjustments
        require(
            targetLowerTick >= minTick &&
                targetUpperTick <= maxTick &&
                targetLowerTick < targetUpperTick,
            "Tick calculation error"
        );

        // *** OPTIMIZATION: Check if adjustment is actually needed ***
        if (
            hasPosition &&
            targetLowerTick == lowerTick &&
            targetUpperTick == upperTick
        ) {
            // Position exists and target ticks are the same as current ticks.
            // Emit metrics but skip on-chain remove/create operations.
            emit BaselineAdjustmentMetrics(
                block.timestamp,
                sqrtPriceX96, // Include current price info
                currentTick,
                targetLowerTick,
                targetUpperTick,
                false // Adjusted = false
            );
            return; // Exit early to save gas
        }

        // --- Adjustment is needed ---

        // 3. Remove existing position if it exists
        // _removePosition resets state: hasPosition, currentTokenId, currentLiquidity, lowerTick, upperTick
        if (hasPosition) {
            _removePosition();
        }

        // 4. Create new position with current token balances at the target ticks
        // _createPosition updates state if mint is successful
        _createPosition(targetLowerTick, targetUpperTick);

        // 5. Emit metrics indicating an adjustment was performed (or attempted)
        emit BaselineAdjustmentMetrics(
            block.timestamp,
            sqrtPriceX96, // Include current price info
            currentTick,
            targetLowerTick,
            targetUpperTick,
            true // Adjusted = true (on-chain action performed)
        );
    }

    // --- Internal function to remove position (UPDATED) ---
    /**
     * @notice Removes liquidity, collects fees/tokens, and burns the NFT. Resets position state.
     * @dev Uses try-catch for resilience against external call failures.
     */
    function _removePosition() internal {
        require(hasPosition, "No position to remove"); // Should already be checked by caller

        uint256 _tokenId = currentTokenId; // Cache tokenId

        // Reset state *before* external calls (Checks-Effects-Interactions)
        hasPosition = false;
        currentTokenId = 0;
        uint128 _oldLiquidity = currentLiquidity; // Cache liquidity for event if needed
        currentLiquidity = 0; // Reset liquidity state
        int24 _oldLowerTick = lowerTick;
        int24 _oldUpperTick = upperTick;
        lowerTick = 0;
        upperTick = 0;

        // Emit state change *after* state reset
        emit PositionStateChanged(_tokenId, false, 0, 0, 0); // Signal removal

        // --- External Interactions (decrease, collect, burn) ---
        // Use try-catch to handle potential reverts in Uniswap contracts
        try
            positionManager.decreaseLiquidity(
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: _tokenId,
                    liquidity: type(uint128).max, // Attempt to remove all liquidity
                    amount0Min: 0,
                    amount1Min: 0,
                    deadline: block.timestamp
                })
            )
        returns (
            // No return values needed here for core logic
            uint256 amount0,
            uint256 amount1
        ) {
            // Can optionally log amount0, amount1 off-chain if needed
        } catch {
            // Silently ignore decreaseLiquidity failure, proceed to collect/burn
        }

        try
            positionManager.collect(
                INonfungiblePositionManager.CollectParams({
                    tokenId: _tokenId,
                    recipient: address(this), // Collect to this contract
                    amount0Max: type(uint128).max,
                    amount1Max: type(uint128).max
                })
            )
        returns (
            // No return values needed here for core logic
            uint256 collected0,
            uint256 collected1
        ) {
            // Can optionally log collected0, collected1 off-chain if needed
        } catch {
            // Silently ignore collect failure, proceed to burn
        }

        // Burn the NFT regardless of previous steps' success/failure
        try positionManager.burn(_tokenId) {} catch {
            // Silently ignore burn failure
        }
    }

    // --- Internal function to create position (UPDATED) ---
    /**
     * @notice Mints a new Uniswap V3 position with the contract's available token balances.
     * @param _newLowerTick The target lower tick for the new position.
     * @param _newUpperTick The target upper tick for the new position.
     * @dev Updates position state variables and emits events on success. Uses try-catch.
     */
    function _createPosition(
        int24 _newLowerTick,
        int24 _newUpperTick
    ) internal {
        require(_newLowerTick < _newUpperTick, "Invalid tick range");

        uint256 amount0Desired = IERC20(token0).balanceOf(address(this));
        uint256 amount1Desired = IERC20(token1).balanceOf(address(this));

        // Skip minting if contract has no tokens to provide
        if (amount0Desired == 0 && amount1Desired == 0) {
            // Optional: Log or emit event indicating skip?
            return;
        }

        // Attempt to mint the position
        try
            positionManager.mint(
                INonfungiblePositionManager.MintParams({
                    token0: token0,
                    token1: token1,
                    fee: fee,
                    tickLower: _newLowerTick,
                    tickUpper: _newUpperTick,
                    amount0Desired: amount0Desired, // Use available balance
                    amount1Desired: amount1Desired, // Use available balance
                    amount0Min: 0, // Allow partial fills if price is at edge
                    amount1Min: 0,
                    recipient: address(this), // This contract owns the NFT
                    deadline: block.timestamp
                })
            )
        returns (
            // Capture the return values from mint
            uint256 _tokenId,
            uint128 _liquidity,
            uint256 _amount0Actual,
            uint256 _amount1Actual
        ) {
            // Check if liquidity was actually created
            if (_liquidity > 0) {
                // --- Update state on successful mint ---
                currentTokenId = _tokenId;
                currentLiquidity = _liquidity; // Store the new liquidity amount
                hasPosition = true;
                lowerTick = _newLowerTick;
                upperTick = _newUpperTick;

                // --- Emit events on successful mint ---
                emit PositionStateChanged(
                    _tokenId,
                    true, // hasPosition
                    _newLowerTick,
                    _newUpperTick,
                    _liquidity // Include liquidity
                );
                emit PositionMinted(
                    _tokenId,
                    _liquidity,
                    _newLowerTick,
                    _newUpperTick,
                    _amount0Actual,
                    _amount1Actual
                );
            } else {
                // Mint call succeeded but returned 0 liquidity (e.g., amounts too small)
                // If a (useless) NFT was created, try to burn it.
                if (_tokenId > 0) {
                    try positionManager.burn(_tokenId) {} catch {}
                }
                // Ensure state remains consistent (no active position)
                if (currentTokenId != _tokenId) {
                    // Avoid resetting if somehow state got updated partially
                    hasPosition = false;
                    currentLiquidity = 0;
                    lowerTick = 0;
                    upperTick = 0;
                }
            }
        } catch Error(string memory reason) {
            // Catch specific revert reasons from mint
            // Optional: Emit an event with the revert reason for off-chain logging
            // emit MintFailed(reason);
            // Ensure state reflects no active position after failure
            hasPosition = false;
            currentTokenId = 0;
            currentLiquidity = 0;
            lowerTick = 0;
            upperTick = 0;
        } catch {
            // Catch generic errors during mint
            // Ensure state reflects no active position after failure
            hasPosition = false;
            currentTokenId = 0;
            currentLiquidity = 0;
            lowerTick = 0;
            upperTick = 0;
        }
    }

    // --- Emergency functions (Owner only) ---

    /**
     * @notice Allows the owner to withdraw any ERC20 token from this contract.
     * @param token The address of the ERC20 token to withdraw.
     * @param to The address to send the tokens to.
     */
    function rescueTokens(address token, address to) external onlyOwner {
        require(token != address(0), "Invalid token address");
        require(to != address(0), "Invalid recipient address");
        uint256 amount = IERC20(token).balanceOf(address(this));
        if (amount > 0) {
            IERC20(token).safeTransfer(to, amount);
        }
    }

    /**
     * @notice Allows the owner to withdraw any ETH held by this contract.
     */
    function rescueETH() external onlyOwner {
        uint256 balance = address(this).balance;
        if (balance > 0) {
            // Using call for safer ETH transfer (no fixed gas limit)
            (bool success, ) = owner().call{value: balance}("");
            require(success, "ETH rescue failed");
        }
    }

    /**
     * @notice Allows the owner to withdraw WETH (token1) as ETH.
     * @dev Assumes token1 is WETH. Withdraws WETH, receives ETH, sends ETH to owner.
     * @param amount The amount of WETH (in wei) to withdraw as ETH.
     */
    function rescueWETH(uint256 amount) external onlyOwner {
        require(amount > 0, "Amount must be positive");
        // Assumes token1 is WETH address
        IWETH(token1).withdraw(amount); // Converts WETH to ETH within this contract
        // Forward the received ETH to the owner
        (bool success, ) = owner().call{value: amount}("");
        require(success, "WETH rescue failed: ETH transfer failed");
    }

    // --- Read-only function to get current position details ---
    /**
     * @notice Returns the details of the currently managed liquidity position.
     * @return tokenId The NFT ID of the current position (0 if none).
     * @return active True if a position is currently managed, false otherwise.
     * @return tickLower The lower tick of the current position (0 if none).
     * @return tickUpper The upper tick of the current position (0 if none).
     * @return liquidity The liquidity amount of the current position (0 if none).
     */
    function getCurrentPosition()
        external
        view
        returns (
            uint256 tokenId,
            bool active,
            int24 tickLower,
            int24 tickUpper,
            uint128 liquidity
        )
    {
        return (
            currentTokenId,
            hasPosition,
            lowerTick,
            upperTick,
            currentLiquidity
        );
    }
}
