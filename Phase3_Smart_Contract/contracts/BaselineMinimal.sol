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
    address public immutable token0;
    address public immutable token1;
    uint24 public immutable fee;
    int24 public tickSpacing;
    uint24 public rangeWidthMultiplier;

    // --- Variables for position tracking ---
    uint256 public currentTokenId; // NFT ID of the current position
    uint128 public currentLiquidity; // Liquidity amount of the current position
    bool public hasPosition; // Flag indicating if a position is active
    int24 public lowerTick; // Lower tick of the current position
    int24 public upperTick; // Upper tick of the current position

    // --- Event declarations ---
    event StrategyParamUpdated(string indexed paramName, uint256 newValue);

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
        address _token0,
        address _token1,
        uint24 _fee,
        uint24 _initialRangeWidthMultiplier
    ) {
        require(_factory != address(0), "Invalid factory address");
        require(_positionManager != address(0), "Invalid position manager");
        require(_token0 != address(0), "Invalid token0");
        require(_token1 != address(0), "Invalid token1");
        require(_token0 != _token1, "Tokens must be different");
        require(_fee > 0, "Invalid fee");

        factory = IUniswapV3Factory(_factory);
        positionManager = INonfungiblePositionManager(_positionManager);
        token0 = _token0;
        token1 = _token1;
        fee = _fee;

        require(_initialRangeWidthMultiplier > 0, "Initial RWM must be > 0");
        rangeWidthMultiplier = _initialRangeWidthMultiplier;

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

        // Set infinite approvals for position manager
        IERC20(_token0).approve(address(positionManager), type(uint256).max);
        IERC20(_token1).approve(address(positionManager), type(uint256).max);
    }

    // --- Function to set Range Width Multiplier ---
    function setRangeWidthMultiplier(uint24 _newMultiplier) external onlyOwner {
        require(_newMultiplier > 0, "RWM must be > 0");
        rangeWidthMultiplier = _newMultiplier;
        emit StrategyParamUpdated("rangeWidthMultiplier", uint256(_newMultiplier));
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
        int24 halfWidth = (tickSpacing * int24(rangeWidthMultiplier)) / 2;
        if (halfWidth < tickSpacing) {
            halfWidth = tickSpacing;
        }
        int24 targetLowerTick = ((currentTick - halfWidth) / tickSpacing) * tickSpacing;
        int24 targetUpperTick = ((currentTick + halfWidth) / tickSpacing) * tickSpacing;

        if (targetLowerTick >= targetUpperTick) {
            targetUpperTick = targetLowerTick + tickSpacing;
        }
        int24 minTick = -887272;
        int24 maxTick = 887272;
        if (targetLowerTick < minTick) targetLowerTick = minTick;
        if (targetUpperTick > maxTick) targetUpperTick = maxTick;
        if (targetLowerTick >= targetUpperTick) {
            if (targetUpperTick == maxTick) {
                targetLowerTick = targetUpperTick - tickSpacing;
            } else {
                targetUpperTick = targetLowerTick + tickSpacing;
            }
        }
        require(
            targetLowerTick >= minTick &&
                targetUpperTick <= maxTick &&
                targetLowerTick < targetUpperTick,
            "Tick calculation error"
        );

        if (
            hasPosition &&
            targetLowerTick == lowerTick &&
            targetUpperTick == upperTick
        ) {
            emit BaselineAdjustmentMetrics(
                block.timestamp,
                sqrtPriceX96,
                currentTick,
                targetLowerTick,
                targetUpperTick,
                false
            );
            return;
        }

        if (hasPosition) {
            _removePosition();
        }

        _createPosition(targetLowerTick, targetUpperTick);

        emit BaselineAdjustmentMetrics(
            block.timestamp,
            sqrtPriceX96,
            currentTick,
            targetLowerTick,
            targetUpperTick,
            true
        );
    }

    // --- Internal function to remove position ---
    /**
     * @notice Removes liquidity, collects fees/tokens, and burns the NFT. Resets position state.
     * @dev Uses try-catch for resilience against external call failures.
     */
    function _removePosition() internal {
        require(hasPosition, "No position to remove");

        uint256 _tokenId = currentTokenId;

        hasPosition = false;
        currentTokenId = 0;
        currentLiquidity = 0;
        lowerTick = 0;
        upperTick = 0;

        emit PositionStateChanged(_tokenId, false, 0, 0, 0);

        try
            positionManager.decreaseLiquidity(
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: _tokenId,
                    liquidity: type(uint128).max,
                    amount0Min: 0,
                    amount1Min: 0,
                    deadline: block.timestamp
                })
            )
        {} catch {}

        try
            positionManager.collect(
                INonfungiblePositionManager.CollectParams({
                    tokenId: _tokenId,
                    recipient: address(this),
                    amount0Max: type(uint128).max,
                    amount1Max: type(uint128).max
                })
            )
        {} catch {}

        try positionManager.burn(_tokenId) {} catch {}
    }

    // --- Internal function to create position ---
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

        if (amount0Desired == 0 && amount1Desired == 0) {
            return;
        }

        try
            positionManager.mint(
                INonfungiblePositionManager.MintParams({
                    token0: token0,
                    token1: token1,
                    fee: fee,
                    tickLower: _newLowerTick,
                    tickUpper: _newUpperTick,
                    amount0Desired: amount0Desired,
                    amount1Desired: amount1Desired,
                    amount0Min: 0,
                    amount1Min: 0,
                    recipient: address(this),
                    deadline: block.timestamp
                })
            )
        returns (
            uint256 _tokenId,
            uint128 _liquidity,
            uint256 _amount0Actual,
            uint256 _amount1Actual
        ) {
            if (_liquidity > 0) {
                currentTokenId = _tokenId;
                currentLiquidity = _liquidity;
                hasPosition = true;
                lowerTick = _newLowerTick;
                upperTick = _newUpperTick;

                emit PositionStateChanged(
                    _tokenId,
                    true,
                    _newLowerTick,
                    _newUpperTick,
                    _liquidity
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
                if (_tokenId > 0) {
                    try positionManager.burn(_tokenId) {} catch {}
                }
                hasPosition = false;
                currentTokenId = 0;
                currentLiquidity = 0;
                lowerTick = 0;
                upperTick = 0;
            }
        } catch {
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
        IWETH(token1).withdraw(amount);
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
