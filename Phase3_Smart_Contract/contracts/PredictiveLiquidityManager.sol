// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;
pragma abicoder v2; // To support structs in parameters

// OpenZeppelin ~3.4.0 Imports
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

// Uniswap V3 Core
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3MintCallback.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";

// Uniswap V3 Periphery
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";

// Interface for decimals()
interface IERC20Decimals {
    function decimals() external view returns (uint8);
}

/**
 * @title PredictiveLiquidityManager
 * @notice The main liquidity management contract that adjusts positions based on price predictions
 */
contract PredictiveLiquidityManager is
    Ownable,
    ReentrancyGuard,
    IUniswapV3MintCallback
{
    using SafeERC20 for IERC20;

    // --- State Variables ---
    IUniswapV3Factory public immutable factory;
    INonfungiblePositionManager public immutable positionManager;
    address public immutable token0;
    address public immutable token1;
    uint8 public immutable token0Decimals;
    uint8 public immutable token1Decimals;
    uint24 public immutable fee;
    int24 public immutable tickSpacing;

    // Structure for liquidity positions
    struct Position {
        uint256 tokenId;
        uint128 liquidity;
        int24 tickLower;
        int24 tickUpper;
        bool active;
    }
    Position public currentPosition;

    // Strategy Parameters
    uint24 public rangeWidthMultiplier;

    // --- Events ---
    // Event for liquidity operations
    event LiquidityOperation(
        string operationType,
        uint256 indexed tokenId,
        int24 tickLower,
        int24 tickUpper,
        uint128 liquidity,
        uint256 amount0,
        uint256 amount1,
        bool success
    );

    // Event for outputting main adjustment logic
    event PredictionAdjustmentMetrics(
        int24 predictedTick,
        int24 finalTickLower,
        int24 finalTickUpper,
        uint128 liquidity,
        bool adjusted
    );

    event StrategyParamUpdated(string indexed paramName, uint256 newValue);

    // --- Constructor ---
    constructor(
        address _factory,
        address _positionManager,
        address _token0,
        address _token1,
        uint24 _fee,
        address _initialOwner,
        uint24 _initialRangeWidthMultiplier
    ) {
        // Store values in immutable variables
        factory = IUniswapV3Factory(_factory);
        positionManager = INonfungiblePositionManager(_positionManager);
        token0 = _token0;
        token1 = _token1;
        fee = _fee;
        require(_initialRangeWidthMultiplier > 0, "Initial RWM must be > 0");
        rangeWidthMultiplier = _initialRangeWidthMultiplier;

        // Check decimals using try-catch
        try IERC20Decimals(_token0).decimals() returns (uint8 _decimals) {
            token0Decimals = _decimals;
        } catch {
            revert("Token0 does not support decimals()");
        }

        try IERC20Decimals(_token1).decimals() returns (uint8 _decimals) {
            token1Decimals = _decimals;
        } catch {
            revert("Token1 does not support decimals()");
        }

        // Store tickSpacing in a temporary variable
        address poolAddress = IUniswapV3Factory(_factory).getPool(
            _token0,
            _token1,
            _fee
        );
        require(poolAddress != address(0), "Pool does not exist");
        tickSpacing = IUniswapV3Pool(poolAddress).tickSpacing();

        // Set approvals for token0 and token1
        IERC20(_token0).safeApprove(
            address(_positionManager),
            type(uint256).max
        );
        IERC20(_token1).safeApprove(
            address(_positionManager),
            type(uint256).max
        );

        if (_initialOwner != address(0)) {
            transferOwnership(_initialOwner);
        }
    }
    
     // --- Function to set Range Width Multiplier ---
     function setRangeWidthMultiplier(uint24 _newMultiplier) external onlyOwner {
        require(_newMultiplier > 0, "RWM must be > 0");
         // You can add a check for a maximum value if necessary
        rangeWidthMultiplier = _newMultiplier;
        emit StrategyParamUpdated("rangeWidthMultiplier", uint256(_newMultiplier));
     }

    // --- Automated Liquidity Management (Owner Only) ---
    function updatePredictionAndAdjust(
        int24 predictedTick
    ) external nonReentrant onlyOwner {
        (int24 targetTickLower, int24 targetTickUpper) = _calculateTicks(
            predictedTick
        );

        bool adjusted = false;
        if (
            currentPosition.active &&
            _isTickRangeClose(
                currentPosition.tickLower,
                currentPosition.tickUpper,
                targetTickLower,
                targetTickUpper
            )
        ) {
            _emitPredictionMetrics(
                predictedTick,
                targetTickLower,
                targetTickUpper,
                false
            );
            return;
        }

        adjusted = _updatePositionIfNeeded(targetTickLower, targetTickUpper);

        _emitPredictionMetrics(
            predictedTick,
            targetTickLower,
            targetTickUpper,
            adjusted
        );
    }

    function _calculateTicks(
        int24 targetCenterTick
    ) internal view returns (int24 tickLower, int24 tickUpper) {
        require(tickSpacing > 0, "Invalid tick spacing");

        int24 halfWidth = (tickSpacing * int24(rangeWidthMultiplier)) / 2;
        if (halfWidth <= 0) halfWidth = tickSpacing;

        halfWidth = (halfWidth / tickSpacing) * tickSpacing;
        if (halfWidth == 0) halfWidth = tickSpacing;

        int24 rawTickLower = targetCenterTick - halfWidth;
        int24 rawTickUpper = targetCenterTick + halfWidth;

        tickLower = floorToTickSpacing(rawTickLower, tickSpacing);
        tickUpper = floorToTickSpacing(rawTickUpper, tickSpacing);

        if ((rawTickUpper % tickSpacing) != 0) {
            tickUpper += tickSpacing;
        }

        if (tickLower >= tickUpper) {
            tickUpper = tickLower + tickSpacing;
        }

        tickLower = tickLower < TickMath.MIN_TICK
            ? floorToTickSpacing(TickMath.MIN_TICK, tickSpacing)
            : tickLower;

        tickUpper = tickUpper > TickMath.MAX_TICK
            ? floorToTickSpacing(TickMath.MAX_TICK, tickSpacing)
            : tickUpper;

        if (tickLower >= tickUpper) {
            tickUpper = tickLower + tickSpacing;

            if (tickUpper > TickMath.MAX_TICK) {
                tickUpper = floorToTickSpacing(TickMath.MAX_TICK, tickSpacing);
                tickLower = tickUpper - tickSpacing;
            }
        }

        return (tickLower, tickUpper);
    }

    function _isTickRangeClose(
        int24 oldLower,
        int24 oldUpper,
        int24 newLower,
        int24 newUpper
    ) internal view returns (bool) {
        int24 minDiff = (tickSpacing * int24(rangeWidthMultiplier)) / 2;
        return (_abs(oldLower - newLower) < minDiff &&
            _abs(oldUpper - newUpper) < minDiff);
    }

    function _abs(int24 x) internal pure returns (int24) {
        return x >= 0 ? x : -x;
    }

    function _updatePositionIfNeeded(
        int24 targetTickLower,
        int24 targetTickUpper
    ) internal returns (bool adjusted) {
        if (
            !currentPosition.active ||
            targetTickLower != currentPosition.tickLower ||
            targetTickUpper != currentPosition.tickUpper
        ) {
            _adjustLiquidity(targetTickLower, targetTickUpper);
            return true;
        }
        return false;
    }

    // --- Internal Liquidity Management Helpers ---
    function _adjustLiquidity(int24 tickLower, int24 tickUpper) internal {
        if (currentPosition.active) {
            _removeLiquidity();
        }
        uint256 balance0 = IERC20(token0).balanceOf(address(this));
        uint256 balance1 = IERC20(token1).balanceOf(address(this));
        if (balance0 > 0 || balance1 > 0) {
            _mintLiquidity(tickLower, tickUpper, balance0, balance1);
        } else {
            currentPosition = Position(0, 0, 0, 0, false);
        }
    }

    function _removeLiquidity() internal {
        require(
            currentPosition.active && currentPosition.tokenId != 0,
            "No active position"
        );
        uint256 _tokenId = currentPosition.tokenId;
        uint128 _liquidity = currentPosition.liquidity;
        int24 _tickLower = currentPosition.tickLower;
        int24 _tickUpper = currentPosition.tickUpper;

        currentPosition = Position(0, 0, 0, 0, false);

        bool decreaseSuccess = false;
        bool collectSuccess = false;
        uint256 amount0Collected = 0;
        uint256 amount1Collected = 0;

        if (_liquidity > 0) {
            try
                positionManager.decreaseLiquidity(
                    INonfungiblePositionManager.DecreaseLiquidityParams({
                        tokenId: _tokenId,
                        liquidity: _liquidity,
                        amount0Min: 0,
                        amount1Min: 0,
                        deadline: block.timestamp
                    })
                )
            {
                decreaseSuccess = true;
            } catch {}
        } else {
            decreaseSuccess = true;
        }

        if (decreaseSuccess) {
            try
                positionManager.collect(
                    INonfungiblePositionManager.CollectParams({
                        tokenId: _tokenId,
                        recipient: address(this),
                        amount0Max: type(uint128).max,
                        amount1Max: type(uint128).max
                    })
                )
            returns (uint256 a0, uint256 a1) {
                amount0Collected = a0;
                amount1Collected = a1;
                collectSuccess = true;
            } catch {}
            try positionManager.burn(_tokenId) {} catch {}
        }
        bool overallSuccess = decreaseSuccess && collectSuccess;
        emit LiquidityOperation(
            "REMOVE",
            _tokenId,
            _tickLower,
            _tickUpper,
            _liquidity,
            amount0Collected,
            amount1Collected,
            overallSuccess
        );
    }

    function _mintLiquidity(
        int24 tickLower,
        int24 tickUpper,
        uint256 amount0Desired,
        uint256 amount1Desired
    ) internal {
        require(!currentPosition.active, "Position already active");
        INonfungiblePositionManager.MintParams
            memory params = INonfungiblePositionManager.MintParams({
                token0: token0,
                token1: token1,
                fee: fee,
                tickLower: tickLower,
                tickUpper: tickUpper,
                amount0Desired: amount0Desired,
                amount1Desired: amount1Desired,
                amount0Min: 0,
                amount1Min: 0,
                recipient: address(this),
                deadline: block.timestamp
            });
        uint256 tokenId = 0;
        uint128 liquidity = 0;
        uint256 amount0Actual = 0;
        uint256 amount1Actual = 0;
        bool success = false;
        try positionManager.mint(params) returns (
            uint256 _tokenId,
            uint128 _liquidity,
            uint256 _amount0,
            uint256 _amount1
        ) {
            tokenId = _tokenId;
            liquidity = _liquidity;
            amount0Actual = _amount0;
            amount1Actual = _amount1;
            if (liquidity > 0) {
                currentPosition = Position(
                    tokenId,
                    liquidity,
                    tickLower,
                    tickUpper,
                    true
                );
                success = true;
            } else if (tokenId != 0) {
                try positionManager.burn(tokenId) {} catch {}
            }
        } catch {}
        emit LiquidityOperation(
            "MINT",
            tokenId,
            tickLower,
            tickUpper,
            liquidity,
            amount0Actual,
            amount1Actual,
            success
        );
        if (!success) {
            currentPosition = Position(0, 0, 0, 0, false);
        }
    }

    function floorToTickSpacing(
        int24 tick,
        int24 _tickSpacing
    ) internal pure returns (int24) {
        require(_tickSpacing > 0, "Tick spacing must be positive");
        int24 compressed = tick / _tickSpacing;
        if (tick < 0 && (tick % _tickSpacing != 0)) {
            compressed--;
        }
        return compressed * _tickSpacing;
    }

    function _emitPredictionMetrics(
        int24 predictedTick,
        int24 finalTickLower,
        int24 finalTickUpper,
        bool adjusted
    ) internal {
        uint128 liquidity = currentPosition.active
            ? currentPosition.liquidity
            : 0;
        emit PredictionAdjustmentMetrics(
            predictedTick,
            finalTickLower,
            finalTickUpper,
            liquidity,
            adjusted
        );
    }

    function uniswapV3MintCallback(
        uint256 amount0Owed,
        uint256 amount1Owed,
        bytes calldata data
    ) external override {
        require(
            msg.sender == address(positionManager),
            "Unauthorized callback"
        );

        if (amount0Owed > 0) {
            IERC20(token0).safeTransfer(msg.sender, amount0Owed);
        }
        if (amount1Owed > 0) {
            IERC20(token1).safeTransfer(msg.sender, amount1Owed);
        }
    }
}
