"""
VoxelVM Market Data API — High-performance tick analysis

Usage:
    import voxel_market as vm
    data = vm.MarketData(prices=np.array([...]), volumes=np.array([...]))
    vwap = data.vwap()
    bb = data.bollinger_bands(window=20, num_std=2)
    vol = data.volatility(window=20)
    ohlc = data.resample_ohlc(window=100)
"""
import numpy as np
import voxel_py as _vx
from voxel_query import Query, from_numpy


class MarketData:
    """Columnar market data with VoxelVM-accelerated analytics."""

    def __init__(self, prices=None, volumes=None, timestamps=None,
                 highs=None, lows=None, opens=None, closes=None):
        self._prices = np.asarray(prices, dtype=np.float64) if prices is not None else None
        self._volumes = np.asarray(volumes, dtype=np.float64) if volumes is not None else None
        self._timestamps = np.asarray(timestamps, dtype=np.int64) if timestamps is not None else None
        self._highs = np.asarray(highs, dtype=np.float64) if highs is not None else None
        self._lows = np.asarray(lows, dtype=np.float64) if lows is not None else None
        self._opens = np.asarray(opens, dtype=np.float64) if opens is not None else None
        self._closes = np.asarray(closes, dtype=np.float64) if closes is not None else None

        self._engine = _vx.EngineF64()
        self._segs = {}

    def _get_seg(self, arr, name):
        if name not in self._segs:
            self._segs[name] = self._engine.add_segment(arr)
        return self._segs[name]

    def _run_agg(self, data, threshold=None, op="sum"):
        """Run a filter+aggregate on a column. Uses VoxelVM engine."""
        if data is None:
            data = self._prices
        if data is None:
            raise ValueError("No data available")
        q = from_numpy(data)
        if threshold is not None:
            q.filter_gt(threshold)
        if op == "sum":
            q.sum()
        elif op == "count":
            q.count()
        elif op == "min":
            q.min()
        elif op == "max":
            q.max()
        return q.run()

    # ================================================================
    # Basic analytics
    # ================================================================

    def sum(self, data=None):
        """Sum of all values."""
        return self._run_agg(data if data is not None else self._prices, op="sum")

    def mean(self, data=None):
        """Arithmetic mean."""
        d = data if data is not None else self._prices
        return self.sum(d) / len(d)

    def count_above(self, threshold, data=None):
        """Count values above threshold."""
        return int(self._run_agg(data or self._prices, threshold=threshold, op="count"))

    def sum_above(self, threshold, data=None):
        """Sum of values above threshold."""
        return self._run_agg(data or self._prices, threshold=threshold, op="sum")

    # ================================================================
    # Market-specific analytics
    # ================================================================

    def vwap(self):
        """Volume-Weighted Average Price."""
        if self._prices is None or self._volumes is None:
            raise ValueError("prices and volumes required for VWAP")
        notional = np.multiply(self._prices, self._volumes)
        total_notional = self.sum(notional)
        total_volume = self.sum(self._volumes)
        return total_notional / total_volume if total_volume > 0 else 0.0

    def returns(self, period=1):
        """Simple returns over given period."""
        if self._prices is None:
            raise ValueError("prices required")
        p = self._prices
        return (p[period:] - p[:-period]) / p[:-period]

    def log_returns(self, period=1):
        """Logarithmic returns."""
        p = self._prices
        return np.log(p[period:] / p[:-period])

    def volatility(self, window=20):
        """Rolling historical volatility (annualized) — VoxelVM-accelerated."""
        r = self.returns()
        if len(r) < window:
            return np.nan
        n_out = len(r) - window + 1
        stddev = np.zeros(n_out, dtype=np.float64)
        e = _vx.EngineF64()
        e.window_stddev(r, stddev, window)
        return np.mean(stddev) * np.sqrt(365 * 24)  # annualize for hourly data

    def rolling_sum(self, window, data=None):
        """Rolling window sum."""
        d = data if data is not None else self._prices
        return np.convolve(d, np.ones(window), mode='valid')

    def rolling_mean(self, window, data=None):
        """Rolling window average."""
        s = self.rolling_sum(window, data)
        return s / window

    def rolling_max(self, window, data=None):
        """Rolling window maximum."""
        d = data if data is not None else self._prices
        return np.array([np.max(d[i:i+window]) for i in range(len(d) - window + 1)])

    def rolling_min(self, window, data=None):
        """Rolling window minimum."""
        d = data if data is not None else self._prices
        return np.array([np.min(d[i:i+window]) for i in range(len(d) - window + 1)])

    def bollinger_bands(self, window=20, num_std=2.0, data=None):
        """Bollinger Bands — VoxelVM-accelerated via WINDOW_MEAN + WINDOW_STDDEV."""
        d = data if data is not None else self._prices
        if d is None:
            raise ValueError("prices required")
        if len(d) < window:
            return {"middle": np.array([]), "upper": np.array([]), "lower": np.array([])}
        n_out = len(d) - window + 1
        middle = np.zeros(n_out, dtype=np.float64)
        stddev = np.zeros(n_out, dtype=np.float64)
        e = _vx.EngineF64()
        e.window_mean(d, middle, window)
        e.window_stddev(d, stddev, window)
        upper = middle + num_std * stddev
        lower = middle - num_std * stddev
        return {"middle": middle, "upper": upper, "lower": lower}

    def rsi(self, window=14, data=None):
        """Relative Strength Index — VoxelVM-accelerated via window-streaming ops.
        
        Falls back to NumPy for small inputs (< 2*window).
        """
        d = data if data is not None else self._prices
        if d is None:
            raise ValueError("prices required")
        N = len(d)
        n_out = N - window
        if n_out < 1:
            return np.array([])

        # Use NumPy for tiny datasets (the engine overhead isn't worth it)
        if N < 1000:
            delta = np.diff(d)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
            avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
            rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
            return 100.0 - (100.0 / (1.0 + rs))

        e = _vx.EngineF64()

        # Step 1: window_delta — compute price differences
        deltas = np.zeros(N, dtype=np.float64)
        e.window_delta(d, deltas)

        # Step 2: separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Step 3: window_mean on gains and losses (first value = avg of first window)
        avg_gain = np.zeros(n_out, dtype=np.float64)
        avg_loss = np.zeros(n_out, dtype=np.float64)
        e.window_mean(gains, avg_gain, window)
        e.window_mean(losses, avg_loss, window)

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        return 100.0 - (100.0 / (1.0 + rs))

    def ema(self, span=20, data=None):
        """Exponential Moving Average."""
        d = data if data is not None else self._prices
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(d)
        result[0] = d[0]
        for i in range(1, len(d)):
            result[i] = alpha * d[i] + (1 - alpha) * result[i-1]
        return result

    def macd(self, fast=12, slow=26, signal=9, data=None):
        """MACD indicator."""
        d = data if data is not None else self._prices
        ema_fast = self.ema(fast, d)
        ema_slow = self.ema(slow, d)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(signal, macd_line)
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def time_ohlc(self, timestamps, duration_sec=3600):
        """Resample tick data into time-indexed OHLC bars using VoxelVM.

        Args:
            timestamps: i64 array of unix timestamps (seconds)
            duration_sec: bar duration in seconds (default 3600 = 1 hour)
        """
        if self._prices is None or timestamps is None:
            raise ValueError("prices and timestamps required")
        ts = np.asarray(timestamps, dtype=np.int64)
        p = self._prices
        N = len(p)
        if N == 0:
            return MarketData()
        start = ts[0]
        max_buckets = int((ts[-1] - start) / duration_sec) + 1
        e = _vx.EngineF64()

        # Bucketed volume sum
        volumes = None
        if self._volumes is not None:
            volumes = np.zeros(max_buckets, dtype=np.float64)
            e.time_window_sum(ts, self._volumes, start, duration_sec, volumes)

        opens = np.zeros(max_buckets, dtype=np.float64)
        highs  = np.zeros(max_buckets, dtype=np.float64)
        lows   = np.full(max_buckets, np.inf, dtype=np.float64)
        closes = np.zeros(max_buckets, dtype=np.float64)

        last_bucket = -1
        for i in range(N):
            b = int((ts[i] - start) // duration_sec)
            if b != last_bucket:
                if 0 <= b < max_buckets:
                    opens[b] = p[i]
                    if last_bucket >= 0 and last_bucket < max_buckets:
                        closes[last_bucket] = p[i - 1]
                last_bucket = b
            if 0 <= b < max_buckets:
                highs[b] = max(highs[b], p[i])
                lows[b] = min(lows[b], p[i])
        if last_bucket >= 0 and last_bucket < max_buckets:
            closes[last_bucket] = p[-1]
        for i in range(max_buckets):
            if np.isinf(lows[i]): lows[i] = opens[i] if opens[i] > 0 else 0.0
            if highs[i] == 0 and opens[i] > 0: highs[i] = opens[i]

        return MarketData(opens=opens, highs=highs, lows=lows, closes=closes,
                          volumes=volumes, prices=closes,
                          timestamps=np.arange(max_buckets) * duration_sec + start)

    def resample_ohlc(self, window=100):
        """Resample tick data into OHLC bars."""
        if self._prices is None:
            raise ValueError("prices required")
        p = self._prices
        n = len(p) // window
        opens = np.array([p[i*window] for i in range(n)])
        highs = np.array([np.max(p[i*window:(i+1)*window]) for i in range(n)])
        lows = np.array([np.min(p[i*window:(i+1)*window]) for i in range(n)])
        closes = np.array([p[min((i+1)*window - 1, len(p)-1)] for i in range(n)])
        volumes = None
        if self._volumes is not None:
            v = self._volumes
            volumes = np.array([np.sum(v[i*window:(i+1)*window]) for i in range(n)])
        return MarketData(opens=opens, highs=highs, lows=lows, closes=closes,
                         volumes=volumes, prices=closes)

    def sharpe_ratio(self, risk_free=0.02):
        """Annualized Sharpe ratio."""
        r = self.returns()
        excess = r - risk_free / (365 * 24)
        return np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(365 * 24)

    def var(self, window=250, confidence=0.95):
        """Value at Risk — historical method, VoxelVM-accelerated quantile."""
        r = self.returns()
        if len(r) < window:
            return np.nan
        n_out = len(r) - window + 1
        var_series = np.zeros(n_out, dtype=np.float64)
        e = _vx.EngineF64()
        e.window_quantile(r, var_series, window, 1.0 - confidence)
        return var_series

    def max_drawdown(self, data=None):
        """Maximum drawdown from peak."""
        d = data if data is not None else self._prices
        peak = np.maximum.accumulate(d)
        drawdown = (d - peak) / peak
        return np.min(drawdown)

    def sortino_ratio(self, risk_free=0.02):
        """Sortino ratio (downside-only risk)."""
        r = self.returns()
        downside = r[r < 0]
        if len(downside) == 0:
            return 0.0
        excess = np.mean(r) - risk_free / (365 * 24)
        return excess / (np.std(downside) + 1e-10) * np.sqrt(365 * 24)

    def correlation(self, other_prices):
        """Pearson correlation with another price series."""
        if self._prices is None:
            raise ValueError("prices required")
        p1 = self.returns()
        p2 = MarketData(prices=other_prices).returns()
        min_len = min(len(p1), len(p2))
        return np.corrcoef(p1[:min_len], p2[:min_len])[0, 1]

    def beta(self, market_prices):
        """Beta coefficient vs market."""
        r_stock = self.returns()
        r_market = MarketData(prices=market_prices).returns()
        min_len = min(len(r_stock), len(r_market))
        cov = np.cov(r_stock[:min_len], r_market[:min_len])[0, 1]
        var = np.var(r_market[:min_len])
        return cov / var if var > 0 else 0.0

    # ================================================================
    # Batch analytics (process full dataset at once)
    # ================================================================

    def stats(self):
        """Full statistical summary."""
        p = self._prices
        if p is None:
            return {}
        r = self.returns()
        return {
            "count": len(p),
            "sum": self.sum(),
            "mean": self.mean(),
            "min": np.min(p),
            "max": np.max(p),
            "std": np.std(p),
            "vwap": self.vwap() if self._volumes is not None else None,
            "volatility": self.volatility(),
            "sharpe": self.sharpe_ratio(),
            "sortino": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "skew": float(np.mean((r - np.mean(r))**3) / (np.std(r)**3 + 1e-10)) if len(r) > 2 else 0,
            "kurtosis": float(np.mean((r - np.mean(r))**4) / (np.std(r)**4 + 1e-10)) if len(r) > 3 else 0,
        }

    def __len__(self):
        if self._prices is not None:
            return len(self._prices)
        if self._volumes is not None:
            return len(self._volumes)
        if self._highs is not None:
            return len(self._highs)
        return 0


# ================================================================
# Quick one-shot functions
# ================================================================

def vwap(prices, volumes):
    return MarketData(prices=prices, volumes=volumes).vwap()

def bollinger(prices, window=20, num_std=2.0):
    return MarketData(prices=prices).bollinger_bands(window, num_std)

def rsi(prices, window=14):
    return MarketData(prices=prices).rsi(window)

def macd(prices, fast=12, slow=26, signal=9):
    return MarketData(prices=prices).macd(fast, slow, signal)

def ohlc(prices, window=100, volumes=None):
    return MarketData(prices=prices, volumes=volumes).resample_ohlc(window)

def volatility(prices, window=20):
    return MarketData(prices=prices).volatility(window)

def sharpe(prices):
    return MarketData(prices=prices).sharpe_ratio()

def max_dd(prices):
    return MarketData(prices=prices).max_drawdown()
