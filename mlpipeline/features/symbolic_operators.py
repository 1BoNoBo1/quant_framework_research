"""
Opérateurs Symboliques inspirés du papier:
"Synergistic Formulaic Alpha Generation for Quantitative Trading"

Implémentation des opérateurs avancés pour enrichir nos alphas
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class SymbolicOperators:
    """
    Collection d'opérateurs symboliques pour génération de features avancées
    Basé sur le papier de recherche et Alpha101
    """

    @staticmethod
    def sign(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Returns 0 if the given x value is 0, 1 if it is positive, and -1 if it is negative.
        """
        return np.sign(x)

    @staticmethod
    def cs_rank(x: pd.Series) -> pd.Series:
        """
        Cross-sectional rank (CSRank) - returns the rank of the current
        stock's feature value x relative to the feature values of all stocks on today's date.

        Pour un seul asset, on utilise rolling rank
        """
        if isinstance(x, pd.Series):
            # Rolling cross-sectional rank simulation
            return x.rank(pct=True)  # Percentile rank
        return x

    @staticmethod
    def product(x: pd.Series, t: int) -> pd.Series:
        """
        Returns the product of the feature values for each date from the current date up to t days ago.
        Product(x, t) = ∏(i=0 to t) x_{t-i}
        """
        return x.rolling(window=t+1, min_periods=1).apply(lambda vals: np.prod(vals), raw=True)

    @staticmethod
    def scale(x: pd.Series) -> pd.Series:
        """
        Returns the value obtained by dividing the current feature value x
        by the total sum of the absolute values of the feature.
        Scale(x) = x / Σ|x_i|
        """
        abs_sum = np.abs(x).sum()
        if abs_sum == 0:
            return pd.Series(0, index=x.index)
        return x / abs_sum

    @staticmethod
    def pow_op(x: pd.Series, y: Union[float, pd.Series]) -> pd.Series:
        """
        Pow(x,y) = x^y
        """
        try:
            # Éviter les overflow et problèmes numériques
            x_safe = np.clip(x, -1e6, 1e6)
            if isinstance(y, (int, float)):
                y_safe = np.clip(y, -10, 10)
            else:
                y_safe = np.clip(y, -10, 10)

            result = np.power(np.abs(x_safe), y_safe) * np.sign(x_safe)
            result = np.where(np.isfinite(result), result, 0)
            return pd.Series(result, index=x.index)
        except (OverflowError, ZeroDivisionError, ValueError):
            return pd.Series(0, index=x.index)

    @staticmethod
    def skew(x: pd.Series, window: int = 20) -> pd.Series:
        """
        Skewness - represents the asymmetry of a data distribution
        using the third standard moment.
        """
        return x.rolling(window=window, min_periods=3).skew()

    @staticmethod
    def kurt(x: pd.Series, window: int = 20) -> pd.Series:
        """
        Kurtosis - value indicating the peakedness of a data distribution
        Kurt(x) = μ₄/μ₂² - 3
        """
        return x.rolling(window=window, min_periods=4).kurt()

    @staticmethod
    def ts_rank(x: pd.Series, t: int) -> pd.Series:
        """
        Time-series rank - returns the rank of the current feature value x
        among feature values from the current date up to t days ago.
        """
        def rank_last(values):
            if len(values) < 2:
                return 0.5
            return (values < values.iloc[-1]).sum() / (len(values) - 1)

        return x.rolling(window=t, min_periods=1).apply(rank_last, raw=False)

    @staticmethod
    def delta(x: pd.Series, t: int) -> pd.Series:
        """
        Returns the difference between the current feature value x
        and the feature value from t days ago.
        Delta(x, t) = x - Ref(x, t)
        """
        return x - x.shift(t)

    @staticmethod
    def argmax(x: pd.Series, t: int) -> pd.Series:
        """
        Returns the date when the feature value x was the highest
        within the period from the current date up to t days ago.
        """
        def get_argmax_idx(values):
            if len(values) == 0:
                return 0
            return len(values) - 1 - np.argmax(values[::-1])

        return x.rolling(window=t, min_periods=1).apply(get_argmax_idx, raw=True)

    @staticmethod
    def argmin(x: pd.Series, t: int) -> pd.Series:
        """
        Returns the date when the feature value x was the lowest
        within the period from the current date up to t days ago.
        """
        def get_argmin_idx(values):
            if len(values) == 0:
                return 0
            return len(values) - 1 - np.argmin(values[::-1])

        return x.rolling(window=t, min_periods=1).apply(get_argmin_idx, raw=True)

    @staticmethod
    def cond(x: pd.Series, y: Union[pd.Series, float], t_val: float, f_val: float) -> pd.Series:
        """
        Conditional operator - returns t_val if x > y is true, and f_val if it is false.
        Cond(x,y,t,f)
        """
        try:
            condition = x > y
            result = np.where(condition, t_val, f_val)
            return pd.Series(result, index=x.index)
        except Exception:
            return pd.Series(f_val, index=x.index)

    @staticmethod
    def wma(x: pd.Series, window: int) -> pd.Series:
        """
        Weighted Moving Average
        """
        def weighted_mean(values):
            weights = np.arange(1, len(values) + 1)
            return np.average(values, weights=weights)

        return x.rolling(window=window, min_periods=1).apply(weighted_mean, raw=True)

    @staticmethod
    def ema(x: pd.Series, window: int) -> pd.Series:
        """
        Exponential Moving Average
        """
        return x.ewm(span=window, adjust=False).mean()

    @staticmethod
    def mad(x: pd.Series, window: int) -> pd.Series:
        """
        Mean Absolute Deviation
        """
        return x.rolling(window=window).apply(lambda vals: np.mean(np.abs(vals - np.mean(vals))))

class AlphaFormulaGenerator:
    """
    Générateur de formules alpha utilisant les opérateurs symboliques
    """

    def __init__(self):
        self.ops = SymbolicOperators()

    def generate_alpha_006(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha #006 du papier: (-1 * Corr(open, volume, 10))
        """
        corr_values = data['open'].rolling(10).corr(data['volume'])
        return -1 * corr_values

    def generate_alpha_099(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha #099: Less(CSRank(Corr(Sum(((high + low) / 2), 19.8975),
        Sum(Mean(volume, 60), 19.8975), 8.8136)), CSRank(Corr(low, volume, 6.28259))) * -1
        """
        try:
            # Simplification de la formule complexe avec sécurité
            hl_mid = (data['high'] + data['low']) / 2
            vol_mean = data['volume'].rolling(60, min_periods=10).mean()

            # Calculs avec gestion des NaN
            sum1 = hl_mid.rolling(20, min_periods=5).sum()
            sum2 = vol_mean.rolling(20, min_periods=5).sum()

            corr1 = sum1.rolling(8, min_periods=3).corr(sum2)
            corr2 = data['low'].rolling(6, min_periods=3).corr(data['volume'])

            # Remplacer NaN par 0 avant ranking
            corr1_clean = corr1.fillna(0)
            corr2_clean = corr2.fillna(0)

            rank1 = self.ops.cs_rank(corr1_clean)
            rank2 = self.ops.cs_rank(corr2_clean)

            # Comparaison sécurisée
            result = pd.Series(np.where(rank1 < rank2, -1, 0), index=data.index)
            return result.fillna(0)

        except Exception as e:
            logger.warning(f"Erreur Alpha 099: {e}")
            return pd.Series(0, index=data.index)

    def generate_alpha_061(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha #061: Less(CSRank((vwap - Min(vwap, 16.1219))),
        CSRank(Corr(vwap, Mean(volume, 180), 17.9282)))
        """
        try:
            vwap = data.get('vwap', (data['high'] + data['low'] + data['close']) / 3)

            min_vwap = vwap.rolling(16, min_periods=5).min()
            vol_mean = data['volume'].rolling(180, min_periods=20).mean()
            corr_vwap_vol = vwap.rolling(18, min_periods=5).corr(vol_mean)

            # Nettoyage avant ranking
            diff_clean = (vwap - min_vwap).fillna(0)
            corr_clean = corr_vwap_vol.fillna(0)

            rank1 = self.ops.cs_rank(diff_clean)
            rank2 = self.ops.cs_rank(corr_clean)

            # Comparaison sécurisée
            result = pd.Series(np.where(rank1 < rank2, 1, 0), index=data.index)
            return result.fillna(0)

        except Exception as e:
            logger.warning(f"Erreur Alpha 061: {e}")
            return pd.Series(0, index=data.index)

    def generate_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère un ensemble de features avancées utilisant les opérateurs symboliques
        """
        features = pd.DataFrame(index=data.index)

        try:
            # Features de base
            close = data['close']
            volume = data['volume']
            high = data['high']
            low = data['low']
            open_price = data['open']

            # Opérateurs symboliques appliqués
            features['sign_returns'] = self.ops.sign(close.pct_change())
            features['cs_rank_volume'] = self.ops.cs_rank(volume)
            features['product_close_5'] = self.ops.product(close, 5)
            features['scale_volume'] = self.ops.scale(volume)
            features['skew_returns_20'] = self.ops.skew(close.pct_change(), 20)
            features['kurt_returns_20'] = self.ops.kurt(close.pct_change(), 20)

            # Time-series features
            features['ts_rank_close_10'] = self.ops.ts_rank(close, 10)
            features['delta_close_5'] = self.ops.delta(close, 5)
            features['argmax_high_20'] = self.ops.argmax(high, 20)
            features['argmin_low_20'] = self.ops.argmin(low, 20)

            # Moving averages avancées
            features['wma_close_10'] = self.ops.wma(close, 10)
            features['ema_volume_20'] = self.ops.ema(volume, 20)
            features['mad_close_15'] = self.ops.mad(close, 15)

            # Conditional features
            features['cond_high_low'] = self.ops.cond(high, low, 1.0, -1.0)

            # Combinaisons avancées
            features['pow_volume_half'] = self.ops.pow_op(volume, 0.5)

            # Alpha formulas
            features['alpha_006'] = self.generate_alpha_006(data)
            features['alpha_061'] = self.generate_alpha_061(data)
            features['alpha_099'] = self.generate_alpha_099(data)

            # Nettoyage des NaN et infinis
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().fillna(0)

            logger.info(f"✅ Génération de {len(features.columns)} features symboliques")
            return features

        except Exception as e:
            logger.error(f"❌ Erreur génération features symboliques: {e}")
            return pd.DataFrame(index=data.index)

def test_symbolic_operators():
    """Test des opérateurs symboliques"""
    print("🧪 Test des opérateurs symboliques...")

    # Données de test
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.02) + 1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.02) - 1,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Test générateur
    generator = AlphaFormulaGenerator()
    features = generator.generate_enhanced_features(data)

    print(f"✅ Features générées: {list(features.columns)}")
    print(f"✅ Shape: {features.shape}")
    print(f"✅ NaN count: {features.isnull().sum().sum()}")

    return features

if __name__ == "__main__":
    test_symbolic_operators()