"""
Factor Analyzer
==============

Advanced factor analysis for portfolio and strategy decomposition.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import logging

from ...core.container import injectable
from ...core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class FactorModel(str, Enum):
    """Types de modèles de facteurs"""
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    CAPM = "capm"
    PCA = "pca"
    CUSTOM = "custom"
    MACRO_ECONOMIC = "macro_economic"


@dataclass
class FactorExposure:
    """Exposition à un facteur"""
    factor_name: str
    exposure: float
    t_statistic: float
    p_value: float
    contribution_to_return: float
    contribution_to_risk: float


@dataclass
class FactorAnalysisResult:
    """Résultats de l'analyse factorielle"""
    model_type: FactorModel
    analysis_period: Tuple[datetime, datetime]

    # Expositions aux facteurs
    factor_exposures: List[FactorExposure]

    # Statistiques du modèle
    r_squared: float
    adjusted_r_squared: float
    tracking_error: float
    information_ratio: float

    # Décomposition des returns
    factor_returns: pd.DataFrame
    residual_returns: pd.Series
    explained_variance: float

    # Alpha et sélectivité
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float


@injectable
class FactorAnalyzer:
    """
    Analyseur de facteurs avancé.

    Capacités:
    - Modèles Fama-French 3 et 5 facteurs
    - Analyse en composantes principales (PCA)
    - Facteurs macroéconomiques custom
    - Attribution de performance par facteur
    - Tests de significativité statistique
    - Analyse de style drift
    """

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        default_model: FactorModel = FactorModel.FAMA_FRENCH_3,
        confidence_level: float = 0.95
    ):
        self.config = config
        self.default_model = default_model
        self.confidence_level = confidence_level

        # Cache des facteurs
        self.factor_cache: Dict[str, pd.DataFrame] = {}

        # Modèles pré-entraînés
        self.trained_models: Dict[str, Any] = {}

        self._setup_default_factors()

    async def analyze_portfolio(
        self,
        returns: pd.Series,
        model_type: Optional[FactorModel] = None,
        custom_factors: Optional[pd.DataFrame] = None
    ) -> FactorAnalysisResult:
        """
        Analyse factorielle complète d'un portfolio.

        Args:
            returns: Série de returns du portfolio
            model_type: Type de modèle à utiliser
            custom_factors: Facteurs personnalisés

        Returns:
            Résultats complets de l'analyse
        """
        model = model_type or self.default_model

        logger.info(f"Starting factor analysis using {model.value} model")

        try:
            # Préparer les facteurs
            factors_df = await self._prepare_factors(returns, model, custom_factors)

            # Aligner les données
            aligned_returns, aligned_factors = await self._align_data(returns, factors_df)

            # Exécuter l'analyse selon le modèle
            if model == FactorModel.PCA:
                result = await self._run_pca_analysis(aligned_returns, aligned_factors)
            elif model in [FactorModel.FAMA_FRENCH_3, FactorModel.FAMA_FRENCH_5]:
                result = await self._run_fama_french_analysis(aligned_returns, aligned_factors, model)
            elif model == FactorModel.CAPM:
                result = await self._run_capm_analysis(aligned_returns, aligned_factors)
            elif model == FactorModel.MACRO_ECONOMIC:
                result = await self._run_macro_analysis(aligned_returns, aligned_factors)
            else:  # CUSTOM
                result = await self._run_custom_analysis(aligned_returns, aligned_factors)

            logger.info(f"Factor analysis complete. R²: {result.r_squared:.3f}, Alpha: {result.alpha:.4f}")
            return result

        except Exception as e:
            logger.error(f"Factor analysis failed: {e}")
            raise

    async def analyze_factor_loadings_over_time(
        self,
        returns: pd.Series,
        window: int = 252,
        model_type: Optional[FactorModel] = None
    ) -> pd.DataFrame:
        """
        Analyse l'évolution des expositions factorielles dans le temps.

        Args:
            returns: Série de returns
            window: Taille de la fenêtre glissante (jours)
            model_type: Type de modèle

        Returns:
            DataFrame avec expositions factorielles temporelles
        """
        model = model_type or self.default_model

        logger.info(f"Analyzing factor loadings over time with {window}-day window")

        # Préparer facteurs
        factors_df = await self._prepare_factors(returns, model)
        aligned_returns, aligned_factors = await self._align_data(returns, factors_df)

        # Analyse sur fenêtre glissante
        rolling_loadings = []
        dates = []

        for i in range(window, len(aligned_returns)):
            window_returns = aligned_returns.iloc[i-window:i]
            window_factors = aligned_factors.iloc[i-window:i]

            # Régression pour cette fenêtre
            loadings = await self._calculate_factor_loadings(window_returns, window_factors)

            rolling_loadings.append(loadings)
            dates.append(aligned_returns.index[i])

        # Créer DataFrame des résultats
        loadings_df = pd.DataFrame(rolling_loadings, index=dates)

        return loadings_df

    async def calculate_factor_attribution(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        model_type: Optional[FactorModel] = None
    ) -> Dict[str, Any]:
        """
        Calcule l'attribution de performance par facteur.

        Args:
            returns: Returns du portfolio
            benchmark_returns: Returns du benchmark (optionnel)
            model_type: Type de modèle

        Returns:
            Attribution détaillée par facteur
        """
        model = model_type or self.default_model

        # Analyse principale
        analysis_result = await self.analyze_portfolio(returns, model)

        # Calcul des contributions
        total_return = returns.sum()

        factor_contributions = {}
        for exposure in analysis_result.factor_exposures:
            factor_contributions[exposure.factor_name] = {
                "exposure": exposure.exposure,
                "factor_return": exposure.contribution_to_return,
                "contribution_pct": exposure.contribution_to_return / total_return * 100 if total_return != 0 else 0,
                "risk_contribution": exposure.contribution_to_risk
            }

        # Alpha contribution
        alpha_contribution = analysis_result.alpha * len(returns)
        alpha_contribution_pct = alpha_contribution / total_return * 100 if total_return != 0 else 0

        # Résiduel
        residual_contribution = analysis_result.residual_returns.sum()
        residual_contribution_pct = residual_contribution / total_return * 100 if total_return != 0 else 0

        attribution = {
            "total_return": total_return,
            "factor_contributions": factor_contributions,
            "alpha_contribution": alpha_contribution,
            "alpha_contribution_pct": alpha_contribution_pct,
            "residual_contribution": residual_contribution,
            "residual_contribution_pct": residual_contribution_pct,
            "explained_variance": analysis_result.explained_variance,
            "tracking_error": analysis_result.tracking_error
        }

        # Attribution vs benchmark si fourni
        if benchmark_returns is not None:
            benchmark_analysis = await self.analyze_portfolio(benchmark_returns, model)

            # Calcul des différences d'exposition
            active_exposures = {}
            for exposure in analysis_result.factor_exposures:
                factor_name = exposure.factor_name
                benchmark_exposure = next(
                    (e.exposure for e in benchmark_analysis.factor_exposures if e.factor_name == factor_name),
                    0.0
                )
                active_exposures[factor_name] = {
                    "portfolio_exposure": exposure.exposure,
                    "benchmark_exposure": benchmark_exposure,
                    "active_exposure": exposure.exposure - benchmark_exposure
                }

            attribution["benchmark_comparison"] = {
                "active_exposures": active_exposures,
                "relative_alpha": analysis_result.alpha - benchmark_analysis.alpha,
                "tracking_error": analysis_result.tracking_error
            }

        return attribution

    async def detect_style_drift(
        self,
        returns: pd.Series,
        lookback_period: int = 252,
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Détecte les dérives de style dans le temps.

        Args:
            returns: Returns du portfolio
            lookback_period: Période de lookback pour comparaison
            significance_threshold: Seuil de significativité

        Returns:
            Analyse de dérive de style
        """
        logger.info("Analyzing style drift")

        # Analyse factorielle sur fenêtre glissante
        rolling_loadings = await self.analyze_factor_loadings_over_time(
            returns, window=lookback_period
        )

        if rolling_loadings.empty:
            return {"error": "Insufficient data for style drift analysis"}

        # Calcul des changements significatifs
        drift_analysis = {}

        for factor in rolling_loadings.columns:
            factor_series = rolling_loadings[factor]

            # Test de stationnarité (Augmented Dickey-Fuller simplifié)
            # Dans une vraie implémentation, utiliserait statsmodels
            recent_mean = factor_series.tail(63).mean()  # 3 derniers mois
            historical_mean = factor_series.head(len(factor_series) - 63).mean()

            # Test t simple pour changement significatif
            recent_std = factor_series.tail(63).std()
            historical_std = factor_series.head(len(factor_series) - 63).std()

            # Statistique de test simplifiée
            if recent_std > 0 and historical_std > 0:
                pooled_std = np.sqrt((recent_std**2 + historical_std**2) / 2)
                t_stat = abs(recent_mean - historical_mean) / pooled_std if pooled_std > 0 else 0

                # Approximation critique (devrait utiliser table t)
                critical_value = 1.96  # ~5% significance level
                is_significant = t_stat > critical_value

                drift_analysis[factor] = {
                    "recent_exposure": recent_mean,
                    "historical_exposure": historical_mean,
                    "change": recent_mean - historical_mean,
                    "t_statistic": t_stat,
                    "is_significant_drift": is_significant,
                    "volatility_recent": recent_std,
                    "volatility_historical": historical_std
                }

        # Résumé global
        significant_drifts = [f for f, data in drift_analysis.items() if data["is_significant_drift"]]

        summary = {
            "analysis_period": (rolling_loadings.index[0], rolling_loadings.index[-1]),
            "factors_analyzed": list(rolling_loadings.columns),
            "significant_drifts": significant_drifts,
            "drift_count": len(significant_drifts),
            "stability_score": 1 - (len(significant_drifts) / len(rolling_loadings.columns)),
            "detailed_analysis": drift_analysis
        }

        return summary

    # === Méthodes privées ===

    async def _prepare_factors(
        self,
        returns: pd.Series,
        model_type: FactorModel,
        custom_factors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Prépare les facteurs selon le modèle choisi"""

        if model_type == FactorModel.CUSTOM and custom_factors is not None:
            return custom_factors

        # Générer facteurs selon le modèle
        if model_type == FactorModel.FAMA_FRENCH_3:
            return await self._generate_fama_french_3_factors(returns)
        elif model_type == FactorModel.FAMA_FRENCH_5:
            return await self._generate_fama_french_5_factors(returns)
        elif model_type == FactorModel.CAPM:
            return await self._generate_capm_factors(returns)
        elif model_type == FactorModel.MACRO_ECONOMIC:
            return await self._generate_macro_factors(returns)
        else:  # PCA ou autres
            return await self._generate_pca_factors(returns)

    async def _generate_fama_french_3_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Génère les facteurs Fama-French 3 (simulation)"""

        # Dans la réalité, récupérerait les vrais facteurs FF depuis Kenneth French's website
        # Ici on simule avec des données corrélées

        dates = returns.index
        n_periods = len(dates)

        # Market factor (Mkt-RF)
        market_factor = np.random.normal(0.0008, 0.015, n_periods)  # ~8bp daily, 1.5% vol

        # SMB (Small Minus Big)
        smb_factor = np.random.normal(0.0002, 0.008, n_periods)  # Size factor

        # HML (High Minus Low)
        hml_factor = np.random.normal(0.0001, 0.007, n_periods)  # Value factor

        # Risk-free rate
        rf_rate = np.full(n_periods, 0.00008)  # ~2% annual risk-free rate

        factors_df = pd.DataFrame({
            'Mkt-RF': market_factor,
            'SMB': smb_factor,
            'HML': hml_factor,
            'RF': rf_rate
        }, index=dates)

        return factors_df

    async def _generate_fama_french_5_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Génère les facteurs Fama-French 5 (simulation)"""

        # Commencer avec FF3
        ff3_factors = await self._generate_fama_french_3_factors(returns)

        dates = returns.index
        n_periods = len(dates)

        # Ajouter RMW (Robust Minus Weak) - Profitability
        rmw_factor = np.random.normal(0.0001, 0.006, n_periods)

        # Ajouter CMA (Conservative Minus Aggressive) - Investment
        cma_factor = np.random.normal(0.00005, 0.005, n_periods)

        ff3_factors['RMW'] = rmw_factor
        ff3_factors['CMA'] = cma_factor

        return ff3_factors

    async def _generate_capm_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Génère les facteurs CAPM (Market + Risk-free)"""

        dates = returns.index
        n_periods = len(dates)

        # Market factor
        market_factor = np.random.normal(0.0008, 0.015, n_periods)

        # Risk-free rate
        rf_rate = np.full(n_periods, 0.00008)

        factors_df = pd.DataFrame({
            'Market': market_factor,
            'RF': rf_rate
        }, index=dates)

        return factors_df

    async def _generate_macro_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Génère des facteurs macroéconomiques (simulation)"""

        dates = returns.index
        n_periods = len(dates)

        # Facteurs macro simulés
        factors_df = pd.DataFrame({
            'GDP_Growth': np.random.normal(0.0005, 0.002, n_periods),  # Croissance PIB
            'Inflation': np.random.normal(0.0002, 0.001, n_periods),   # Inflation
            'Interest_Rate': np.random.normal(0.0001, 0.0008, n_periods),  # Taux d'intérêt
            'USD_Index': np.random.normal(0.0, 0.005, n_periods),      # Force du dollar
            'VIX': np.random.normal(0.0, 0.02, n_periods),             # Volatilité implicite
            'Commodity': np.random.normal(0.0003, 0.012, n_periods)    # Commodities
        }, index=dates)

        return factors_df

    async def _generate_pca_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Génère des facteurs via PCA (nécessite données de référence)"""

        # Simulation de facteurs PCA
        # Dans la réalité, ferait PCA sur un univers de returns d'actifs

        dates = returns.index
        n_periods = len(dates)

        # Simuler 5 composantes principales
        factors_df = pd.DataFrame({
            'PC1': np.random.normal(0.0008, 0.015, n_periods),   # 1ère composante (market-like)
            'PC2': np.random.normal(0.0002, 0.008, n_periods),   # 2ème composante (size-like)
            'PC3': np.random.normal(0.0001, 0.006, n_periods),   # 3ème composante (value-like)
            'PC4': np.random.normal(0.0, 0.004, n_periods),      # 4ème composante
            'PC5': np.random.normal(0.0, 0.003, n_periods)       # 5ème composante
        }, index=dates)

        return factors_df

    async def _align_data(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Aligne les returns et facteurs sur les mêmes dates"""

        # Intersection des dates
        common_dates = returns.index.intersection(factors.index)

        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and factors")

        aligned_returns = returns.loc[common_dates]
        aligned_factors = factors.loc[common_dates]

        # Supprimer les NaN
        mask = ~(aligned_returns.isna() | aligned_factors.isna().any(axis=1))
        aligned_returns = aligned_returns[mask]
        aligned_factors = aligned_factors[mask]

        return aligned_returns, aligned_factors

    async def _run_fama_french_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        model_type: FactorModel
    ) -> FactorAnalysisResult:
        """Exécute l'analyse Fama-French"""

        # Préparer les données pour régression
        # Returns en excès du risk-free si disponible
        if 'RF' in factors.columns:
            excess_returns = returns - factors['RF']
            factor_cols = [col for col in factors.columns if col != 'RF']
        else:
            excess_returns = returns
            factor_cols = factors.columns.tolist()

        X = factors[factor_cols].values
        y = excess_returns.values

        # Régression linéaire multiple
        from sklearn.linear_model import LinearRegression
        from scipy import stats

        model = LinearRegression()
        model.fit(X, y)

        # Prédictions et résidus
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Statistiques du modèle
        r_squared = model.score(X, y)
        n, k = X.shape
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Tracking error (std des résidus)
        tracking_error = np.std(residuals) * np.sqrt(252)

        # Alpha (intercept)
        alpha = model.intercept_

        # Calcul des t-statistiques (simplifié)
        mse = np.sum(residuals**2) / (n - k - 1)
        var_coef = mse * np.diag(np.linalg.pinv(X.T @ X))
        se_coef = np.sqrt(var_coef)
        t_stats = model.coef_ / se_coef

        # p-values (approximation)
        p_values = [2 * (1 - stats.t.cdf(abs(t), n - k - 1)) for t in t_stats]

        # Alpha t-stat et p-value
        se_alpha = np.sqrt(mse / n)  # Approximation
        alpha_t_stat = alpha / se_alpha if se_alpha > 0 else 0
        alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), n - k - 1))

        # Créer expositions aux facteurs
        factor_exposures = []

        for i, factor_name in enumerate(factor_cols):
            # Contribution aux returns (approximation)
            factor_return_contrib = model.coef_[i] * factors[factor_name].mean() * 252

            # Contribution au risque (approximation)
            factor_risk_contrib = (model.coef_[i]**2 * factors[factor_name].var() * 252) / (returns.var() * 252)

            exposure = FactorExposure(
                factor_name=factor_name,
                exposure=model.coef_[i],
                t_statistic=t_stats[i],
                p_value=p_values[i],
                contribution_to_return=factor_return_contrib,
                contribution_to_risk=factor_risk_contrib
            )
            factor_exposures.append(exposure)

        # DataFrame des returns factoriels
        factor_returns = pd.DataFrame(
            X @ model.coef_.reshape(-1, 1),
            index=returns.index,
            columns=['Factor_Returns']
        )

        # Information ratio
        information_ratio = alpha * np.sqrt(252) / tracking_error if tracking_error > 0 else 0

        return FactorAnalysisResult(
            model_type=model_type,
            analysis_period=(returns.index[0], returns.index[-1]),
            factor_exposures=factor_exposures,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            factor_returns=factor_returns,
            residual_returns=pd.Series(residuals, index=returns.index),
            explained_variance=r_squared,
            alpha=alpha * 252,  # Annualisé
            alpha_t_stat=alpha_t_stat,
            alpha_p_value=alpha_p_value
        )

    async def _run_capm_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> FactorAnalysisResult:
        """Exécute l'analyse CAPM"""
        # Utilise la même logique que FF mais avec seulement le facteur market
        return await self._run_fama_french_analysis(returns, factors, FactorModel.CAPM)

    async def _run_macro_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> FactorAnalysisResult:
        """Exécute l'analyse macroéconomique"""
        # Utilise la même logique que FF mais avec facteurs macro
        return await self._run_fama_french_analysis(returns, factors, FactorModel.MACRO_ECONOMIC)

    async def _run_custom_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> FactorAnalysisResult:
        """Exécute l'analyse avec facteurs custom"""
        return await self._run_fama_french_analysis(returns, factors, FactorModel.CUSTOM)

    async def _run_pca_analysis(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> FactorAnalysisResult:
        """Exécute l'analyse PCA"""

        # Pour PCA, les "facteurs" sont en fait les composantes principales
        # calculées à partir d'un univers de returns

        # Standardiser les facteurs
        scaler = StandardScaler()
        factors_scaled = scaler.fit_transform(factors.values)

        # PCA
        pca = PCA(n_components=min(5, factors.shape[1]))
        components = pca.fit_transform(factors_scaled)

        # Régression sur les composantes principales
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(components, returns.values)

        # Créer un DataFrame des facteurs PCA
        pca_factors = pd.DataFrame(
            components,
            index=factors.index,
            columns=[f'PC{i+1}' for i in range(components.shape[1])]
        )

        # Utiliser la logique de régression standard
        return await self._run_fama_french_analysis(returns, pca_factors, FactorModel.PCA)

    async def _calculate_factor_loadings(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> Dict[str, float]:
        """Calcule les loadings factoriels pour une période donnée"""

        from sklearn.linear_model import LinearRegression

        X = factors.values
        y = returns.values

        model = LinearRegression()
        model.fit(X, y)

        loadings = {}
        for i, factor_name in enumerate(factors.columns):
            loadings[factor_name] = model.coef_[i]

        return loadings

    def _setup_default_factors(self) -> None:
        """Configure les facteurs par défaut"""
        # Dans une vraie implémentation, chargerait les facteurs depuis une source de données
        # Par exemple, télécharger les facteurs FF depuis Kenneth French's website

        logger.info("Default factors setup completed")

        # Facteurs par défaut en cache (simulation)
        self.factor_cache["default_market"] = pd.DataFrame()  # Placeholder