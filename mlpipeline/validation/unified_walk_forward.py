#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Analysis UNIFIÉ - Moteur événementiel unique
UN SEUL chemin d'exécution pour walk-forward ET backtest simple
Même boucle: Événements → Ordres → Fills → PnL
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from scipy import stats
from sklearn.model_selection import ParameterGrid
import mlflow
import sys
import os

# Import du moteur événementiel unifié
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from mlpipeline.execution.event_engine import EventEngine, StrategyRunner, create_event_engine

logger = logging.getLogger(__name__)

class UnifiedWalkForwardAnalyzer:
    """
    WALK-FORWARD UNIFIÉ avec moteur événementiel
    
    ✅ MÊME chemin d'exécution que backtest simple
    ✅ Boucle unifiée: Événements → Ordres → Fills → PnL  
    ✅ Walk-forward ne fait que enchaîner des périodes avec paramètres gelés
    ✅ Crédibilité maximale - pas de calculs divergents
    """
    
    def __init__(self, 
                 train_window: int = 90,      # jours
                 test_window: int = 15,       # jours  
                 step_size: int = 7,          # jours
                 min_trades: int = 5,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.002,
                 slippage: float = 0.0001):
        
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_trades = min_trades
        
        # Configuration moteur événementiel UNIQUE
        self.engine_config = {
            'initial_capital': initial_capital,
            'commission_rate': commission_rate,
            'slippage': slippage
        }
        
        # Résultats stockés
        self.walk_forward_results = []
        self.performance_summary = {}
        
        logger.info("✅ UNIFIED Walk-Forward Analyzer initialisé")
        logger.info(f"   - Train: {train_window}j, Test: {test_window}j, Step: {step_size}j")
        logger.info(f"   - ⚡ MOTEUR ÉVÉNEMENTIEL UNIFIÉ - Même chemin que backtest simple")
    
    def analyze_strategy(self, 
                        data: pd.DataFrame,
                        strategy_func: Callable,
                        param_ranges: Dict[str, List],
                        optimization_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        ANALYSE WALK-FORWARD UNIFIÉE
        
        Utilise le MÊME moteur événementiel que le backtest simple
        Garantit la cohérence totale des résultats
        
        Args:
            data: DataFrame avec données OHLCV + timestamp
            strategy_func: Fonction de stratégie compatible moteur événementiel
            param_ranges: Ranges de paramètres pour optimisation
            optimization_metric: Métrique d'optimisation
            
        Returns:
            Dict avec résultats complets walk-forward
        """
        
        logger.info("🚀 DÉMARRAGE Walk-Forward Analysis UNIFIÉ")
        logger.info(f"   📊 Données: {len(data)} points, {data.index[0]} → {data.index[-1]}")
        
        # 1. Création des splits temporels
        time_splits = self._create_time_splits(data)
        
        if len(time_splits) < 3:
            raise ValueError("❌ Pas assez de splits pour analyse robuste (minimum 3)")
        
        logger.info(f"   🔄 {len(time_splits)} périodes walk-forward créées")
        
        # 2. Walk-forward avec moteur événementiel unifié
        walk_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(time_splits):
            logger.info(f"📈 Split {i+1}/{len(time_splits)}: Train({train_start.date()}→{train_end.date()}) Test({test_start.date()}→{test_end.date()})")
            
            # Données pour ce split
            train_data = data[train_start:train_end].copy()
            test_data = data[test_start:test_end].copy()
            
            # 3. OPTIMISATION sur période TRAIN avec moteur événementiel
            logger.info(f"   🔍 Optimisation paramètres sur train ({len(train_data)} points)")
            best_params = self._optimize_parameters_unified(
                train_data, strategy_func, param_ranges, optimization_metric
            )
            
            # 4. TEST avec paramètres GELÉS sur OOS - MÊME moteur!
            logger.info(f"   🧪 Test OOS avec paramètres gelés ({len(test_data)} points)")
            oos_result = self._backtest_unified(test_data, strategy_func, best_params)
            
            # 5. Comparaison IS vs OOS pour détection overfitting
            is_result = self._backtest_unified(train_data, strategy_func, best_params)
            
            # Stockage résultats période
            period_result = {
                'split_id': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'is_performance': is_result,
                'oos_performance': oos_result,
                'overfitting_score': self._calculate_overfitting_score(is_result, oos_result)
            }
            
            walk_results.append(period_result)
            
            # Log résultats période
            logger.info(f"   📊 IS Sharpe: {is_result.get('sharpe_ratio', 0):.3f}, "
                       f"OOS Sharpe: {oos_result.get('sharpe_ratio', 0):.3f}")
        
        # 6. Analyse globale des résultats
        self.walk_forward_results = walk_results
        global_analysis = self._analyze_global_results(walk_results)
        
        logger.info("✅ Walk-Forward Analysis UNIFIÉ terminé")
        
        return {
            'individual_periods': walk_results,
            'global_analysis': global_analysis,
            'engine_config': self.engine_config,
            'methodology': 'unified_event_engine'
        }
    
    def _create_time_splits(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Création des splits temporels pour walk-forward
        """
        splits = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        
        while current_date + timedelta(days=self.train_window + self.test_window) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_window)
            test_start = train_end + timedelta(days=1)  # Gap d'1 jour
            test_end = test_start + timedelta(days=self.test_window)
            
            splits.append((train_start, train_end, test_start, test_end))
            current_date += timedelta(days=self.step_size)
        
        return splits
    
    def _optimize_parameters_unified(self, 
                                   train_data: pd.DataFrame,
                                   strategy_func: Callable,
                                   param_ranges: Dict[str, List],
                                   metric: str) -> Dict[str, Any]:
        """
        OPTIMISATION avec moteur événementiel unifié
        
        Teste chaque combinaison de paramètres avec le MÊME moteur
        que celui utilisé pour le test OOS final
        """
        
        # Génération grille paramètres
        param_grid = list(ParameterGrid(param_ranges))
        
        if len(param_grid) > 50:
            logger.warning(f"⚠️  Grille paramètres large ({len(param_grid)}), limitation à 50 combinaisons")
            param_grid = param_grid[:50]
        
        best_score = float('-inf')
        best_params = param_grid[0] if param_grid else {}
        
        # Test chaque combinaison avec moteur événementiel
        for i, params in enumerate(param_grid):
            try:
                # MÊME fonction backtest que pour OOS final
                result = self._backtest_unified(train_data, strategy_func, params)
                
                score = result.get(metric, 0)
                
                # Filtre validité (nombre trades minimum)
                if (result.get('total_trades', 0) >= self.min_trades and 
                    score > best_score):
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"⚠️  Erreur params {i}: {e}")
                continue
        
        logger.info(f"   ✅ Meilleurs paramètres: {best_params} (score: {best_score:.3f})")
        return best_params
    
    def _backtest_unified(self, 
                         data: pd.DataFrame,
                         strategy_func: Callable, 
                         params: Dict[str, Any]) -> Dict[str, float]:
        """
        BACKTEST UNIFIÉ - Même moteur pour optimisation ET test final
        
        ⚡ CRITIQUE: Cette fonction est utilisée pour:
        1. Optimisation des paramètres (période train)
        2. Test final OOS (période test) 
        3. Test IS pour comparaison
        
        Garantit la cohérence totale des résultats
        """
        
        try:
            # Création moteur événementiel avec config standard
            event_engine = create_event_engine(self.engine_config)
            strategy_runner = StrategyRunner(event_engine)
            
            # Préparation données pour moteur
            market_data = self._prepare_market_data(data)
            
            # EXÉCUTION avec moteur événementiel unifié
            performance_metrics = strategy_runner.run_strategy(
                market_data_df=market_data,
                strategy_func=strategy_func,
                strategy_params=params  # Paramètres gelés
            )
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur backtest unifié: {e}")
            return self._create_empty_result()
    
    def _prepare_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Préparation données pour moteur événementiel
        """
        market_data = data.copy()
        
        # Assurer colonnes requises
        if 'timestamp' not in market_data.columns:
            market_data['timestamp'] = market_data.index
        
        if 'symbol' not in market_data.columns:
            market_data['symbol'] = 'BTCUSDT'  # Default
        
        # Assurer OHLCV de base
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in market_data.columns:
                if col == 'volume':
                    market_data[col] = 1000000  # Volume par défaut
                else:
                    # OHLC par défaut depuis close
                    market_data[col] = market_data.get('close', market_data.iloc[:, 0])
        
        return market_data
    
    def _calculate_overfitting_score(self, is_result: Dict, oos_result: Dict) -> float:
        """
        Score d'overfitting basé sur dégradation IS → OOS
        """
        is_sharpe = is_result.get('sharpe_ratio', 0)
        oos_sharpe = oos_result.get('sharpe_ratio', 0)
        
        if is_sharpe <= 0:
            return 1.0  # Overfitting maximal
        
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        overfitting_score = max(0, min(1, degradation))
        
        return float(overfitting_score)
    
    def _analyze_global_results(self, walk_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyse globale des résultats walk-forward
        """
        if not walk_results:
            return {"status": "no_results"}
        
        # Extraction métriques
        is_sharpes = [r['is_performance'].get('sharpe_ratio', 0) for r in walk_results]
        oos_sharpes = [r['oos_performance'].get('sharpe_ratio', 0) for r in walk_results]
        overfitting_scores = [r['overfitting_score'] for r in walk_results]
        
        # Statistiques globales
        mean_is_sharpe = float(np.mean(is_sharpes))
        mean_oos_sharpe = float(np.mean(oos_sharpes))
        std_oos_sharpe = float(np.std(oos_sharpes))
        
        # Performance consistency
        consistency_score = 1.0 - (std_oos_sharpe / abs(mean_oos_sharpe)) if mean_oos_sharpe != 0 else 0
        
        # Score overfitting global
        mean_overfitting = float(np.mean(overfitting_scores))
        
        # Tests de significativité
        if len(oos_sharpes) >= 3:
            # Test t sur Sharpe OOS
            t_stat, p_value = stats.ttest_1samp(oos_sharpes, 0)
            is_significant = p_value < 0.05 and mean_oos_sharpe > 0
        else:
            t_stat, p_value = 0, 1
            is_significant = False
        
        # Verdict final
        if mean_overfitting > 0.5:
            verdict = "❌ OVERFITTING DÉTECTÉ"
        elif mean_oos_sharpe < 0.5:
            verdict = "❌ PERFORMANCE FAIBLE"
        elif not is_significant:
            verdict = "⚠️ NON-SIGNIFICATIF"
        else:
            verdict = "✅ ROBUSTE"
        
        return {
            'periods_analyzed': len(walk_results),
            'mean_is_sharpe': mean_is_sharpe,
            'mean_oos_sharpe': mean_oos_sharpe,
            'oos_sharpe_std': std_oos_sharpe,
            'consistency_score': float(consistency_score),
            'mean_overfitting_score': mean_overfitting,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'verdict': verdict,
            'engine_methodology': 'unified_event_driven'
        }
    
    def _create_empty_result(self) -> Dict[str, float]:
        """Résultat vide en cas d'erreur"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'final_value': self.engine_config['initial_capital']
        }
    
    def simple_backtest_unified(self, 
                               data: pd.DataFrame,
                               strategy_func: Callable,
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """
        BACKTEST SIMPLE utilisant le MÊME moteur que walk-forward
        
        ⚡ CRITIQUE: Utilise exactement la même fonction _backtest_unified()
        Garantit que backtest simple et walk-forward utilisent le même chemin
        """
        
        logger.info("🎯 BACKTEST SIMPLE avec moteur événementiel unifié")
        
        result = self._backtest_unified(data, strategy_func, params)
        
        # Enrichissement avec détails moteur
        result.update({
            'methodology': 'unified_event_engine',
            'engine_config': self.engine_config,
            'data_points': len(data),
            'same_engine_as_walkforward': True
        })
        
        logger.info(f"✅ Backtest simple terminé: Sharpe {result.get('sharpe_ratio', 0):.3f}")
        
        return result


# Factory function
def create_unified_analyzer(config: Dict[str, Any]) -> UnifiedWalkForwardAnalyzer:
    """
    Factory pour créer analyseur walk-forward unifié
    """
    return UnifiedWalkForwardAnalyzer(
        train_window=config.get('train_window', 90),
        test_window=config.get('test_window', 15),
        step_size=config.get('step_size', 7),
        min_trades=config.get('min_trades', 5),
        initial_capital=config.get('initial_capital', 100000),
        commission_rate=config.get('commission_rate', 0.002),
        slippage=config.get('slippage', 0.0001)
    )


# Fonction de stratégie exemple pour tests
def example_mean_reversion_strategy(market_data: pd.DataFrame, 
                                  lookback: int = 20,
                                  z_entry: float = 2.0,
                                  z_exit: float = 0.5,
                                  **kwargs) -> pd.DataFrame:
    """
    Stratégie mean reversion exemple compatible moteur événementiel
    """
    
    signals_df = market_data.copy()
    
    # Signal mean reversion basique
    sma = market_data['close'].rolling(lookback).mean()
    std = market_data['close'].rolling(lookback).std()
    
    z_score = (market_data['close'] - sma) / (std + 1e-8)
    
    # Signaux
    signals = pd.Series(0, index=market_data.index)
    signals[z_score < -z_entry] = 1  # Long
    signals[z_score > z_entry] = -1  # Short
    signals[abs(z_score) < z_exit] = 0  # Exit
    
    signals_df['signal'] = signals
    signals_df['position_size'] = abs(signals) * 1000  # Size fixe
    
    return signals_df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🧪 Test Unified Walk-Forward Analyzer")
    
    # Données test
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': 45000 + np.random.randn(len(dates)) * 1000,
        'high': 45500 + np.random.randn(len(dates)) * 1000,
        'low': 44500 + np.random.randn(len(dates)) * 1000,
        'close': 45000 + np.random.randn(len(dates)) * 1000,
        'volume': 1000000 + np.random.randn(len(dates)) * 100000
    }, index=dates)
    
    # Création analyseur
    config = {
        'train_window': 90,
        'test_window': 15, 
        'step_size': 30,
        'initial_capital': 100000
    }
    
    analyzer = create_unified_analyzer(config)
    
    # Paramètres à tester
    param_ranges = {
        'lookback': [10, 20, 30],
        'z_entry': [1.5, 2.0],
        'z_exit': [0.3, 0.5]
    }
    
    # Test walk-forward unifié
    try:
        results = analyzer.analyze_strategy(
            data=test_data,
            strategy_func=example_mean_reversion_strategy,
            param_ranges=param_ranges,
            optimization_metric='sharpe_ratio'
        )
        
        print("\n" + "="*60)
        print("🎯 RÉSULTATS WALK-FORWARD UNIFIÉ")
        print("="*60)
        
        global_analysis = results['global_analysis']
        print(f"Périodes analysées: {global_analysis['periods_analyzed']}")
        print(f"Sharpe IS moyen: {global_analysis['mean_is_sharpe']:.3f}")
        print(f"Sharpe OOS moyen: {global_analysis['mean_oos_sharpe']:.3f}")
        print(f"Score overfitting: {global_analysis['mean_overfitting_score']:.3f}")
        print(f"Verdict: {global_analysis['verdict']}")
        print(f"Méthodologie: {global_analysis['engine_methodology']}")
        
        # Test backtest simple avec MÊME moteur
        print("\n" + "="*60)
        print("🎯 BACKTEST SIMPLE (MÊME MOTEUR)")
        print("="*60)
        
        simple_result = analyzer.simple_backtest_unified(
            data=test_data,
            strategy_func=example_mean_reversion_strategy,
            params={'lookback': 20, 'z_entry': 2.0, 'z_exit': 0.5}
        )
        
        print(f"Sharpe ratio: {simple_result['sharpe_ratio']:.3f}")
        print(f"Total return: {simple_result['total_return']:.3f}")
        print(f"Max drawdown: {simple_result['max_drawdown']:.3f}")
        print(f"Même moteur: {simple_result['same_engine_as_walkforward']}")
        
        print("\n✅ COHÉRENCE GARANTIE - Walk-forward et backtest utilisent le MÊME moteur événementiel")
        
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        import traceback
        logger.error(traceback.format_exc())