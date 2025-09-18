#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moteur de Backtesting Quantitatif Avancé
Support multi-alpha, multi-timeframe avec coûts de transaction réalistes
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import mlflow
import mlflow.sklearn
from mlpipeline.utils.risk_metrics import comprehensive_metrics
from mlpipeline.portfolio.optimizer import QuantPortfolioOptimizer

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Configuration pour backtest"""
    
    # Paramètres temporels
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    timeframe: str = "1h"
    
    # Capital et sizing
    initial_capital: float = 10000.0
    max_leverage: float = 1.0
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility_target
    
    # Coûts de transaction
    trading_fee: float = 0.001  # 0.1% par trade
    slippage: float = 0.0005   # 0.05% slippage
    funding_cost: float = 0.0  # Coût funding (futures)
    
    # Gestion des risques
    max_position_size: float = 0.3  # 30% max par position
    stop_loss: Optional[float] = None  # Stop loss en %
    take_profit: Optional[float] = None  # Take profit en %
    max_drawdown_limit: Optional[float] = 0.2  # 20% max drawdown
    
    # Portfolio
    rebalancing_frequency: str = "daily"  # daily, weekly, monthly
    portfolio_method: str = "kelly_markowitz"
    
    # Données et validation
    minimum_history: int = 252  # Minimum jours pour démarrer
    walk_forward_window: int = 252  # Fenêtre walk-forward
    out_of_sample_ratio: float = 0.2  # 20% out-of-sample
    
    # Logging
    save_trades: bool = True
    save_positions: bool = True
    log_to_mlflow: bool = True

@dataclass 
class Trade:
    """Représentation d'un trade"""
    
    timestamp: datetime
    symbol: str
    alpha_source: str
    side: str  # 'long' ou 'short'
    quantity: float
    price: float
    fees: float = 0.0
    slippage: float = 0.0
    trade_id: str = field(default_factory=lambda: f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)
    
    @property
    def total_cost(self) -> float:
        return self.fees + self.slippage

@dataclass
class Position:
    """Représentation d'une position"""
    
    symbol: str
    alpha_source: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0

class BacktestEngine:
    """
    Moteur de backtesting sophistiqué pour stratégies quantitatives
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # État du portefeuille
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Historiques
        self.equity_curve = []
        self.daily_returns = []
        self.portfolio_weights_history = []
        
        # Composants
        self.portfolio_optimizer = QuantPortfolioOptimizer(
            method=config.portfolio_method,
            rebalancing_frequency=config.rebalancing_frequency
        )
        
        # Métriques
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.max_drawdown_hit = 0.0
        
        logger.info(f"🎯 BacktestEngine initialisé: capital={config.initial_capital}, "
                   f"période={config.start_date} → {config.end_date}")
    
    def run_backtest(self, 
                    alpha_signals: Dict[str, pd.DataFrame],
                    market_data: pd.DataFrame,
                    regime_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Exécution du backtest principal
        
        Args:
            alpha_signals: Dict {alpha_name: DataFrame avec signaux}
            market_data: DataFrame avec prix OHLCV
            regime_data: DataFrame optionnel avec info régimes
            
        Returns:
            Dict avec résultats complets du backtest
        """
        
        logger.info("🚀 Début backtest...")
        
        try:
            # 1. Préparation données
            aligned_data = self._prepare_data(alpha_signals, market_data, regime_data)
            
            # 2. Walk-forward validation
            if self.config.out_of_sample_ratio > 0:
                results = self._run_walk_forward_backtest(aligned_data)
            else:
                results = self._run_simple_backtest(aligned_data)
            
            # 3. Calcul métriques finales
            final_metrics = self._calculate_final_metrics()
            results.update(final_metrics)
            
            # 4. Logging MLflow
            if self.config.log_to_mlflow:
                self._log_backtest_to_mlflow(results)
            
            logger.info(f"✅ Backtest terminé: return={results.get('total_return', 0)*100:.2f}%, "
                       f"sharpe={results.get('sharpe_ratio', 0):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur backtest: {e}")
            raise
    
    def _prepare_data(self, alpha_signals: Dict[str, pd.DataFrame], 
                     market_data: pd.DataFrame, regime_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Préparation et alignement des données"""
        
        # Date range filtering
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        # Alignement sur market_data
        aligned_data = market_data.copy()
        aligned_data = aligned_data[(aligned_data.index >= start_date) & 
                                  (aligned_data.index <= end_date)]
        
        # Ajout signaux alpha
        for alpha_name, signals_df in alpha_signals.items():
            # Détection de la colonne signal principale
            signal_cols = [col for col in signals_df.columns if 'signal' in col.lower() or 'position' in col.lower()]
            if not signal_cols:
                # Prendre première colonne numérique
                signal_cols = [col for col in signals_df.columns if signals_df[col].dtype in ['float64', 'int64']]
            
            if signal_cols:
                main_signal_col = signal_cols[0]
                aligned_data[f'{alpha_name}_signal'] = signals_df[main_signal_col].reindex(aligned_data.index, fill_value=0)
            else:
                logger.warning(f"⚠️ Pas de signal trouvé pour {alpha_name}")
                aligned_data[f'{alpha_name}_signal'] = 0
        
        # Ajout données régimes
        if regime_data is not None:
            regime_cols = [col for col in regime_data.columns if 'regime' in col.lower()]
            for col in regime_cols:
                aligned_data[f'regime_{col}'] = regime_data[col].reindex(aligned_data.index, fill_value=0)
        
        # Calcul returns
        aligned_data['returns'] = aligned_data['close'].pct_change().fillna(0)
        
        logger.info(f"📊 Données alignées: {aligned_data.shape}, colonnes={aligned_data.columns.tolist()}")
        
        return aligned_data
    
    def _run_simple_backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest simple sans walk-forward"""
        
        logger.info("📈 Backtest simple...")
        
        # Extraction signaux alpha
        alpha_columns = [col for col in data.columns if col.endswith('_signal')]
        alpha_names = [col.replace('_signal', '') for col in alpha_columns]
        
        if not alpha_names:
            logger.warning("⚠️ Aucun signal alpha trouvé")
            return {'error': 'No alpha signals found'}
        
        # Simulation jour par jour
        for i, (timestamp, row) in enumerate(data.iterrows()):
            
            if i < self.config.minimum_history:
                continue  # Période de warm-up
            
            # Données historiques pour optimisation
            historical_data = data.iloc[max(0, i-self.config.walk_forward_window):i]
            
            # Portfolio optimization si rebalancement nécessaire
            if self._should_rebalance(timestamp):
                target_weights = self._optimize_portfolio(historical_data, alpha_names)
                self._rebalance_portfolio(timestamp, row, target_weights)
            
            # Mise à jour positions et equity
            self._update_positions(timestamp, row)
            self._record_equity(timestamp)
        
        return {
            'equity_curve': pd.Series(self.equity_curve, index=data.index[-len(self.equity_curve):]),
            'trades': self.trades,
            'positions_history': self.portfolio_weights_history
        }
    
    def _run_walk_forward_backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest avec validation walk-forward"""
        
        logger.info("📈 Backtest walk-forward...")
        
        # Paramètres walk-forward
        total_periods = len(data)
        in_sample_size = int(total_periods * (1 - self.config.out_of_sample_ratio))
        step_size = max(1, in_sample_size // 10)  # 10 périodes de test
        
        results = []
        
        for start_idx in range(self.config.minimum_history, total_periods - step_size, step_size):
            end_idx = min(start_idx + in_sample_size, total_periods - step_size)
            test_start = end_idx
            test_end = min(test_start + step_size, total_periods)
            
            if test_end <= test_start:
                break
            
            logger.debug(f"Walk-forward: train={start_idx}:{end_idx}, test={test_start}:{test_end}")
            
            # Données train/test
            train_data = data.iloc[start_idx:end_idx]
            test_data = data.iloc[test_start:test_end]
            
            # Optimisation sur train
            alpha_columns = [col for col in train_data.columns if col.endswith('_signal')]
            alpha_names = [col.replace('_signal', '') for col in alpha_columns]
            
            if alpha_names:
                # Simulation sur période test
                period_results = self._simulate_period(train_data, test_data, alpha_names)
                results.append(period_results)
        
        # Agrégation résultats
        return self._aggregate_walk_forward_results(results)
    
    def _optimize_portfolio(self, historical_data: pd.DataFrame, alpha_names: List[str]) -> pd.Series:
        """Optimisation allocation portfolio"""
        
        try:
            # Construction matrice returns alphas
            alpha_returns = pd.DataFrame()
            
            for alpha_name in alpha_names:
                signal_col = f'{alpha_name}_signal'
                if signal_col in historical_data.columns:
                    # Return théorique = signal * market return
                    market_returns = historical_data['returns']
                    alpha_signal = historical_data[signal_col]
                    
                    # Lag signal pour éviter lookahead bias
                    alpha_signal_lagged = alpha_signal.shift(1).fillna(0)
                    alpha_return = alpha_signal_lagged * market_returns
                    
                    alpha_returns[alpha_name] = alpha_return
            
            if alpha_returns.empty:
                # Fallback equal weight
                return pd.Series(1/len(alpha_names), index=alpha_names)
            
            # Optimisation
            optimization_result = self.portfolio_optimizer.optimize_portfolio(alpha_returns)
            return optimization_result['weights']
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur optimisation portfolio: {e}")
            return pd.Series(1/len(alpha_names), index=alpha_names)
    
    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Vérification nécessité rebalancement"""
        
        return self.portfolio_optimizer.rebalance_check(timestamp)
    
    def _rebalance_portfolio(self, timestamp: datetime, market_row: pd.Series, target_weights: pd.Series):
        """Rebalancement du portefeuille"""
        
        current_price = market_row['close']
        
        for alpha_name, target_weight in target_weights.items():
            target_value = self.current_capital * target_weight
            target_quantity = target_value / current_price
            
            # Position actuelle
            position_key = f"BTCUSDT_{alpha_name}"  # Simplifié pour test
            current_position = self.positions.get(position_key)
            current_quantity = current_position.quantity if current_position else 0
            
            # Trade nécessaire
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > 0.001:  # Seuil minimal
                # Exécution trade
                trade = self._execute_trade(
                    timestamp=timestamp,
                    symbol="BTCUSDT",
                    alpha_source=alpha_name,
                    quantity=trade_quantity,
                    price=current_price
                )
                
                # Mise à jour position
                self._update_position_from_trade(trade)
    
    def _execute_trade(self, timestamp: datetime, symbol: str, alpha_source: str, 
                      quantity: float, price: float) -> Trade:
        """Exécution d'un trade avec coûts réalistes"""
        
        # Calcul coûts
        notional = abs(quantity * price)
        fees = notional * self.config.trading_fee
        slippage_cost = notional * self.config.slippage
        
        # Ajustement prix avec slippage
        adjusted_price = price * (1 + self.config.slippage * np.sign(quantity))
        
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            alpha_source=alpha_source,
            side='long' if quantity > 0 else 'short',
            quantity=quantity,
            price=adjusted_price,
            fees=fees,
            slippage=slippage_cost
        )
        
        # Mise à jour capital
        self.current_capital -= (fees + slippage_cost)
        self.total_fees += fees + slippage_cost
        
        # Enregistrement
        self.trades.append(trade)
        self.total_trades += 1
        
        logger.debug(f"Trade exécuté: {trade.side} {abs(trade.quantity):.4f} {symbol} @ {trade.price:.2f}")
        
        return trade
    
    def _update_position_from_trade(self, trade: Trade):
        """Mise à jour position suite à un trade"""
        
        position_key = f"{trade.symbol}_{trade.alpha_source}"
        
        if position_key in self.positions:
            # Position existante
            position = self.positions[position_key]
            
            # Calcul prix moyen pondéré
            total_quantity = position.quantity + trade.quantity
            if total_quantity != 0:
                weighted_price = (position.quantity * position.entry_price + 
                                trade.quantity * trade.price) / total_quantity
                position.entry_price = weighted_price
            
            position.quantity = total_quantity
            
            # Suppression si position fermée
            if abs(position.quantity) < 0.001:
                del self.positions[position_key]
        else:
            # Nouvelle position
            if abs(trade.quantity) > 0.001:
                self.positions[position_key] = Position(
                    symbol=trade.symbol,
                    alpha_source=trade.alpha_source,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    current_price=trade.price
                )
    
    def _update_positions(self, timestamp: datetime, market_row: pd.Series):
        """Mise à jour valeur des positions"""
        
        current_price = market_row['close']
        
        for position in self.positions.values():
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def _record_equity(self, timestamp: datetime):
        """Enregistrement équité du portfolio"""
        
        # Valeur des positions
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        total_equity = self.current_capital + positions_value
        
        self.equity_curve.append(total_equity)
        
        # Return daily
        if len(self.equity_curve) > 1:
            daily_return = (total_equity / self.equity_curve[-2]) - 1
            self.daily_returns.append(daily_return)
        
        # Vérification drawdown limit
        if self.config.max_drawdown_limit:
            peak_equity = max(self.equity_curve)
            current_drawdown = (peak_equity - total_equity) / peak_equity
            
            if current_drawdown > self.config.max_drawdown_limit:
                logger.warning(f"⚠️ Max drawdown atteint: {current_drawdown:.2%}")
                self.max_drawdown_hit = current_drawdown
    
    def _simulate_period(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                        alpha_names: List[str]) -> Dict:
        """Simulation sur une période test"""
        
        # Optimisation sur train
        target_weights = self._optimize_portfolio(train_data, alpha_names)
        
        # Simulation sur test
        period_returns = []
        
        for timestamp, row in test_data.iterrows():
            # Return portfolio
            portfolio_return = 0
            
            for alpha_name in alpha_names:
                signal_col = f'{alpha_name}_signal'
                if signal_col in row.index:
                    alpha_signal = row[signal_col]
                    alpha_weight = target_weights.get(alpha_name, 0)
                    market_return = row['returns']
                    
                    # Return contribution (avec lag)
                    contribution = alpha_weight * alpha_signal * market_return
                    portfolio_return += contribution
            
            # Ajustement coûts
            portfolio_return -= self.config.trading_fee  # Coût turnover
            
            period_returns.append(portfolio_return)
        
        return {
            'returns': period_returns,
            'weights': target_weights.to_dict(),
            'period_start': test_data.index[0],
            'period_end': test_data.index[-1]
        }
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Agrégation résultats walk-forward"""
        
        all_returns = []
        all_weights = []
        
        for period_result in results:
            all_returns.extend(period_result['returns'])
            all_weights.append(period_result['weights'])
        
        # Equity curve synthétique
        equity_curve = [self.config.initial_capital]
        for ret in all_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        return {
            'equity_curve': pd.Series(equity_curve[1:]),  # Sans valeur initiale
            'returns': all_returns,
            'weights_history': all_weights
        }
    
    def _calculate_final_metrics(self) -> Dict:
        """Calcul métriques finales du backtest"""
        
        if len(self.equity_curve) < 2:
            return {}
        
        # Return total
        total_return = (self.equity_curve[-1] / self.config.initial_capital) - 1
        
        # Conversion en Series pour métriques
        equity_series = pd.Series(self.equity_curve)
        returns_series = equity_series.pct_change().dropna()
        
        # Métriques complètes
        metrics = comprehensive_metrics(returns_series.values)
        
        # Métriques additionnelles trading
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_trade_cost = self.total_fees / max(self.total_trades, 1)
        
        # Métriques finales
        final_metrics = {
            'total_return': total_return,
            'final_capital': self.equity_curve[-1],
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_fees': self.total_fees,
            'avg_trade_cost': avg_trade_cost,
            'max_drawdown_hit': self.max_drawdown_hit,
            **metrics
        }
        
        return final_metrics
    
    def _log_backtest_to_mlflow(self, results: Dict):
        """Logging résultats vers MLflow"""
        
        try:
            with mlflow.start_run(nested=True):
                # Configuration
                mlflow.log_param("start_date", self.config.start_date)
                mlflow.log_param("end_date", self.config.end_date)
                mlflow.log_param("initial_capital", self.config.initial_capital)
                mlflow.log_param("portfolio_method", self.config.portfolio_method)
                mlflow.log_param("trading_fee", self.config.trading_fee)
                
                # Métriques performance
                for key, value in results.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        mlflow.log_metric(f"backtest_{key}", value)
                
                logger.debug("📊 Backtest loggé dans MLflow")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging MLflow: {e}")


# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def run_simple_backtest(alpha_signals: Dict[str, pd.DataFrame],
                       market_data: pd.DataFrame,
                       config: Optional[BacktestConfig] = None) -> Dict:
    """
    Interface simplifiée pour backtest rapide
    """
    
    if config is None:
        config = BacktestConfig()
    
    engine = BacktestEngine(config)
    return engine.run_backtest(alpha_signals, market_data)


def compare_strategies(strategies: Dict[str, Dict],
                      market_data: pd.DataFrame,
                      config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    """
    Comparaison multiple de stratégies
    """
    
    if config is None:
        config = BacktestConfig()
    
    results = []
    
    for strategy_name, alpha_signals in strategies.items():
        logger.info(f"📊 Backtest {strategy_name}...")
        
        engine = BacktestEngine(config)
        backtest_result = engine.run_backtest(alpha_signals, market_data)
        
        summary = {
            'strategy': strategy_name,
            'total_return': backtest_result.get('total_return', 0),
            'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
            'max_drawdown': backtest_result.get('max_drawdown', 0),
            'calmar_ratio': backtest_result.get('calmar_ratio', 0),
            'total_trades': backtest_result.get('total_trades', 0),
            'win_rate': backtest_result.get('win_rate', 0),
        }
        
        results.append(summary)
    
    comparison_df = pd.DataFrame(results)
    comparison_df.set_index('strategy', inplace=True)
    
    return comparison_df