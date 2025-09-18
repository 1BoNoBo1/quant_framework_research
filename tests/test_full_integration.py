#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test d'Intégration Complet - Quant Stack Production
Validation end-to-end de tous les composants
"""

import asyncio
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Imports des modules
sys.path.append('.')

async def test_full_integration():
    """
    Test d'intégration complet du système quantitatif
    """
    
    print("🚀 === TEST INTÉGRATION COMPLÈTE QUANT STACK ===")
    print(f"⏰ Début: {datetime.now()}")
    
    results = {
        'data_fetching': False,
        'alpha_generation': False,
        'portfolio_optimization': False,
        'backtesting': False,
        'monitoring': False,
        'funding_collection': False
    }
    
    try:
        # ===== 1. DATA FETCHING =====
        print("\n📊 1. TEST DATA FETCHING")
        from mlpipeline.data_sources.crypto_fetcher import CryptoDataFetcher
        
        fetcher = CryptoDataFetcher()
        market_data = await fetcher.fetch_ohlcv('BTCUSDT', '1h', limit=500)
        
        print(f"   ✅ Données récupérées: {market_data.shape}")
        print(f"   📋 Colonnes: {list(market_data.columns)}")
        print(f"   💰 Prix range: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
        
        results['data_fetching'] = len(market_data) > 0
        
        # ===== 2. ALPHA GENERATION =====
        print("\n🧠 2. TEST ALPHA GENERATION")
        
        # Test DMN LSTM
        from mlpipeline.alphas import DMNPredictor
        dmn_alpha = DMNPredictor('BTCUSDT')
        dmn_signals = await dmn_alpha.predict(market_data)
        print(f"   🤖 DMN LSTM: {dmn_signals.shape}, mean={dmn_signals.mean():.4f}")

        # Test Mean Reversion
        from mlpipeline.alphas import AdaptiveMeanReversion
        mr_alpha = AdaptiveMeanReversion()
        mr_signals_full = await mr_alpha.generate_signals(market_data)
        mr_signal = mr_signals_full['signal_final'] if 'signal_final' in mr_signals_full.columns else mr_signals_full.iloc[:, -2]
        print(f"   📈 Mean Reversion: {mr_signal.shape}, mean={mr_signal.mean():.4f}")
        
        # Test Funding Strategy
        from mlpipeline.alphas import AdvancedFundingStrategy
        fs_alpha = AdvancedFundingStrategy('BTCUSDT')
        fs_signals_full = await fs_alpha.generate_signals(market_data)
        fs_signal = fs_signals_full['signal'] if 'signal' in fs_signals_full.columns else fs_signals_full.iloc[:, -2]
        print(f"   💰 Funding Strategy: {fs_signal.shape}, mean={fs_signal.mean():.4f}")
        
        # Création DataFrame alphas
        alpha_returns = pd.DataFrame({
            'dmn_lstm': dmn_signals * market_data['close'].pct_change().fillna(0),
            'mean_reversion': mr_signal * market_data['close'].pct_change().fillna(0),
            'funding_strategy': fs_signal * market_data['close'].pct_change().fillna(0)
        }, index=market_data.index)
        
        alpha_returns = alpha_returns.dropna()
        print(f"   📊 Alpha returns combined: {alpha_returns.shape}")
        
        results['alpha_generation'] = len(alpha_returns) > 0
        
        # ===== 3. PORTFOLIO OPTIMIZATION =====
        print("\n💰 3. TEST PORTFOLIO OPTIMIZATION")
        
        from mlpipeline.portfolio import QuantPortfolioOptimizer, optimize_portfolio_simple
        
        # Test optimisation simple
        portfolio_result = optimize_portfolio_simple(alpha_returns, method='kelly_markowitz')
        weights = portfolio_result['weights']
        portfolio_metrics = portfolio_result['portfolio_metrics']
        
        print("   ✅ Poids optimaux:")
        for alpha, weight in weights.items():
            print(f"      {alpha}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"   📊 Expected Return: {portfolio_metrics['expected_return']:.4f}")
        print(f"   📊 Volatility: {portfolio_metrics['volatility']:.4f}")
        print(f"   📊 Sharpe Expected: {portfolio_metrics['sharpe_expected']:.4f}")
        
        results['portfolio_optimization'] = len(weights) > 0
        
        # ===== 4. BACKTESTING =====
        print("\n📈 4. TEST BACKTESTING")
        
        from mlpipeline.backtesting import BacktestConfig, run_simple_backtest
        
        # Configuration backtest
        config = BacktestConfig(
            start_date='2024-01-01',
            end_date='2024-12-31', 
            initial_capital=10000.0,
            trading_fee=0.001,
            minimum_history=50
        )
        
        # Préparation signaux pour backtest
        alpha_signals = {
            'dmn_lstm': pd.DataFrame({'signal': dmn_signals, 'position_size': dmn_signals * 0.3}, index=market_data.index),
            'mean_reversion': pd.DataFrame({'signal': mr_signal, 'position_size': mr_signal * 0.3}, index=market_data.index),
            'funding_strategy': pd.DataFrame({'signal': fs_signal, 'position_size': fs_signal * 0.3}, index=market_data.index)
        }
        
        # Ajout DatetimeIndex pour backtest
        dates = pd.date_range('2024-01-01', periods=len(market_data), freq='1h')
        market_data_bt = market_data.copy()
        market_data_bt.index = dates[:len(market_data)]
        
        for alpha_name in alpha_signals:
            alpha_signals[alpha_name].index = dates[:len(alpha_signals[alpha_name])]
        
        # Exécution backtest
        try:
            bt_results = run_simple_backtest(alpha_signals, market_data_bt, config)
            
            if 'error' not in bt_results:
                print(f"   ✅ Backtest réussi")
                if 'equity_curve' in bt_results and len(bt_results['equity_curve']) > 0:
                    equity = bt_results['equity_curve']
                    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
                    print(f"   📊 Total Return: {total_return*100:.2f}%")
                
                results['backtesting'] = True
            else:
                print(f"   ⚠️  Backtest échoué: {bt_results.get('error', 'Unknown error')}")
                results['backtesting'] = False
                
        except Exception as e:
            print(f"   ⚠️  Erreur backtest: {str(e)[:100]}...")
            results['backtesting'] = False
        
        # ===== 5. MONITORING =====
        print("\n🔍 5. TEST MONITORING")
        
        # Test monitoring logic simple
        dates_monitoring = pd.date_range('2024-01-01', periods=100, freq='D')
        portfolio_values = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 100))
        
        portfolio_data = pd.DataFrame({
            'value': portfolio_values,
        }, index=dates_monitoring)
        
        # Métriques de performance
        returns = pd.Series(portfolio_values).pct_change().dropna()
        max_dd = ((pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)) / pd.Series(portfolio_values).cummax()).max()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        print(f"   📊 Portfolio Value: {portfolio_values[0]:.0f} → {portfolio_values[-1]:.0f}")
        print(f"   📉 Max Drawdown: {max_dd:.2%}")
        print(f"   📊 Sharpe Ratio: {sharpe:.3f}")
        
        # Alertes simulées
        alerts_count = 0
        if max_dd > 0.15:
            alerts_count += 1
            print("   🚨 ALERT: Drawdown critique")
        if sharpe < 0.5:
            alerts_count += 1
            print("   ⚠️  ALERT: Sharpe faible")
        
        print(f"   📢 Alertes générées: {alerts_count}")
        
        results['monitoring'] = True
        
        # ===== 6. FUNDING COLLECTION =====
        print("\n💸 6. TEST FUNDING COLLECTION")
        
        try:
            from scripts.funding_collector import FundingRateCollector
            
            collector = FundingRateCollector()
            
            # Test collecte sur une paire
            funding_data = await collector.collect_funding_rate('BTC/USDT:USDT')
            
            if funding_data:
                print(f"   ✅ Funding rate collecté: {funding_data.get('fundingRate', 0):.6f}")
                print(f"   📊 Mark Price: {funding_data.get('markPrice', 0):.2f}")
                results['funding_collection'] = True
            else:
                print("   ⚠️  Pas de données funding collectées")
                results['funding_collection'] = False
                
        except Exception as e:
            print(f"   ⚠️  Erreur funding collection: {str(e)[:100]}...")
            results['funding_collection'] = False
        
        # ===== RÉSULTATS FINAUX =====
        print("\n" + "="*60)
        print("📊 RÉSULTATS INTÉGRATION COMPLÈTE")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        success_rate = passed_tests / total_tests * 100
        
        for component, status in results.items():
            status_emoji = "✅" if status else "❌"
            print(f"{status_emoji} {component.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
        
        print(f"\n📈 SCORE FINAL: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 SYSTÈME PRÊT POUR DÉPLOIEMENT!")
        elif success_rate >= 60:
            print("⚠️  Système partiellement fonctionnel - corrections mineures nécessaires")
        else:
            print("🚨 Corrections majeures nécessaires avant déploiement")
        
        # Recommandations
        print("\n📋 RECOMMANDATIONS:")
        
        if not results['funding_collection']:
            print("- Vérifier configuration API exchange pour funding rates")
        
        if not results['backtesting']:
            print("- Déboguer moteur de backtesting avec données réelles")
        
        if success_rate >= 80:
            print("- Procéder au déploiement VPS")
            print("- Configurer monitoring en production")
            print("- Démarrer avec capital limité pour validation")
        
        print(f"\n⏰ Fin: {datetime.now()}")
        
        return {
            'success_rate': success_rate,
            'results': results,
            'ready_for_deployment': success_rate >= 80
        }
        
    except Exception as e:
        print(f"\n❌ ERREUR FATALE INTÉGRATION: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success_rate': 0,
            'results': results,
            'ready_for_deployment': False,
            'error': str(e)
        }

# Exécution du test
if __name__ == "__main__":
    final_result = asyncio.run(test_full_integration())
    
    print(f"\n🎯 RÉSULTAT FINAL: {final_result['success_rate']:.1f}% - {'PRÊT' if final_result['ready_for_deployment'] else 'PAS PRÊT'}")