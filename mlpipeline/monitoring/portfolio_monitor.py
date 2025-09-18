#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Monitor - Monitoring final du portfolio
Synthèse des performances, métriques et alertes en temps réel
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioMonitor:
    """Monitoring complet du portfolio quantitatif"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = self.data_dir / "artifacts"
        self.logs_dir = Path("logs")

        # Métriques de performance
        self.metrics = {
            "portfolio": {},
            "strategies": {},
            "risk": {},
            "execution": {},
            "alerts": []
        }

    def load_strategy_metrics(self) -> Dict[str, Any]:
        """Charge les métriques de toutes les stratégies"""
        strategy_metrics = {}

        # DMN Strategy
        dmn_files = list(self.artifacts_dir.glob("dmn_metrics_*.json"))
        for dmn_file in dmn_files:
            symbol = dmn_file.stem.split('_')[-1]
            try:
                with open(dmn_file, 'r') as f:
                    data = json.load(f)
                    strategy_metrics[f"DMN_{symbol}"] = data
                    logger.info(f"📊 DMN {symbol} métriques chargées")
            except Exception as e:
                logger.warning(f"⚠️ Erreur DMN {symbol}: {e}")

        # Mean Reversion Strategy
        mr_files = list(self.artifacts_dir.glob("mr_metrics_*.json"))
        for mr_file in mr_files:
            symbol = mr_file.stem.split('_')[-1]
            try:
                with open(mr_file, 'r') as f:
                    data = json.load(f)
                    strategy_metrics[f"MR_{symbol}"] = data
                    logger.info(f"📊 Mean Reversion {symbol} métriques chargées")
            except Exception as e:
                logger.warning(f"⚠️ Erreur MR {symbol}: {e}")

        # Alternative naming convention
        alt_mr_files = list(self.artifacts_dir.glob("mean_reversion_metrics_*.json"))
        for mr_file in alt_mr_files:
            symbol = mr_file.stem.split('_')[-1]
            strategy_key = f"MR_{symbol}"
            if strategy_key not in strategy_metrics:
                try:
                    with open(mr_file, 'r') as f:
                        data = json.load(f)
                        strategy_metrics[strategy_key] = data
                        logger.info(f"📊 Mean Reversion (alt) {symbol} métriques chargées")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur MR alt {symbol}: {e}")

        return strategy_metrics

    def load_signals_data(self) -> Dict[str, pd.DataFrame]:
        """Charge les données de signaux"""
        signals_data = {}

        # Mean Reversion signals
        mr_signals = list(self.artifacts_dir.glob("mr_signals_*.parquet"))
        for signal_file in mr_signals:
            symbol = signal_file.stem.split('_')[-1]
            try:
                df = pd.read_parquet(signal_file)
                signals_data[f"MR_signals_{symbol}"] = df
                logger.info(f"📈 Signaux MR {symbol} chargés ({len(df)} lignes)")
            except Exception as e:
                logger.warning(f"⚠️ Erreur signaux MR {symbol}: {e}")

        return signals_data

    def load_regime_data(self) -> Dict[str, pd.DataFrame]:
        """Charge les données de régimes de marché"""
        regime_data = {}

        regime_files = list(self.artifacts_dir.glob("regime_states_*.parquet"))
        for regime_file in regime_files:
            symbol = regime_file.stem.split('_')[-1]
            try:
                df = pd.read_parquet(regime_file)
                regime_data[f"regime_{symbol}"] = df
                logger.info(f"🎯 Régimes {symbol} chargés ({len(df)} lignes)")
            except Exception as e:
                logger.warning(f"⚠️ Erreur régimes {symbol}: {e}")

        return regime_data

    def calculate_portfolio_metrics(self, strategy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les métriques agrégées du portfolio"""
        if not strategy_metrics:
            logger.warning("⚠️ Aucune métrique de stratégie disponible")
            return {}

        # Agrégation des métriques
        total_trades = 0
        total_pnl = 0
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []

        for strategy_name, metrics in strategy_metrics.items():
            if isinstance(metrics, dict):
                # Extraction métriques selon format
                trades = metrics.get('total_trades', metrics.get('trades_count', 0))
                pnl = metrics.get('total_pnl', metrics.get('pnl', 0))
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = metrics.get('max_drawdown', metrics.get('max_dd', 0))
                win_rate = metrics.get('win_rate', 0)

                total_trades += trades
                total_pnl += pnl

                if sharpe != 0:
                    sharpe_ratios.append(sharpe)
                if max_dd != 0:
                    max_drawdowns.append(abs(max_dd))
                if win_rate != 0:
                    win_rates.append(win_rate)

        # Calculs agrégés
        portfolio_metrics = {
            "total_strategies": len(strategy_metrics),
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "avg_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
            "max_sharpe": max(sharpe_ratios) if sharpe_ratios else 0,
            "min_sharpe": min(sharpe_ratios) if sharpe_ratios else 0,
            "avg_max_drawdown": np.mean(max_drawdowns) if max_drawdowns else 0,
            "worst_drawdown": max(max_drawdowns) if max_drawdowns else 0,
            "avg_win_rate": np.mean(win_rates) if win_rates else 0,
            "best_win_rate": max(win_rates) if win_rates else 0
        }

        return portfolio_metrics

    def analyze_risk_metrics(self, portfolio_metrics: Dict[str, Any],
                           strategy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse des métriques de risque"""
        risk_analysis = {
            "risk_level": "UNKNOWN",
            "diversification_score": 0,
            "concentration_risk": 0,
            "drawdown_risk": "LOW",
            "performance_stability": 0,
            "alerts": []
        }

        if not portfolio_metrics:
            return risk_analysis

        # Analyse du drawdown
        avg_dd = portfolio_metrics.get("avg_max_drawdown", 0)
        worst_dd = portfolio_metrics.get("worst_drawdown", 0)

        if worst_dd > 0.3:  # 30%
            risk_analysis["drawdown_risk"] = "HIGH"
            risk_analysis["alerts"].append("⚠️ Drawdown max > 30%")
        elif worst_dd > 0.15:  # 15%
            risk_analysis["drawdown_risk"] = "MEDIUM"
        else:
            risk_analysis["drawdown_risk"] = "LOW"

        # Score de diversification
        num_strategies = portfolio_metrics.get("total_strategies", 0)
        if num_strategies >= 6:
            risk_analysis["diversification_score"] = 0.9
        elif num_strategies >= 4:
            risk_analysis["diversification_score"] = 0.7
        elif num_strategies >= 2:
            risk_analysis["diversification_score"] = 0.5
        else:
            risk_analysis["diversification_score"] = 0.2
            risk_analysis["alerts"].append("⚠️ Faible diversification")

        # Stabilité des performances
        sharpe_ratios = []
        for metrics in strategy_metrics.values():
            if isinstance(metrics, dict):
                sharpe = metrics.get('sharpe_ratio', 0)
                if sharpe != 0:
                    sharpe_ratios.append(sharpe)

        if sharpe_ratios:
            sharpe_std = np.std(sharpe_ratios)
            if sharpe_std < 0.5:
                risk_analysis["performance_stability"] = 0.8
            elif sharpe_std < 1.0:
                risk_analysis["performance_stability"] = 0.6
            else:
                risk_analysis["performance_stability"] = 0.3
                risk_analysis["alerts"].append("⚠️ Performance instable entre stratégies")

        # Niveau de risque global
        avg_sharpe = portfolio_metrics.get("avg_sharpe", 0)
        if avg_sharpe > 1.0 and risk_analysis["drawdown_risk"] == "LOW":
            risk_analysis["risk_level"] = "LOW"
        elif avg_sharpe > 0.5:
            risk_analysis["risk_level"] = "MEDIUM"
        else:
            risk_analysis["risk_level"] = "HIGH"

        return risk_analysis

    def check_execution_health(self) -> Dict[str, Any]:
        """Vérifie la santé de l'exécution"""
        execution_health = {
            "pipeline_status": "UNKNOWN",
            "last_execution": None,
            "success_rate": 0,
            "execution_time": 0,
            "issues": []
        }

        # Chercher le dernier rapport de pipeline
        pattern = str(self.logs_dir / "hybrid_pipeline_report_*.json")
        files = glob.glob(pattern)

        if files:
            latest_file = max(files, key=os.path.getctime)
            try:
                with open(latest_file, 'r') as f:
                    pipeline_data = json.load(f)

                execution_health["last_execution"] = pipeline_data.get("timestamp")
                execution_health["success_rate"] = pipeline_data.get("summary", {}).get("success_rate", 0)
                execution_health["execution_time"] = pipeline_data.get("execution_time", 0)

                if execution_health["success_rate"] == 1.0:
                    execution_health["pipeline_status"] = "HEALTHY"
                elif execution_health["success_rate"] > 0.8:
                    execution_health["pipeline_status"] = "DEGRADED"
                    execution_health["issues"].append("Taux de succès < 100%")
                else:
                    execution_health["pipeline_status"] = "CRITICAL"
                    execution_health["issues"].append("Taux de succès < 80%")

                logger.info(f"📊 Pipeline: {execution_health['pipeline_status']}")

            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture pipeline: {e}")
                execution_health["issues"].append(f"Erreur lecture pipeline: {e}")
        else:
            execution_health["issues"].append("Aucun rapport de pipeline trouvé")

        return execution_health

    def generate_alerts(self, portfolio_metrics: Dict[str, Any],
                       risk_metrics: Dict[str, Any],
                       execution_health: Dict[str, Any]) -> List[str]:
        """Génère les alertes basées sur les métriques"""
        alerts = []

        # Alertes portfolio
        if portfolio_metrics:
            if portfolio_metrics.get("avg_sharpe", 0) < 0:
                alerts.append("🚨 CRITIQUE: Sharpe ratio négatif")
            elif portfolio_metrics.get("avg_sharpe", 0) < 0.5:
                alerts.append("⚠️ WARNING: Sharpe ratio faible")

            if portfolio_metrics.get("total_trades", 0) == 0:
                alerts.append("🚨 CRITIQUE: Aucun trade généré")

        # Alertes risque
        if risk_metrics.get("risk_level") == "HIGH":
            alerts.append("🚨 RISQUE ÉLEVÉ: Révision nécessaire")

        if risk_metrics.get("diversification_score", 0) < 0.5:
            alerts.append("⚠️ Diversification insuffisante")

        # Alertes exécution
        if execution_health.get("pipeline_status") == "CRITICAL":
            alerts.append("🚨 PIPELINE CRITIQUE: Intervention requise")
        elif execution_health.get("pipeline_status") == "DEGRADED":
            alerts.append("⚠️ Pipeline dégradé")

        # Alertes temporelles
        last_exec = execution_health.get("last_execution")
        if last_exec:
            try:
                last_time = datetime.fromisoformat(last_exec.replace('Z', '+00:00'))
                if datetime.now() - last_time.replace(tzinfo=None) > timedelta(hours=24):
                    alerts.append("⚠️ Dernière exécution > 24h")
            except:
                pass

        return alerts

    def display_dashboard(self):
        """Affiche le dashboard de monitoring"""
        print("\n" + "="*80)
        print("📊 PORTFOLIO MONITORING DASHBOARD")
        print("="*80)
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Chargement des données
        strategy_metrics = self.load_strategy_metrics()
        signals_data = self.load_signals_data()
        regime_data = self.load_regime_data()

        # Calculs
        portfolio_metrics = self.calculate_portfolio_metrics(strategy_metrics)
        risk_metrics = self.analyze_risk_metrics(portfolio_metrics, strategy_metrics)
        execution_health = self.check_execution_health()
        alerts = self.generate_alerts(portfolio_metrics, risk_metrics, execution_health)

        # Sauvegarde métriques
        self.metrics = {
            "portfolio": portfolio_metrics,
            "strategies": strategy_metrics,
            "risk": risk_metrics,
            "execution": execution_health,
            "alerts": alerts
        }

        # === PORTFOLIO OVERVIEW ===
        print(f"\n🎯 PORTFOLIO OVERVIEW")
        print("-" * 50)
        if portfolio_metrics:
            print(f"  Stratégies actives: {portfolio_metrics.get('total_strategies', 0)}")
            print(f"  Total trades: {portfolio_metrics.get('total_trades', 0)}")
            print(f"  PnL total: {portfolio_metrics.get('total_pnl', 0):.4f}")
            print(f"  Sharpe moyen: {portfolio_metrics.get('avg_sharpe', 0):.3f}")
            print(f"  Meilleur Sharpe: {portfolio_metrics.get('max_sharpe', 0):.3f}")
            print(f"  Drawdown moyen: {portfolio_metrics.get('avg_max_drawdown', 0):.1%}")
            print(f"  Pire drawdown: {portfolio_metrics.get('worst_drawdown', 0):.1%}")
        else:
            print("  ❌ Aucune métrique portfolio disponible")

        # === RISK ANALYSIS ===
        print(f"\n🛡️ RISK ANALYSIS")
        print("-" * 50)
        print(f"  Niveau de risque: {risk_metrics.get('risk_level', 'UNKNOWN')}")
        print(f"  Score diversification: {risk_metrics.get('diversification_score', 0):.1%}")
        print(f"  Risque drawdown: {risk_metrics.get('drawdown_risk', 'UNKNOWN')}")
        print(f"  Stabilité performance: {risk_metrics.get('performance_stability', 0):.1%}")

        # === EXECUTION HEALTH ===
        print(f"\n⚡ EXECUTION HEALTH")
        print("-" * 50)
        print(f"  Pipeline status: {execution_health.get('pipeline_status', 'UNKNOWN')}")
        print(f"  Taux de succès: {execution_health.get('success_rate', 0):.1%}")
        print(f"  Temps exécution: {execution_health.get('execution_time', 0):.1f}s")
        if execution_health.get("last_execution"):
            print(f"  Dernière exécution: {execution_health['last_execution']}")

        # === STRATEGY BREAKDOWN ===
        print(f"\n📈 STRATEGIES BREAKDOWN")
        print("-" * 50)
        if strategy_metrics:
            for strategy_name, metrics in strategy_metrics.items():
                if isinstance(metrics, dict):
                    sharpe = metrics.get('sharpe_ratio', 0)
                    trades = metrics.get('total_trades', metrics.get('trades_count', 0))
                    win_rate = metrics.get('win_rate', 0)
                    print(f"  {strategy_name:15} | Sharpe: {sharpe:6.3f} | Trades: {trades:4d} | Win: {win_rate:5.1%}")
        else:
            print("  ❌ Aucune stratégie trouvée")

        # === ALERTS ===
        print(f"\n🚨 ALERTS ({len(alerts)})")
        print("-" * 50)
        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("  ✅ Aucune alerte")

        # === DATA SUMMARY ===
        print(f"\n📋 DATA SUMMARY")
        print("-" * 50)
        print(f"  Signaux chargés: {len(signals_data)} datasets")
        print(f"  Régimes chargés: {len(regime_data)} datasets")

        print("\n" + "="*80)

    def save_monitoring_report(self) -> str:
        """Sauvegarde le rapport de monitoring"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.logs_dir / f"portfolio_monitoring_{timestamp}.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_type": "portfolio",
            "metrics": self.metrics
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Rapport monitoring sauvé: {report_file}")
        return str(report_file)

def main():
    """Point d'entrée principal"""
    try:
        monitor = PortfolioMonitor()

        # Affichage dashboard
        monitor.display_dashboard()

        # Sauvegarde rapport
        report_file = monitor.save_monitoring_report()

        print(f"\n✅ Monitoring terminé - Rapport: {report_file}")
        return 0

    except Exception as e:
        logger.error(f"❌ Erreur monitoring: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())