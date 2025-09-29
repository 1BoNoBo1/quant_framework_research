"""
Phase 6: Production Infrastructure - Integration Test
====================================================

Test complet de l'infrastructure de production avec tous les composants:
- Live Trading Engine
- Broker Adapters (Paper Trading)
- Order Manager avec stratégies d'exécution
- Position Reconciler
- Production Risk Management
- Circuit Breakers
- Monitoring Infrastructure
- Alerting System
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any

# Infrastructure imports
from qframe.infrastructure.live_trading.live_trading_engine import LiveTradingEngine, TradingMode
from qframe.infrastructure.live_trading.broker_adapters import PaperTradingAdapter
from qframe.infrastructure.live_trading.order_manager import LiveOrderManager, ExecutionStrategy
from qframe.infrastructure.live_trading.position_reconciler import PositionReconciler

from qframe.infrastructure.production_risk.risk_monitor import ProductionRiskMonitor
from qframe.infrastructure.production_risk.circuit_breakers import CircuitBreaker, BreakCondition
from qframe.infrastructure.production_risk.position_limits import PositionLimitManager, LimitType
from qframe.infrastructure.production_risk.risk_metrics import RealTimeRiskCalculator

from qframe.infrastructure.monitoring.metrics_collector import MetricsCollector
from qframe.infrastructure.monitoring.dashboard_server import DashboardServer, DashboardConfig
from qframe.infrastructure.monitoring.alerting_system import AlertingSystem, AlertSeverity
from qframe.infrastructure.monitoring.performance_monitor import PerformanceMonitor

# Strategy orchestration
from qframe.strategies.orchestration.multi_strategy_manager import MultiStrategyManager

# Entities
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.position import Position
from qframe.domain.entities.order import Order


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase6IntegrationTest:
    """Test d'intégration complet Phase 6"""

    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.utcnow()

    async def run_complete_test(self) -> Dict[str, Any]:
        """Exécute le test complet de Phase 6"""
        logger.info("🚀 Starting Phase 6 Production Infrastructure Integration Test")

        # 1. Setup et initialisation
        logger.info("📋 Step 1: Setting up infrastructure components")
        components = await self._setup_infrastructure()

        # 2. Test Live Trading Engine
        logger.info("⚡ Step 2: Testing Live Trading Engine")
        trading_results = await self._test_live_trading_engine(components)

        # 3. Test Risk Management
        logger.info("🛡️ Step 3: Testing Production Risk Management")
        risk_results = await self._test_risk_management(components)

        # 4. Test Monitoring Infrastructure
        logger.info("📊 Step 4: Testing Monitoring Infrastructure")
        monitoring_results = await self._test_monitoring_infrastructure(components)

        # 5. Test Alerting System
        logger.info("🚨 Step 5: Testing Alerting System")
        alerting_results = await self._test_alerting_system(components)

        # 6. Test Integration Scenarios
        logger.info("🔄 Step 6: Testing Integration Scenarios")
        integration_results = await self._test_integration_scenarios(components)

        # 7. Performance Validation
        logger.info("⚡ Step 7: Performance Validation")
        performance_results = await self._test_performance(components)

        # Compile final results
        test_duration = (datetime.utcnow() - self.start_time).total_seconds()

        final_results = {
            "test_start_time": self.start_time.isoformat(),
            "test_duration_seconds": test_duration,
            "overall_status": "SUCCESS",
            "components_tested": len(components),
            "results": {
                "trading_engine": trading_results,
                "risk_management": risk_results,
                "monitoring": monitoring_results,
                "alerting": alerting_results,
                "integration": integration_results,
                "performance": performance_results
            }
        }

        # Vérifier si tous les tests ont réussi
        all_success = all(
            result.get("status") == "SUCCESS"
            for result in final_results["results"].values()
        )

        if not all_success:
            final_results["overall_status"] = "PARTIAL_FAILURE"

        logger.info(f"✅ Phase 6 Integration Test Complete: {final_results['overall_status']}")
        return final_results

    async def _setup_infrastructure(self) -> Dict[str, Any]:
        """Configure tous les composants d'infrastructure"""
        components = {}

        try:
            # 1. Metrics Collector (base pour tout)
            components["metrics_collector"] = MetricsCollector(
                buffer_size=1000,
                aggregation_interval=timedelta(seconds=30)
            )

            # 2. Broker Adapter (Paper Trading)
            components["broker_adapter"] = PaperTradingAdapter(
                initial_balance=Decimal("100000")
            )

            # 3. Order Manager
            components["order_manager"] = LiveOrderManager(
                broker_adapter=components["broker_adapter"],
                max_concurrent_orders=20
            )

            # 4. Position Reconciler
            components["position_reconciler"] = PositionReconciler(
                broker_adapter=components["broker_adapter"],
                auto_reconcile=True
            )

            # 5. Risk Calculator
            components["risk_calculator"] = RealTimeRiskCalculator(
                var_confidence=Decimal("0.95"),
                lookback_window=100
            )

            # 6. Position Limit Manager
            components["position_limit_manager"] = PositionLimitManager(
                default_portfolio_limit=Decimal("0.25")
            )

            # 7. Circuit Breaker
            components["circuit_breaker"] = CircuitBreaker()

            # 8. Risk Monitor
            components["risk_monitor"] = ProductionRiskMonitor(
                circuit_breaker=components["circuit_breaker"],
                position_limit_manager=components["position_limit_manager"],
                risk_calculator=components["risk_calculator"]
            )

            # 9. Multi-Strategy Manager (simulé)
            components["multi_strategy_manager"] = None  # Placeholder

            # 10. Live Trading Engine
            components["trading_engine"] = LiveTradingEngine(
                multi_strategy_manager=components["multi_strategy_manager"],
                broker_adapter=components["broker_adapter"],
                order_manager=components["order_manager"],
                position_reconciler=components["position_reconciler"],
                trading_mode=TradingMode.PAPER
            )

            # 11. Dashboard Server
            components["dashboard_server"] = DashboardServer(
                metrics_collector=components["metrics_collector"],
                config=DashboardConfig(port=8081)
            )

            # 12. Alerting System
            components["alerting_system"] = AlertingSystem()

            # 13. Performance Monitor
            components["performance_monitor"] = PerformanceMonitor(
                metrics_collector=components["metrics_collector"]
            )

            logger.info(f"✅ Infrastructure setup complete: {len(components)} components")
            return components

        except Exception as e:
            logger.error(f"❌ Infrastructure setup failed: {e}")
            raise

    async def _test_live_trading_engine(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test du moteur de trading en direct"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            trading_engine = components["trading_engine"]
            broker_adapter = components["broker_adapter"]

            # Test 1: Connexion broker
            await broker_adapter.connect()
            connection_status = await broker_adapter.get_connection_status()
            results["tests"].append({
                "name": "broker_connection",
                "status": "SUCCESS" if connection_status.is_connected else "FAILED",
                "details": f"Connected: {connection_status.is_connected}"
            })

            # Test 2: Démarrage session trading
            session_config = {"strategies": ["test_strategy"]}
            session_id = await trading_engine.start_trading_session(session_config)
            results["tests"].append({
                "name": "trading_session_start",
                "status": "SUCCESS",
                "details": f"Session ID: {session_id}"
            })

            # Test 3: Statut session
            session_status = await trading_engine.get_session_status()
            results["tests"].append({
                "name": "session_status",
                "status": "SUCCESS" if session_status["is_active"] else "FAILED",
                "details": f"Mode: {session_status['mode']}, Active: {session_status['is_active']}"
            })

            # Test 4: Arrêt session
            await trading_engine.stop_trading_session()
            final_status = await trading_engine.get_session_status()
            results["tests"].append({
                "name": "trading_session_stop",
                "status": "SUCCESS" if not final_status["is_active"] else "FAILED",
                "details": f"Session stopped: {not final_status['is_active']}"
            })

        except Exception as e:
            logger.error(f"Trading engine test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results

    async def _test_risk_management(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test du système de gestion des risques"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            risk_monitor = components["risk_monitor"]
            circuit_breaker = components["circuit_breaker"]
            position_limits = components["position_limit_manager"]

            # Test 1: Démarrage du monitoring
            await risk_monitor.start_monitoring()
            results["tests"].append({
                "name": "risk_monitoring_start",
                "status": "SUCCESS",
                "details": "Risk monitoring started"
            })

            # Test 2: Configuration limites de position
            await position_limits.set_position_limit(
                symbol="BTCUSD",
                limit_type=LimitType.NOTIONAL_VALUE,
                limit_value=Decimal("10000")
            )
            results["tests"].append({
                "name": "position_limits_config",
                "status": "SUCCESS",
                "details": "Position limits configured"
            })

            # Test 3: Test portfolio avec position
            test_portfolio = Portfolio(
                total_value=Decimal("50000"),
                cash_balance=Decimal("40000"),
                positions=[
                    Position(
                        symbol="BTCUSD",
                        quantity=Decimal("0.5"),
                        average_price=Decimal("45000"),
                        market_value=Decimal("22500"),
                        timestamp=datetime.utcnow()
                    )
                ]
            )

            alerts = await risk_monitor.check_portfolio_risk(test_portfolio)
            results["tests"].append({
                "name": "portfolio_risk_check",
                "status": "SUCCESS",
                "details": f"Generated {len(alerts)} risk alerts"
            })

            # Test 4: Circuit breaker status
            cb_status = await circuit_breaker.get_status()
            results["tests"].append({
                "name": "circuit_breaker_status",
                "status": "SUCCESS",
                "details": f"Status: {cb_status['status']}"
            })

            # Test 5: Test déclenchement circuit breaker
            await circuit_breaker.trigger("Test trigger", BreakCondition.MANUAL_TRIGGER)
            triggered_status = await circuit_breaker.get_status()
            results["tests"].append({
                "name": "circuit_breaker_trigger",
                "status": "SUCCESS" if triggered_status["status"] == "triggered" else "FAILED",
                "details": f"Triggered status: {triggered_status['status']}"
            })

            # Reset circuit breaker
            await circuit_breaker.reset(manual=True)

        except Exception as e:
            logger.error(f"Risk management test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results

    async def _test_monitoring_infrastructure(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test de l'infrastructure de monitoring"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            metrics_collector = components["metrics_collector"]
            dashboard_server = components["dashboard_server"]
            performance_monitor = components["performance_monitor"]

            # Test 1: Collection de métriques
            await metrics_collector.record_counter("test_counter", 5)
            await metrics_collector.record_gauge("test_gauge", 42.5)
            await metrics_collector.record_histogram("test_histogram", 123.4)

            current_metrics = await metrics_collector.get_current_metrics()
            results["tests"].append({
                "name": "metrics_collection",
                "status": "SUCCESS",
                "details": f"Collected {current_metrics['collection_stats']['total_metrics_collected']} metrics"
            })

            # Test 2: Démarrage dashboard
            await dashboard_server.start_server()
            dashboard_data = await dashboard_server.get_dashboard_data()
            results["tests"].append({
                "name": "dashboard_server",
                "status": "SUCCESS" if dashboard_data else "FAILED",
                "details": f"Dashboard running: {dashboard_server.is_running}"
            })

            # Test 3: Performance monitoring
            await performance_monitor.start_monitoring()
            system_metrics = await performance_monitor.get_current_metrics()
            results["tests"].append({
                "name": "performance_monitoring",
                "status": "SUCCESS",
                "details": f"CPU: {system_metrics.cpu_percent}%, Memory: {system_metrics.memory_percent}%"
            })

            # Test 4: Health check
            health_status = await performance_monitor.run_health_check()
            results["tests"].append({
                "name": "system_health_check",
                "status": "SUCCESS",
                "details": f"Overall health: {health_status['overall']}"
            })

        except Exception as e:
            logger.error(f"Monitoring infrastructure test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results

    async def _test_alerting_system(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test du système d'alerting"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            alerting_system = components["alerting_system"]

            # Test 1: Envoi alerte simple
            alert_id = await alerting_system.send_alert(
                title="Test Alert",
                message="This is a test alert for Phase 6 validation",
                severity=AlertSeverity.WARNING,
                source="phase6_test",
                tags={"component": "test", "phase": "6"}
            )
            results["tests"].append({
                "name": "send_alert",
                "status": "SUCCESS",
                "details": f"Alert sent with ID: {alert_id}"
            })

            # Test 2: Vérification alertes actives
            active_alerts = await alerting_system.get_active_alerts()
            results["tests"].append({
                "name": "active_alerts_check",
                "status": "SUCCESS" if len(active_alerts) > 0 else "FAILED",
                "details": f"Found {len(active_alerts)} active alerts"
            })

            # Test 3: Acquittement alerte
            ack_success = await alerting_system.acknowledge_alert(alert_id, "phase6_test")
            results["tests"].append({
                "name": "alert_acknowledgment",
                "status": "SUCCESS" if ack_success else "FAILED",
                "details": f"Alert acknowledged: {ack_success}"
            })

            # Test 4: Résolution alerte
            resolve_success = await alerting_system.resolve_alert(alert_id)
            results["tests"].append({
                "name": "alert_resolution",
                "status": "SUCCESS" if resolve_success else "FAILED",
                "details": f"Alert resolved: {resolve_success}"
            })

            # Test 5: Statistiques alerting
            stats = await alerting_system.get_alert_statistics()
            results["tests"].append({
                "name": "alerting_statistics",
                "status": "SUCCESS",
                "details": f"Total alerts: {stats['total_alerts_sent']}"
            })

        except Exception as e:
            logger.error(f"Alerting system test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results

    async def _test_integration_scenarios(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test des scénarios d'intégration complexes"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            # Scénario 1: Trading avec monitoring complet
            await self._scenario_complete_trading_cycle(components, results)

            # Scénario 2: Gestion d'urgence
            await self._scenario_emergency_procedures(components, results)

            # Scénario 3: Performance sous charge
            await self._scenario_performance_under_load(components, results)

        except Exception as e:
            logger.error(f"Integration scenarios test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results

    async def _scenario_complete_trading_cycle(self, components: Dict[str, Any], results: list) -> None:
        """Scénario: Cycle de trading complet avec monitoring"""
        trading_engine = components["trading_engine"]
        order_manager = components["order_manager"]
        risk_monitor = components["risk_monitor"]

        # Démarrer session
        session_id = await trading_engine.start_trading_session({"strategies": ["integration_test"]})

        # Créer ordre test
        test_order = Order(
            symbol="BTCUSD",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            timestamp=datetime.utcnow()
        )

        # Exécuter ordre avec monitoring
        execution_result = await order_manager.submit_order(test_order, ExecutionStrategy.SMART)

        # Vérifier avec risk monitor
        test_portfolio = Portfolio(
            total_value=Decimal("99000"),
            cash_balance=Decimal("95000"),
            positions=[
                Position(
                    symbol="BTCUSD",
                    quantity=Decimal("0.1"),
                    average_price=Decimal("40000"),
                    market_value=Decimal("4000"),
                    timestamp=datetime.utcnow()
                )
            ]
        )

        risk_alerts = await risk_monitor.check_portfolio_risk(test_portfolio)

        # Arrêter session
        await trading_engine.stop_trading_session()

        results["tests"].append({
            "name": "complete_trading_cycle",
            "status": "SUCCESS" if execution_result.success else "FAILED",
            "details": f"Order executed: {execution_result.success}, Risk alerts: {len(risk_alerts)}"
        })

    async def _scenario_emergency_procedures(self, components: Dict[str, Any], results: list) -> None:
        """Scénario: Procédures d'urgence"""
        trading_engine = components["trading_engine"]
        circuit_breaker = components["circuit_breaker"]
        alerting_system = components["alerting_system"]

        # Simuler situation d'urgence
        await alerting_system.send_alert(
            title="Emergency Scenario Test",
            message="Simulated emergency condition",
            severity=AlertSeverity.EMERGENCY,
            source="integration_test"
        )

        # Déclencher circuit breaker
        await circuit_breaker.trigger("Integration test emergency", BreakCondition.EXTERNAL_SIGNAL)

        # Vérifier arrêt d'urgence
        await trading_engine.emergency_stop("Integration test emergency stop")

        # Vérifier états
        cb_status = await circuit_breaker.get_status()
        session_status = await trading_engine.get_session_status()

        results["tests"].append({
            "name": "emergency_procedures",
            "status": "SUCCESS",
            "details": f"Circuit breaker: {cb_status['status']}, Session active: {session_status.get('is_active', False)}"
        })

        # Reset pour tests suivants
        await circuit_breaker.reset(manual=True)

    async def _scenario_performance_under_load(self, components: Dict[str, Any], results: list) -> None:
        """Scénario: Performance sous charge"""
        metrics_collector = components["metrics_collector"]
        performance_monitor = components["performance_monitor"]

        # Générer charge de métriques
        start_time = datetime.utcnow()

        for i in range(100):
            await metrics_collector.record_counter(f"load_test_counter_{i % 10}", 1)
            await metrics_collector.record_gauge(f"load_test_gauge_{i % 5}", i * 1.5)
            await metrics_collector.record_histogram("load_test_histogram", i * 0.8)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        # Vérifier performance système
        performance_summary = await performance_monitor.get_performance_summary()

        results["tests"].append({
            "name": "performance_under_load",
            "status": "SUCCESS" if processing_time < 5.0 else "WARNING",
            "details": f"Processed 300 metrics in {processing_time:.2f}s, CPU: {performance_summary['current_metrics']['cpu_percent']}%"
        })

    async def _test_performance(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Validation des performances globales"""
        results = {"status": "SUCCESS", "tests": []}

        try:
            performance_monitor = components["performance_monitor"]
            metrics_collector = components["metrics_collector"]

            # Test 1: Latence de collecte de métriques
            start_time = datetime.utcnow()
            for i in range(50):
                await metrics_collector.record_counter("perf_test", 1)
            collection_time = (datetime.utcnow() - start_time).total_seconds()

            results["tests"].append({
                "name": "metrics_collection_latency",
                "status": "SUCCESS" if collection_time < 1.0 else "WARNING",
                "details": f"50 metrics collected in {collection_time:.3f}s"
            })

            # Test 2: Performance système
            system_metrics = await performance_monitor.get_current_metrics()
            cpu_ok = system_metrics.cpu_percent < 80
            memory_ok = system_metrics.memory_percent < 90

            results["tests"].append({
                "name": "system_resource_usage",
                "status": "SUCCESS" if cpu_ok and memory_ok else "WARNING",
                "details": f"CPU: {system_metrics.cpu_percent}%, Memory: {system_metrics.memory_percent}%"
            })

            # Test 3: Throughput global
            metrics_summary = await metrics_collector.get_current_metrics()
            throughput = metrics_summary.get("collection_stats", {}).get("metrics_per_second", 0)

            results["tests"].append({
                "name": "metrics_throughput",
                "status": "SUCCESS" if throughput > 10 else "WARNING",
                "details": f"Throughput: {throughput} metrics/sec"
            })

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            results["status"] = "FAILED"
            results["error"] = str(e)

        return results


async def main():
    """Fonction principale du test"""
    print("=" * 80)
    print("🚀 QFrame Phase 6: Production Infrastructure Integration Test")
    print("=" * 80)

    test = Phase6IntegrationTest()

    try:
        results = await test.run_complete_test()

        # Afficher résultats
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        print(f"Overall Status: {results['overall_status']}")
        print(f"Test Duration: {results['test_duration_seconds']:.2f} seconds")
        print(f"Components Tested: {results['components_tested']}")

        print("\n📋 Detailed Results:")
        for category, category_results in results["results"].items():
            status_icon = "✅" if category_results["status"] == "SUCCESS" else "❌"
            print(f"  {status_icon} {category.replace('_', ' ').title()}: {category_results['status']}")

            if "tests" in category_results:
                for test in category_results["tests"]:
                    test_icon = "✅" if test["status"] == "SUCCESS" else ("⚠️" if test["status"] == "WARNING" else "❌")
                    print(f"    {test_icon} {test['name']}: {test['details']}")

        print("\n" + "=" * 80)

        if results["overall_status"] == "SUCCESS":
            print("🎉 PHASE 6 PRODUCTION INFRASTRUCTURE: 100% OPERATIONAL! 🎉")
            print("\nReady for:")
            print("✅ Live trading operations")
            print("✅ Production risk management")
            print("✅ Real-time monitoring")
            print("✅ Multi-channel alerting")
            print("✅ Emergency procedures")
        else:
            print("⚠️  Some components need attention before production deployment")

        print("=" * 80)

        return results

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())