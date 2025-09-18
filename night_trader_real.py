#!/usr/bin/env python3
"""
Système de Trading Nocturne AVEC DONNÉES RÉELLES
Utilise l'API Binance pour données temps réel
"""

import asyncio
import signal
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import aiohttp

# Configuration logging pour surveillance nocturne
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/night-trading/night_trader_real.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealNightTradingSystem:
    """Système de trading nocturne avec données RÉELLES Binance"""

    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.performance_log = []

        # Configuration API
        self.binance_api_url = "https://api.binance.com/api/v3/klines"

        # Statistiques de session
        self.session_stats = {
            "cycles_completed": 0,
            "total_trades": 0,
            "real_data_fetches": 0,
            "api_errors": 0,
            "uptime_start": self.start_time.isoformat()
        }

        logger.info("🌙 Système Trading Nocturne RÉEL initialisé")
        logger.info("📡 Connexion API Binance prête")

    async def _fetch_real_binance_data(self):
        """Collecte données RÉELLES de Binance"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': '1m',
                    'limit': 240  # 4 heures
                }

                async with session.get(self.binance_api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Convertir en DataFrame
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                 'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                                 'buy_quote_volume', 'ignore']

                        df = pd.DataFrame(data, columns=columns)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)

                        # Conversion numérique
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Calculer returns et vwap
                        df['ret'] = df['close'].pct_change().fillna(0)
                        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

                        current_price = df['close'].iloc[-1]
                        current_time = df.index[-1]

                        self.session_stats["real_data_fetches"] += 1

                        logger.info(f"📊 DONNÉES RÉELLES: {len(df)} points, BTC: {current_price:,.2f}$ ({current_time.strftime('%H:%M:%S')})")
                        return df, current_price

                    else:
                        self.session_stats["api_errors"] += 1
                        logger.error(f"❌ Erreur API Binance: {response.status}")
                        return None, 0

        except Exception as e:
            self.session_stats["api_errors"] += 1
            logger.error(f"❌ Erreur collecte Binance: {e}")
            return None, 0

    async def run_real_trading_cycle(self):
        """Cycle unique avec données réelles"""
        self.cycle_count += 1
        cycle_start = time.time()

        logger.info(f"🔄 CYCLE RÉEL {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Phase 1: Données RÉELLES Binance
            market_data, current_price = await self._fetch_real_binance_data()

            if market_data is None:
                logger.error("❌ Impossible d'obtenir données réelles")
                return None

            # Phase 2: Analyse rapide RL
            logger.info("🧠 Analyse avec alphas RL...")

            # Test rapide du système RL sur données réelles
            try:
                from mlpipeline.alphas.alpha_factory import AlphaFactory

                factory = AlphaFactory(market_data, "BTCUSDT")
                results = await factory.quick_alpha_scan()

                traditional_count = len(results.get('traditional_alphas', {}))
                rl_count = results.get('rl_sample', {}).get('valid_alphas', 0)

                # Score de validation basé sur données réelles
                validation_score = 0.0
                if traditional_count > 1:
                    validation_score += 0.3
                if rl_count > 0:
                    validation_score += 0.4

                # Bonus pour volatilité favorable
                recent_volatility = market_data['ret'].tail(30).std()
                if 0.001 < recent_volatility < 0.005:  # Volatilité favorable
                    validation_score += 0.3

                validated = validation_score > 0.5

                logger.info(f"🎯 Alphas: {traditional_count} traditionnels + {rl_count} RL")
                logger.info(f"✅ Validation: {validation_score:.2f}/1.0 - {'VALIDÉ' if validated else 'REJETÉ'}")

                # Simulation trades si validé
                trades_count = 0
                if validated:
                    trades_count = min(2, int(validation_score * 3))
                    for i in range(trades_count):
                        direction = "BUY" if validation_score > 0.7 else "SELL"
                        amount = 0.001 * (i + 1)
                        logger.info(f"📋 Trade RÉEL Paper #{i+1}: {direction} {amount} BTC @{current_price:,.2f}$")

                self.session_stats["total_trades"] += trades_count

                # Log performance
                cycle_duration = time.time() - cycle_start
                performance = {
                    "cycle_id": f"real_cycle_{self.cycle_count:04d}",
                    "timestamp": datetime.now().isoformat(),
                    "real_btc_price": float(current_price),
                    "market_volatility": float(recent_volatility),
                    "validation_score": validation_score,
                    "trades_executed": trades_count,
                    "traditional_alphas": traditional_count,
                    "rl_alphas": rl_count,
                    "cycle_duration": cycle_duration
                }

                self.performance_log.append(performance)
                logger.info(f"✅ Cycle {self.cycle_count} terminé - {trades_count} trades sur prix RÉEL")

                return performance

            except Exception as e:
                logger.error(f"❌ Erreur analyse RL: {e}")
                return None

        except Exception as e:
            logger.error(f"❌ Erreur cycle: {e}")
            return None

    async def save_real_session(self):
        """Sauvegarde session avec données réelles"""
        try:
            output_dir = Path("logs/night-trading")
            output_dir.mkdir(parents=True, exist_ok=True)

            session_file = output_dir / f"real_session_{self.start_time.strftime('%Y%m%d_%H%M')}.json"

            session_data = {
                "session_type": "REAL_DATA_TRADING",
                "session_stats": self.session_stats,
                "performance_log": self.performance_log,
                "total_cycles": self.cycle_count,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            }

            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            logger.info(f"💾 Session RÉELLE sauvegardée: {session_file}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")

async def test_real_system():
    """Test du système avec données réelles"""
    print("🌙 TEST SYSTÈME TRADING NOCTURNE AVEC DONNÉES RÉELLES")
    print("=" * 60)

    system = RealNightTradingSystem()

    # Test unique avec données réelles
    result = await system.run_real_trading_cycle()

    if result:
        print(f"✅ Test réussi!")
        print(f"📊 Prix BTC réel: {result['real_btc_price']:,.2f}$")
        print(f"🎯 Score validation: {result['validation_score']:.2f}")
        print(f"📋 Trades: {result['trades_executed']}")
        print(f"🧠 Alphas: {result['traditional_alphas']} + {result['rl_alphas']} RL")
    else:
        print("❌ Échec du test")

    # Sauvegarde
    await system.save_real_session()

    return system

if __name__ == "__main__":
    asyncio.run(test_real_system())