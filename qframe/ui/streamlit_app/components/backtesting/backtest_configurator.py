"""
Backtest Configurator - Composant pour configuration de backtests
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

class BacktestConfigurator:
    """Configurateur de backtests avec validation et templates."""

    def __init__(self):
        self.strategy_templates = self._load_strategy_templates()
        self.configuration_presets = self._load_configuration_presets()

    def _load_strategy_templates(self) -> Dict:
        """Charge les templates de stratégies disponibles."""
        return {
            "DMN LSTM Strategy": {
                "description": "Deep Market Networks avec LSTM et attention",
                "default_params": {
                    "window_size": 64,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "use_attention": True,
                    "learning_rate": 0.001,
                    "signal_threshold": 0.1
                },
                "asset_types": ["Crypto", "Stocks", "Forex"],
                "min_history": 1000,
                "recommended_timeframes": ["1h", "4h", "1d"]
            },
            "Adaptive Mean Reversion": {
                "description": "Mean reversion avec détection de régimes",
                "default_params": {
                    "lookback_short": 10,
                    "lookback_long": 50,
                    "z_entry_base": 1.0,
                    "z_exit_base": 0.2,
                    "regime_window": 252,
                    "use_ml_optimization": True
                },
                "asset_types": ["Crypto", "Stocks"],
                "min_history": 500,
                "recommended_timeframes": ["15m", "1h", "4h"]
            },
            "Funding Arbitrage": {
                "description": "Arbitrage de taux de financement crypto",
                "default_params": {
                    "funding_threshold": 0.01,
                    "spread_threshold": 0.005,
                    "max_exposure": 0.5,
                    "rebalance_hours": 8
                },
                "asset_types": ["Crypto"],
                "min_history": 200,
                "recommended_timeframes": ["1h", "8h"]
            },
            "RL Alpha Generator": {
                "description": "Génération d'alphas via reinforcement learning",
                "default_params": {
                    "agent_type": "PPO",
                    "learning_rate": 0.0003,
                    "max_formula_depth": 4,
                    "reward_function": "IC",
                    "complexity_penalty": 0.1
                },
                "asset_types": ["Crypto", "Stocks"],
                "min_history": 1000,
                "recommended_timeframes": ["1h", "4h", "1d"]
            },
            "Grid Trading": {
                "description": "Trading sur grille avec espacement adaptatif",
                "default_params": {
                    "grid_spacing": 0.02,
                    "num_levels": 10,
                    "position_size": 0.1,
                    "rebalance_threshold": 0.05
                },
                "asset_types": ["Crypto", "Stocks"],
                "min_history": 300,
                "recommended_timeframes": ["15m", "1h"]
            },
            "Simple Moving Average": {
                "description": "Stratégie croisement moyennes mobiles",
                "default_params": {
                    "fast_period": 10,
                    "slow_period": 30,
                    "signal_threshold": 0.001
                },
                "asset_types": ["Crypto", "Stocks", "Forex"],
                "min_history": 100,
                "recommended_timeframes": ["1h", "4h", "1d"]
            }
        }

    def _load_configuration_presets(self) -> Dict:
        """Charge les presets de configuration."""
        return {
            "Conservative Portfolio": {
                "initial_capital": 100000,
                "commission_rate": 0.1,
                "slippage_rate": 0.05,
                "max_leverage": 1.0,
                "position_size_limit": 50,
                "risk_free_rate": 3.0,
                "rebalance_frequency": "Daily"
            },
            "Aggressive Trading": {
                "initial_capital": 50000,
                "commission_rate": 0.05,
                "slippage_rate": 0.03,
                "max_leverage": 3.0,
                "position_size_limit": 100,
                "risk_free_rate": 3.0,
                "rebalance_frequency": "Every Trade"
            },
            "Crypto Focus": {
                "initial_capital": 25000,
                "commission_rate": 0.1,
                "slippage_rate": 0.1,
                "max_leverage": 2.0,
                "position_size_limit": 80,
                "risk_free_rate": 5.0,
                "rebalance_frequency": "Every Trade"
            },
            "Multi-Asset Balanced": {
                "initial_capital": 200000,
                "commission_rate": 0.08,
                "slippage_rate": 0.04,
                "max_leverage": 1.5,
                "position_size_limit": 60,
                "risk_free_rate": 3.5,
                "rebalance_frequency": "Daily"
            },
            "High-Frequency": {
                "initial_capital": 100000,
                "commission_rate": 0.02,
                "slippage_rate": 0.01,
                "max_leverage": 5.0,
                "position_size_limit": 100,
                "risk_free_rate": 3.0,
                "rebalance_frequency": "Every Trade"
            }
        }

    def render_strategy_selection(self):
        """Rendu de la sélection de stratégie avec templates."""
        st.subheader("🎯 Strategy Selection")

        col_strat1, col_strat2 = st.columns([2, 1])

        with col_strat1:
            strategy_type = st.selectbox(
                "Select Strategy Type",
                list(self.strategy_templates.keys()),
                help="Choisir le type de stratégie à tester"
            )

            # Affichage de la description
            if strategy_type:
                template = self.strategy_templates[strategy_type]
                st.info(f"📝 {template['description']}")

                # Informations sur la stratégie
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Asset Types", len(template['asset_types']))
                with col_info2:
                    st.metric("Min History", f"{template['min_history']} periods")
                with col_info3:
                    st.metric("Timeframes", len(template['recommended_timeframes']))

        with col_strat2:
            st.markdown("### Strategy Info")
            if strategy_type:
                template = self.strategy_templates[strategy_type]

                st.markdown("**Supported Assets:**")
                for asset_type in template['asset_types']:
                    st.markdown(f"• {asset_type}")

                st.markdown("**Recommended Timeframes:**")
                for tf in template['recommended_timeframes']:
                    st.markdown(f"• {tf}")

        return strategy_type

    def render_strategy_parameters(self, strategy_type: str):
        """Rendu des paramètres de stratégie."""
        if not strategy_type or strategy_type not in self.strategy_templates:
            return {}

        template = self.strategy_templates[strategy_type]
        default_params = template['default_params']

        st.subheader("⚙️ Strategy Parameters")

        with st.expander("Configure Parameters", expanded=True):
            params = {}

            # Génération dynamique des paramètres selon le type
            if strategy_type == "DMN LSTM Strategy":
                col_p1, col_p2, col_p3 = st.columns(3)

                with col_p1:
                    params['window_size'] = st.slider(
                        "Window Size",
                        min_value=10, max_value=200,
                        value=default_params['window_size']
                    )
                    params['hidden_size'] = st.selectbox(
                        "Hidden Size",
                        [64, 128, 256, 512],
                        index=[64, 128, 256, 512].index(default_params['hidden_size'])
                    )

                with col_p2:
                    params['num_layers'] = st.slider(
                        "Number of Layers",
                        min_value=1, max_value=5,
                        value=default_params['num_layers']
                    )
                    params['dropout'] = st.slider(
                        "Dropout Rate",
                        min_value=0.0, max_value=0.5,
                        value=default_params['dropout']
                    )

                with col_p3:
                    params['use_attention'] = st.checkbox(
                        "Use Attention",
                        value=default_params['use_attention']
                    )
                    params['learning_rate'] = st.number_input(
                        "Learning Rate",
                        min_value=0.0001, max_value=0.01,
                        value=default_params['learning_rate'],
                        format="%.4f"
                    )
                    params['signal_threshold'] = st.slider(
                        "Signal Threshold",
                        min_value=0.01, max_value=0.5,
                        value=default_params['signal_threshold']
                    )

            elif strategy_type == "Adaptive Mean Reversion":
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    params['lookback_short'] = st.number_input(
                        "Short Lookback",
                        min_value=5, max_value=50,
                        value=default_params['lookback_short']
                    )
                    params['lookback_long'] = st.number_input(
                        "Long Lookback",
                        min_value=20, max_value=200,
                        value=default_params['lookback_long']
                    )
                    params['z_entry_base'] = st.slider(
                        "Z Entry Threshold",
                        min_value=0.5, max_value=3.0,
                        value=default_params['z_entry_base']
                    )

                with col_p2:
                    params['z_exit_base'] = st.slider(
                        "Z Exit Threshold",
                        min_value=0.1, max_value=1.0,
                        value=default_params['z_exit_base']
                    )
                    params['regime_window'] = st.number_input(
                        "Regime Window",
                        min_value=100, max_value=500,
                        value=default_params['regime_window']
                    )
                    params['use_ml_optimization'] = st.checkbox(
                        "ML Optimization",
                        value=default_params['use_ml_optimization']
                    )

            elif strategy_type == "Funding Arbitrage":
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    params['funding_threshold'] = st.slider(
                        "Funding Threshold (%)",
                        min_value=0.001, max_value=0.05,
                        value=default_params['funding_threshold']
                    )
                    params['spread_threshold'] = st.slider(
                        "Spread Threshold (%)",
                        min_value=0.001, max_value=0.02,
                        value=default_params['spread_threshold']
                    )

                with col_p2:
                    params['max_exposure'] = st.slider(
                        "Max Exposure",
                        min_value=0.1, max_value=1.0,
                        value=default_params['max_exposure']
                    )
                    params['rebalance_hours'] = st.selectbox(
                        "Rebalance Hours",
                        [1, 4, 8, 24],
                        index=[1, 4, 8, 24].index(default_params['rebalance_hours'])
                    )

            elif strategy_type == "RL Alpha Generator":
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    params['agent_type'] = st.selectbox(
                        "Agent Type",
                        ["PPO", "A2C", "DQN", "SAC"],
                        index=["PPO", "A2C", "DQN", "SAC"].index(default_params['agent_type'])
                    )
                    params['learning_rate'] = st.number_input(
                        "Learning Rate",
                        min_value=0.0001, max_value=0.01,
                        value=default_params['learning_rate'],
                        format="%.4f"
                    )
                    params['max_formula_depth'] = st.slider(
                        "Max Formula Depth",
                        min_value=2, max_value=8,
                        value=default_params['max_formula_depth']
                    )

                with col_p2:
                    params['reward_function'] = st.selectbox(
                        "Reward Function",
                        ["IC", "Rank IC", "Sharpe", "Combined"],
                        index=["IC", "Rank IC", "Sharpe", "Combined"].index(default_params['reward_function'])
                    )
                    params['complexity_penalty'] = st.slider(
                        "Complexity Penalty",
                        min_value=0.0, max_value=0.5,
                        value=default_params['complexity_penalty']
                    )

            elif strategy_type == "Grid Trading":
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    params['grid_spacing'] = st.slider(
                        "Grid Spacing (%)",
                        min_value=0.005, max_value=0.1,
                        value=default_params['grid_spacing']
                    )
                    params['num_levels'] = st.slider(
                        "Number of Levels",
                        min_value=5, max_value=20,
                        value=default_params['num_levels']
                    )

                with col_p2:
                    params['position_size'] = st.slider(
                        "Position Size per Level",
                        min_value=0.05, max_value=0.5,
                        value=default_params['position_size']
                    )
                    params['rebalance_threshold'] = st.slider(
                        "Rebalance Threshold",
                        min_value=0.01, max_value=0.2,
                        value=default_params['rebalance_threshold']
                    )

            elif strategy_type == "Simple Moving Average":
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    params['fast_period'] = st.number_input(
                        "Fast MA Period",
                        min_value=5, max_value=50,
                        value=default_params['fast_period']
                    )
                    params['slow_period'] = st.number_input(
                        "Slow MA Period",
                        min_value=20, max_value=200,
                        value=default_params['slow_period']
                    )

                with col_p2:
                    params['signal_threshold'] = st.slider(
                        "Signal Threshold",
                        min_value=0.0001, max_value=0.01,
                        value=default_params['signal_threshold'],
                        format="%.4f"
                    )

        return params

    def render_universe_selection(self, strategy_type: str):
        """Rendu de la sélection d'univers d'actifs."""
        st.subheader("🌍 Asset Universe")

        if not strategy_type or strategy_type not in self.strategy_templates:
            supported_types = ["Crypto", "Stocks", "Forex"]
        else:
            supported_types = self.strategy_templates[strategy_type]['asset_types']

        # Sélection par type d'actif
        asset_categories = {}

        for asset_type in supported_types:
            with st.expander(f"{asset_type} Assets", expanded=asset_type == "Crypto"):
                if asset_type == "Crypto":
                    available_assets = [
                        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
                        "XRP/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT"
                    ]
                elif asset_type == "Stocks":
                    available_assets = [
                        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
                        "META", "NVDA", "NFLX", "SPY", "QQQ"
                    ]
                elif asset_type == "Forex":
                    available_assets = [
                        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
                        "AUD/USD", "USD/CAD", "NZD/USD"
                    ]

                selected_assets = st.multiselect(
                    f"Select {asset_type} Assets",
                    available_assets,
                    default=available_assets[:2] if asset_type == "Crypto" else [],
                    key=f"assets_{asset_type}"
                )

                asset_categories[asset_type] = selected_assets

        # Flatten selected assets
        all_selected = []
        for assets in asset_categories.values():
            all_selected.extend(assets)

        # Timeframe selection
        col_tf1, col_tf2 = st.columns(2)

        with col_tf1:
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
                index=3,  # Default to 1h
                help="Période temporelle des données"
            )

        with col_tf2:
            data_source = st.selectbox(
                "Data Source",
                ["Binance", "CCXT Multi-Exchange", "Yahoo Finance", "Local CSV"],
                help="Source des données historiques"
            )

        return {
            'selected_assets': all_selected,
            'asset_categories': asset_categories,
            'timeframe': timeframe,
            'data_source': data_source
        }

    def render_period_selection(self):
        """Rendu de la sélection de période."""
        st.subheader("📅 Backtest Period")

        col_period1, col_period2, col_period3 = st.columns(3)

        with col_period1:
            start_date = st.date_input(
                "Start Date",
                value=date(2023, 1, 1),
                min_value=date(2020, 1, 1),
                max_value=date.today() - timedelta(days=30),
                help="Date de début du backtest"
            )

        with col_period2:
            end_date = st.date_input(
                "End Date",
                value=date(2024, 1, 1),
                min_value=start_date + timedelta(days=30) if 'start_date' in locals() else date(2023, 2, 1),
                max_value=date.today(),
                help="Date de fin du backtest"
            )

        with col_period3:
            # Quick period selection
            quick_periods = {
                "Last 3 Months": timedelta(days=90),
                "Last 6 Months": timedelta(days=180),
                "Last Year": timedelta(days=365),
                "Last 2 Years": timedelta(days=730),
                "Custom": None
            }

            selected_period = st.selectbox("Quick Select", list(quick_periods.keys()), index=4)

            if selected_period != "Custom" and quick_periods[selected_period]:
                end_date = date.today()
                start_date = end_date - quick_periods[selected_period]

        # Period validation
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return None

        period_days = (end_date - start_date).days
        if period_days < 30:
            st.warning("Period is very short. Consider extending for better results.")

        return {
            'start_date': start_date,
            'end_date': end_date,
            'period_days': period_days
        }

    def render_trading_parameters(self):
        """Rendu des paramètres de trading."""
        st.subheader("💰 Trading Parameters")

        col_trading1, col_trading2 = st.columns(2)

        with col_trading1:
            st.markdown("### Capital & Position")

            initial_capital = st.number_input(
                "Initial Capital (USD)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                help="Capital initial pour le backtest"
            )

            position_size_limit = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=100,
                value=100,
                help="Taille maximale de position en % du capital"
            )

            max_leverage = st.number_input(
                "Maximum Leverage",
                min_value=1.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Levier maximum autorisé"
            )

        with col_trading2:
            st.markdown("### Costs & Constraints")

            commission_rate = st.number_input(
                "Commission Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="Frais de transaction en %"
            )

            slippage_rate = st.number_input(
                "Slippage Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.3f",
                help="Slippage estimé en %"
            )

            rebalance_frequency = st.selectbox(
                "Rebalancing Frequency",
                ["Every Trade", "Daily", "Weekly", "Monthly"],
                index=1,
                help="Fréquence de rééquilibrage du portfolio"
            )

        return {
            'initial_capital': initial_capital,
            'position_size_limit': position_size_limit,
            'max_leverage': max_leverage,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'rebalance_frequency': rebalance_frequency
        }

    def render_benchmark_selection(self):
        """Rendu de la sélection de benchmarks."""
        st.subheader("📈 Benchmarks")

        benchmark_options = {
            "BTC Buy & Hold": "Simple achat et conservation de Bitcoin",
            "ETH Buy & Hold": "Simple achat et conservation d'Ethereum",
            "SPY (S&P 500)": "ETF S&P 500 pour comparaison actions",
            "60/40 Portfolio": "60% actions, 40% obligations",
            "Equal Weight Crypto": "Panier équipondéré crypto",
            "Custom Strategy": "Stratégie personnalisée"
        }

        selected_benchmarks = st.multiselect(
            "Select Benchmarks",
            list(benchmark_options.keys()),
            default=["BTC Buy & Hold"],
            help="Sélectionner les benchmarks pour comparaison"
        )

        # Configuration custom benchmark
        custom_config = {}
        if "Custom Strategy" in selected_benchmarks:
            with st.expander("Custom Benchmark Configuration"):
                custom_config['name'] = st.text_input("Benchmark Name")
                custom_config['description'] = st.text_area("Description")
                custom_config['assets'] = st.text_input("Assets (comma-separated)")

        return {
            'selected_benchmarks': selected_benchmarks,
            'benchmark_descriptions': {k: benchmark_options[k] for k in selected_benchmarks},
            'custom_config': custom_config if "Custom Strategy" in selected_benchmarks else None
        }

    def render_advanced_options(self):
        """Rendu des options avancées."""
        st.subheader("🔧 Advanced Options")

        with st.expander("Optimization Settings"):
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                optimization_mode = st.selectbox(
                    "Parameter Optimization",
                    ["None", "Grid Search", "Random Search", "Bayesian Optimization"],
                    help="Méthode d'optimisation des paramètres"
                )

                if optimization_mode != "None":
                    optimization_metric = st.selectbox(
                        "Optimization Metric",
                        ["Sharpe Ratio", "Total Return", "Calmar Ratio", "Sortino Ratio"],
                        help="Métrique à optimiser"
                    )

            with col_opt2:
                walk_forward = st.checkbox(
                    "Walk-Forward Analysis",
                    help="Effectuer une analyse walk-forward"
                )

                if walk_forward:
                    wf_window = st.number_input(
                        "WF Window (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        help="Taille de la fenêtre walk-forward"
                    )

                    wf_step = st.number_input(
                        "WF Step (days)",
                        min_value=1,
                        max_value=90,
                        value=30,
                        help="Pas de la fenêtre walk-forward"
                    )

        with st.expander("Risk Settings"):
            col_risk1, col_risk2 = st.columns(2)

            with col_risk1:
                risk_free_rate = st.number_input(
                    "Risk-Free Rate (%/year)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                    help="Taux sans risque pour calcul Sharpe"
                )

                confidence_level = st.slider(
                    "VaR Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Niveau de confiance pour VaR/CVaR"
                )

            with col_risk2:
                max_drawdown_limit = st.slider(
                    "Max Drawdown Limit (%)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Limite de drawdown pour arrêt"
                )

                stop_loss_enabled = st.checkbox(
                    "Enable Stop-Loss",
                    help="Activer arrêt automatique en cas de forte perte"
                )

        return {
            'optimization_mode': optimization_mode,
            'optimization_metric': optimization_metric if optimization_mode != "None" else None,
            'walk_forward': walk_forward,
            'wf_window': wf_window if walk_forward else None,
            'wf_step': wf_step if walk_forward else None,
            'risk_free_rate': risk_free_rate,
            'confidence_level': confidence_level,
            'max_drawdown_limit': max_drawdown_limit,
            'stop_loss_enabled': stop_loss_enabled
        }

    def load_configuration_preset(self, preset_name: str):
        """Charge un preset de configuration."""
        if preset_name in self.configuration_presets:
            return self.configuration_presets[preset_name]
        return None

    def validate_configuration(self, config: Dict) -> Tuple[bool, List[str]]:
        """Valide une configuration de backtest."""
        errors = []

        # Validation des dates
        if 'period' in config and config['period']:
            if config['period']['start_date'] >= config['period']['end_date']:
                errors.append("Start date must be before end date")

            if config['period']['period_days'] < 30:
                errors.append("Backtest period is too short (minimum 30 days)")

        # Validation des actifs
        if 'universe' in config and config['universe']:
            if not config['universe']['selected_assets']:
                errors.append("At least one asset must be selected")

        # Validation des paramètres de trading
        if 'trading' in config and config['trading']:
            if config['trading']['initial_capital'] <= 0:
                errors.append("Initial capital must be positive")

            if config['trading']['commission_rate'] < 0:
                errors.append("Commission rate cannot be negative")

            if config['trading']['slippage_rate'] < 0:
                errors.append("Slippage rate cannot be negative")

        # Validation de la stratégie
        if 'strategy_type' not in config:
            errors.append("Strategy type must be selected")

        return len(errors) == 0, errors

    def export_configuration(self, config: Dict) -> str:
        """Exporte une configuration en JSON."""
        # Convertir les objets date en strings
        export_config = config.copy()
        if 'period' in export_config and export_config['period']:
            export_config['period']['start_date'] = export_config['period']['start_date'].isoformat()
            export_config['period']['end_date'] = export_config['period']['end_date'].isoformat()

        return json.dumps(export_config, indent=2, default=str)

    def import_configuration(self, config_json: str) -> Dict:
        """Importe une configuration depuis JSON."""
        try:
            config = json.loads(config_json)

            # Convertir les strings date en objets date
            if 'period' in config and config['period']:
                config['period']['start_date'] = datetime.fromisoformat(
                    config['period']['start_date']
                ).date()
                config['period']['end_date'] = datetime.fromisoformat(
                    config['period']['end_date']
                ).date()

            return config

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON configuration: {e}")
            return {}
        except Exception as e:
            st.error(f"Error importing configuration: {e}")
            return {}

    def render_configuration_summary(self, config: Dict):
        """Rendu du résumé de configuration."""
        st.subheader("📋 Configuration Summary")

        if not config:
            st.info("Configure your backtest parameters above")
            return

        # Résumé général
        summary_data = {}

        if 'strategy_type' in config:
            summary_data['Strategy'] = config['strategy_type']

        if 'universe' in config and config['universe']:
            summary_data['Assets'] = len(config['universe']['selected_assets'])
            summary_data['Timeframe'] = config['universe']['timeframe']

        if 'period' in config and config['period']:
            summary_data['Period'] = f"{config['period']['period_days']} days"

        if 'trading' in config and config['trading']:
            summary_data['Capital'] = f"${config['trading']['initial_capital']:,}"
            summary_data['Commission'] = f"{config['trading']['commission_rate']}%"

        # Affichage en métriques
        if summary_data:
            cols = st.columns(len(summary_data))
            for i, (key, value) in enumerate(summary_data.items()):
                with cols[i]:
                    st.metric(key, value)

        # Validation
        is_valid, errors = self.validate_configuration(config)

        if is_valid:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration errors:")
            for error in errors:
                st.error(f"• {error}")

        return is_valid