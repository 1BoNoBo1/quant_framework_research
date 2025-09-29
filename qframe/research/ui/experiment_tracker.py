"""
🎯 QFrame Experiment Tracker - Research UI Component

UI component for experiment tracking and visualization within the Research Platform.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class ExperimentTracker:
    """
    🎯 Experiment Tracking Interface

    Streamlit-based interface for:
    - Experiment monitoring and management
    - Result visualization and comparison
    - Model performance tracking
    - Research workflow management
    """

    def __init__(self, qframe_api=None):
        """
        Initialize Experiment Tracker

        Args:
            qframe_api: QFrameResearch instance (optional)
        """
        self.qframe_api = qframe_api
        self.experiments_data = []
        self.current_experiment = None

        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available. UI components disabled.")

    def render_experiment_dashboard(self):
        """Render the main experiment dashboard"""
        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available")
            return

        st.title("🎯 QFrame Experiment Tracker")
        st.markdown("Research experiment monitoring and analysis")

        # Sidebar for navigation
        with st.sidebar:
            st.header("📋 Navigation")
            page = st.selectbox(
                "Select Page",
                ["Overview", "Active Experiments", "Results Analysis", "Model Comparison", "Data Explorer"]
            )

        # Main content based on selected page
        if page == "Overview":
            self._render_overview()
        elif page == "Active Experiments":
            self._render_active_experiments()
        elif page == "Results Analysis":
            self._render_results_analysis()
        elif page == "Model Comparison":
            self._render_model_comparison()
        elif page == "Data Explorer":
            self._render_data_explorer()

    def _render_overview(self):
        """Render experiment overview"""
        st.header("📊 Experiment Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", len(self.experiments_data))

        with col2:
            active_count = sum(1 for exp in self.experiments_data if exp.get("status") == "running")
            st.metric("Active Experiments", active_count)

        with col3:
            completed_count = sum(1 for exp in self.experiments_data if exp.get("status") == "completed")
            st.metric("Completed", completed_count)

        with col4:
            if self.experiments_data:
                avg_duration = np.mean([exp.get("duration", 0) for exp in self.experiments_data])
                st.metric("Avg Duration (min)", f"{avg_duration:.1f}")
            else:
                st.metric("Avg Duration (min)", "0.0")

        # Recent experiments
        st.subheader("🕒 Recent Experiments")
        if self.experiments_data:
            df = pd.DataFrame(self.experiments_data[-10:])  # Last 10 experiments
            st.dataframe(df[["name", "strategy", "status", "sharpe_ratio", "total_return"]], use_container_width=True)
        else:
            st.info("No experiments found. Start your first experiment!")

        # Performance trend
        if len(self.experiments_data) > 1:
            st.subheader("📈 Performance Trend")
            self._plot_performance_trend()

    def _render_active_experiments(self):
        """Render active experiments monitoring"""
        st.header("🏃‍♂️ Active Experiments")

        # Create new experiment section
        with st.expander("➕ Create New Experiment"):
            self._render_experiment_creator()

        # Active experiments list
        active_experiments = [exp for exp in self.experiments_data if exp.get("status") == "running"]

        if active_experiments:
            for exp in active_experiments:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.subheader(f"🔬 {exp['name']}")
                        st.write(f"Strategy: {exp.get('strategy', 'Unknown')}")
                        st.write(f"Started: {exp.get('start_time', 'Unknown')}")

                    with col2:
                        if st.button(f"Stop", key=f"stop_{exp['name']}"):
                            self._stop_experiment(exp['name'])

                    with col3:
                        if st.button(f"View", key=f"view_{exp['name']}"):
                            self._view_experiment_details(exp)

                    # Progress indicator
                    progress = exp.get("progress", 0)
                    st.progress(progress / 100 if progress <= 100 else 1.0)

                    st.divider()
        else:
            st.info("No active experiments. Create a new one above!")

    def _render_results_analysis(self):
        """Render results analysis page"""
        st.header("📊 Results Analysis")

        completed_experiments = [exp for exp in self.experiments_data if exp.get("status") == "completed"]

        if not completed_experiments:
            st.warning("No completed experiments to analyze.")
            return

        # Experiment selector
        exp_names = [exp["name"] for exp in completed_experiments]
        selected_exp = st.selectbox("Select Experiment", exp_names)

        if selected_exp:
            experiment = next(exp for exp in completed_experiments if exp["name"] == selected_exp)

            # Results summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Return", f"{experiment.get('total_return', 0):.2%}")

            with col2:
                st.metric("Sharpe Ratio", f"{experiment.get('sharpe_ratio', 0):.2f}")

            with col3:
                st.metric("Max Drawdown", f"{experiment.get('max_drawdown', 0):.2%}")

            # Detailed results
            st.subheader("📈 Performance Charts")
            self._plot_experiment_results(experiment)

            # Parameters and metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⚙️ Parameters")
                params = experiment.get("parameters", {})
                for key, value in params.items():
                    st.write(f"• {key}: {value}")

            with col2:
                st.subheader("📊 Metrics")
                metrics = experiment.get("metrics", {})
                for key, value in metrics.items():
                    st.write(f"• {key}: {value}")

    def _render_model_comparison(self):
        """Render model comparison page"""
        st.header("🔍 Model Comparison")

        completed_experiments = [exp for exp in self.experiments_data if exp.get("status") == "completed"]

        if len(completed_experiments) < 2:
            st.warning("Need at least 2 completed experiments for comparison.")
            return

        # Multi-select for experiments
        exp_names = [exp["name"] for exp in completed_experiments]
        selected_exps = st.multiselect("Select Experiments to Compare", exp_names, default=exp_names[:3])

        if selected_exps:
            selected_experiments = [exp for exp in completed_experiments if exp["name"] in selected_exps]

            # Comparison table
            st.subheader("📊 Performance Comparison")
            comparison_data = []

            for exp in selected_experiments:
                comparison_data.append({
                    "Experiment": exp["name"],
                    "Strategy": exp.get("strategy", "Unknown"),
                    "Total Return": f"{exp.get('total_return', 0):.2%}",
                    "Sharpe Ratio": f"{exp.get('sharpe_ratio', 0):.2f}",
                    "Max Drawdown": f"{exp.get('max_drawdown', 0):.2%}",
                    "Duration": f"{exp.get('duration', 0):.1f} min"
                })

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

            # Comparison charts
            st.subheader("📈 Visual Comparison")
            self._plot_experiment_comparison(selected_experiments)

    def _render_data_explorer(self):
        """Render data explorer page"""
        st.header("🗄️ Data Explorer")

        if not self.qframe_api:
            st.warning("QFrame API not available for data exploration.")
            return

        # Data manager integration
        try:
            data_manager = self.qframe_api.data_manager()
            datasets = data_manager.list_datasets()

            if datasets:
                st.subheader("📊 Available Datasets")

                dataset_names = [ds["name"] for ds in datasets]
                selected_dataset = st.selectbox("Select Dataset", dataset_names)

                if selected_dataset:
                    # Dataset info
                    dataset_info = data_manager.get_dataset_info(selected_dataset)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("ℹ️ Dataset Info")
                        st.json(dataset_info)

                    with col2:
                        st.subheader("📈 Preview")
                        data = data_manager.get_dataset(selected_dataset)
                        if data is not None:
                            st.dataframe(data.head(100))

                            # Basic visualization
                            if 'close' in data.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['close'],
                                    mode='lines',
                                    name='Close Price'
                                ))
                                fig.update_layout(title=f"{selected_dataset} - Close Price")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to load dataset")
            else:
                st.info("No datasets available. Ingest some data first!")

        except Exception as e:
            st.error(f"Data exploration error: {e}")

    def _render_experiment_creator(self):
        """Render experiment creation form"""
        st.subheader("🧪 New Experiment")

        with st.form("create_experiment"):
            exp_name = st.text_input("Experiment Name", value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            strategy = st.selectbox(
                "Strategy",
                ["adaptive_mean_reversion", "dmn_lstm", "funding_arbitrage", "rl_alpha", "custom"]
            )

            symbol = st.text_input("Trading Pair", value="BTC/USDT")
            days = st.slider("Historical Data (days)", 7, 90, 30)

            # Strategy parameters
            st.subheader("⚙️ Strategy Parameters")
            params = {}

            if strategy == "adaptive_mean_reversion":
                params["lookback_short"] = st.number_input("Lookback Short", value=10)
                params["lookback_long"] = st.number_input("Lookback Long", value=50)
                params["z_entry"] = st.number_input("Z Entry Threshold", value=1.0)

            elif strategy == "dmn_lstm":
                params["window_size"] = st.number_input("Window Size", value=64)
                params["hidden_size"] = st.number_input("Hidden Size", value=64)
                params["learning_rate"] = st.number_input("Learning Rate", value=0.001, format="%.4f")

            submitted = st.form_submit_button("🚀 Start Experiment")

            if submitted and exp_name:
                self._start_experiment(exp_name, strategy, symbol, days, params)

    def _start_experiment(self, name: str, strategy: str, symbol: str, days: int, params: Dict):
        """Start a new experiment"""
        if not self.qframe_api:
            st.error("QFrame API not available")
            return

        try:
            # Create experiment record
            experiment = {
                "name": name,
                "strategy": strategy,
                "symbol": symbol,
                "days": days,
                "parameters": params,
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "progress": 0
            }

            self.experiments_data.append(experiment)
            st.success(f"Started experiment: {name}")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to start experiment: {e}")

    def _stop_experiment(self, name: str):
        """Stop an active experiment"""
        for exp in self.experiments_data:
            if exp["name"] == name and exp["status"] == "running":
                exp["status"] = "stopped"
                exp["end_time"] = datetime.now().isoformat()
                st.success(f"Stopped experiment: {name}")
                st.rerun()
                break

    def _view_experiment_details(self, experiment: Dict):
        """View experiment details in sidebar"""
        with st.sidebar:
            st.header(f"🔬 {experiment['name']}")
            st.write(f"**Status:** {experiment['status']}")
            st.write(f"**Strategy:** {experiment.get('strategy', 'Unknown')}")
            st.write(f"**Symbol:** {experiment.get('symbol', 'Unknown')}")
            st.write(f"**Started:** {experiment.get('start_time', 'Unknown')}")

            if experiment.get("parameters"):
                st.subheader("Parameters")
                for key, value in experiment["parameters"].items():
                    st.write(f"• {key}: {value}")

    def _plot_performance_trend(self):
        """Plot performance trend across experiments"""
        if len(self.experiments_data) < 2:
            return

        # Create performance timeline
        fig = go.Figure()

        experiment_names = [exp["name"] for exp in self.experiments_data[-20:]]
        sharpe_ratios = [exp.get("sharpe_ratio", 0) for exp in self.experiments_data[-20:]]
        total_returns = [exp.get("total_return", 0) for exp in self.experiments_data[-20:]]

        fig.add_trace(go.Scatter(
            x=experiment_names,
            y=sharpe_ratios,
            mode='lines+markers',
            name='Sharpe Ratio',
            yaxis='y'
        ))

        fig.add_trace(go.Scatter(
            x=experiment_names,
            y=total_returns,
            mode='lines+markers',
            name='Total Return',
            yaxis='y2'
        ))

        fig.update_layout(
            title="Performance Trend",
            xaxis_title="Experiments",
            yaxis=dict(title="Sharpe Ratio", side="left"),
            yaxis2=dict(title="Total Return", side="right", overlaying="y"),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_experiment_results(self, experiment: Dict):
        """Plot detailed results for a single experiment"""
        # Mock time series data for visualization
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='H'
        )

        # Generate mock portfolio value
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        portfolio_value = 100000 * (1 + returns).cumprod()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Value", "Daily Returns"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_value, name="Portfolio Value"),
            row=1, col=1
        )

        # Daily returns
        daily_returns = np.diff(portfolio_value) / portfolio_value[:-1] * 100
        fig.add_trace(
            go.Scatter(x=dates[1:], y=daily_returns, name="Daily Returns"),
            row=2, col=1
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_experiment_comparison(self, experiments: List[Dict]):
        """Plot comparison between multiple experiments"""
        # Comparison metrics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        exp_names = [exp["name"] for exp in experiments]

        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * len(metrics)]
        )

        for i, metric in enumerate(metrics):
            values = [exp.get(metric, 0) for exp in experiments]

            fig.add_trace(
                go.Bar(x=exp_names, y=values, name=metric.replace('_', ' ').title()),
                row=1, col=i+1
            )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def add_experiment_result(self, experiment: Dict):
        """Add completed experiment result"""
        self.experiments_data.append(experiment)

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics"""
        if not self.experiments_data:
            return {}

        return {
            "total_experiments": len(self.experiments_data),
            "active_experiments": sum(1 for exp in self.experiments_data if exp.get("status") == "running"),
            "completed_experiments": sum(1 for exp in self.experiments_data if exp.get("status") == "completed"),
            "avg_sharpe_ratio": np.mean([exp.get("sharpe_ratio", 0) for exp in self.experiments_data]),
            "best_experiment": max(self.experiments_data, key=lambda x: x.get("sharpe_ratio", -999), default=None)
        }

    def __repr__(self):
        return f"ExperimentTracker({len(self.experiments_data)} experiments)"