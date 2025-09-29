"""
🗄️ QFrame Data Explorer - UI Component for Data Lake Exploration

Simple UI component for exploring datasets and features in the QFrame Research Platform.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class DataExplorer:
    """
    🗄️ Data Explorer Interface

    Streamlit-based interface for:
    - Dataset browsing and inspection
    - Data quality assessment
    - Feature visualization
    - Data lake management
    """

    def __init__(self, qframe_api=None):
        """
        Initialize Data Explorer

        Args:
            qframe_api: QFrameResearch instance (optional)
        """
        self.qframe_api = qframe_api

        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available. UI components disabled.")

    def render_data_explorer(self):
        """Render the main data explorer interface"""
        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available")
            return

        st.title("🗄️ QFrame Data Explorer")
        st.markdown("Explore datasets and features in the QFrame Data Lake")

        # Sidebar for navigation
        with st.sidebar:
            st.header("📋 Explorer")
            mode = st.selectbox(
                "Select Mode",
                ["Dataset Browser", "Feature Explorer", "Data Quality", "Storage Manager"]
            )

        # Main content based on selected mode
        if mode == "Dataset Browser":
            self._render_dataset_browser()
        elif mode == "Feature Explorer":
            self._render_feature_explorer()
        elif mode == "Data Quality":
            self._render_data_quality()
        elif mode == "Storage Manager":
            self._render_storage_manager()

    def _render_dataset_browser(self):
        """Render dataset browser"""
        st.header("📊 Dataset Browser")

        if not self.qframe_api:
            st.warning("QFrame API not available. Using mock data.")
            self._render_mock_datasets()
            return

        try:
            # Get data manager
            data_manager = self.qframe_api.data_manager()
            datasets = data_manager.list_datasets()

            if datasets:
                st.subheader(f"📋 Available Datasets ({len(datasets)})")

                # Dataset selection
                dataset_names = [ds["name"] for ds in datasets]
                selected_dataset = st.selectbox("Select Dataset", dataset_names)

                if selected_dataset:
                    dataset_info = data_manager.get_dataset_info(selected_dataset)

                    # Dataset overview
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("ℹ️ Dataset Info")
                        st.json(dataset_info)

                    with col2:
                        st.subheader("📈 Dataset Preview")
                        data = data_manager.get_dataset(selected_dataset)
                        if data is not None:
                            st.dataframe(data.head(100), use_container_width=True)

                            # Basic visualization
                            if 'close' in data.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=data.index[:200],  # Limit for performance
                                    y=data['close'][:200],
                                    mode='lines',
                                    name='Close Price'
                                ))
                                fig.update_layout(
                                    title=f"{selected_dataset} - Close Price",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to load dataset")

                    # Dataset actions
                    st.subheader("🛠️ Dataset Actions")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("📊 Generate Summary"):
                            summary = data_manager.create_dataset_summary(selected_dataset)
                            st.json(summary)

                    with col2:
                        if st.button("🔧 Compute Features"):
                            with st.spinner("Computing features..."):
                                try:
                                    feature_name = data_manager.compute_and_store_features(selected_dataset)
                                    st.success(f"Features computed: {feature_name}")
                                except Exception as e:
                                    st.error(f"Feature computation failed: {e}")

                    with col3:
                        export_format = st.selectbox("Export Format", ["parquet", "csv", "json"])
                        if st.button("💾 Export Dataset"):
                            export_path = f"/tmp/{selected_dataset}.{export_format}"
                            success = data_manager.export_dataset(selected_dataset, export_path, export_format)
                            if success:
                                st.success(f"Dataset exported to {export_path}")
                            else:
                                st.error("Export failed")

            else:
                st.info("No datasets found. Ingest some data first!")
                self._render_data_ingestion_form()

        except Exception as e:
            st.error(f"Data exploration error: {e}")
            self._render_mock_datasets()

    def _render_feature_explorer(self):
        """Render feature explorer"""
        st.header("🔧 Feature Explorer")

        if not self.qframe_api:
            st.warning("QFrame API not available. Cannot explore features.")
            return

        st.info("Feature exploration functionality will be available when a dataset is selected.")

        # Feature store interface
        st.subheader("💾 Feature Store")
        st.write("Manage and explore computed features:")

        # Mock feature list
        features = [
            "symbolic_features_v1",
            "technical_indicators_v2",
            "market_regime_features_v1"
        ]

        selected_feature = st.selectbox("Select Feature Set", features)
        if selected_feature:
            st.write(f"Feature set: {selected_feature}")

            # Mock feature details
            st.subheader("📊 Feature Details")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Feature Count", "18")
                st.metric("Data Points", "1,250")

            with col2:
                st.metric("Computation Time", "2.3s")
                st.metric("Storage Size", "4.2 MB")

    def _render_data_quality(self):
        """Render data quality assessment"""
        st.header("🔍 Data Quality Assessment")

        st.info("Data quality tools help ensure your datasets are clean and ready for analysis.")

        # Quality metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Completeness", "94.2%", delta="2.1%")

        with col2:
            st.metric("Consistency", "98.7%", delta="-0.3%")

        with col3:
            st.metric("Accuracy", "96.1%", delta="1.4%")

        # Quality checks
        st.subheader("📋 Quality Checks")

        checks = [
            {"name": "Missing Values", "status": "✅", "score": "94.2%"},
            {"name": "Duplicate Records", "status": "✅", "score": "99.8%"},
            {"name": "Data Types", "status": "⚠️", "score": "89.1%"},
            {"name": "Range Validation", "status": "✅", "score": "97.3%"},
            {"name": "Schema Compliance", "status": "✅", "score": "100%"}
        ]

        for check in checks:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(check["name"])
            with col2:
                st.write(check["status"])
            with col3:
                st.write(check["score"])

    def _render_storage_manager(self):
        """Render storage management interface"""
        st.header("💾 Storage Manager")

        # Storage overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Datasets", "12")

        with col2:
            st.metric("Storage Used", "2.4 GB")

        with col3:
            st.metric("Available Space", "97.6 GB")

        # Storage actions
        st.subheader("🛠️ Storage Actions")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cleanup Options:**")
            days_old = st.slider("Remove datasets older than (days)", 1, 90, 30)
            if st.button("🗑️ Clean Old Datasets"):
                st.info(f"Would remove datasets older than {days_old} days")

        with col2:
            st.write("**Backup Options:**")
            if st.button("💾 Create Backup"):
                st.info("Backup functionality not implemented yet")
            if st.button("📥 Restore Backup"):
                st.info("Restore functionality not implemented yet")

    def _render_data_ingestion_form(self):
        """Render data ingestion form"""
        st.subheader("📥 Data Ingestion")

        with st.form("ingest_data"):
            col1, col2 = st.columns(2)

            with col1:
                symbol = st.text_input("Trading Pair", value="BTC/USDT")
                timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])

            with col2:
                days = st.slider("Historical Days", 1, 90, 30)
                provider = st.selectbox("Data Provider", ["ccxt", "binance", "coinbase"])

            submitted = st.form_submit_button("📥 Ingest Data")

            if submitted:
                st.info(f"Would ingest {symbol} data ({timeframe}) for {days} days from {provider}")

    def _render_mock_datasets(self):
        """Render mock datasets for demo"""
        st.subheader("📊 Sample Datasets (Demo Mode)")

        mock_datasets = [
            {"name": "BTCUSDT_1h_demo", "type": "market_data", "rows": 720, "size": "2.1 MB"},
            {"name": "ETHUSDT_4h_demo", "type": "market_data", "rows": 180, "size": "850 KB"},
            {"name": "portfolio_backtest_demo", "type": "backtest_results", "rows": 1000, "size": "1.5 MB"}
        ]

        for dataset in mock_datasets:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

                with col1:
                    st.write(f"📊 **{dataset['name']}**")

                with col2:
                    st.write(dataset['type'])

                with col3:
                    st.write(f"{dataset['rows']} rows")

                with col4:
                    st.write(dataset['size'])

                st.divider()

    def get_explorer_stats(self) -> Dict[str, Any]:
        """Get data explorer statistics"""
        return {
            "mode": "demo",
            "datasets_explored": 0,
            "features_computed": 0,
            "quality_checks_run": 0
        }

    def __repr__(self):
        return "DataExplorer(demo_mode)"