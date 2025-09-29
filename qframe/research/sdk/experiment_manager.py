"""
🧪 QFrame Experiment Manager - MLflow Integration

High-level interface for experiment tracking and ML model management.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import pandas as pd
import numpy as np
import json

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class ExperimentManager:
    """
    🧪 MLflow Experiment Manager

    Provides high-level interface for:
    - Experiment creation and management
    - Model artifact logging
    - Parameter and metric tracking
    - QFrame strategy integration
    """

    def __init__(self, experiment_name: str, qframe_api):
        """
        Initialize Experiment Manager

        Args:
            experiment_name: Name of the MLflow experiment
            qframe_api: QFrameResearch instance for integration
        """
        self.experiment_name = experiment_name
        self.qframe_api = qframe_api
        self.experiment_id = None
        self.current_run_id = None

        # State
        self._mlflow_initialized = False
        self._current_run = None
        self._metrics_buffer = {}
        self._params_buffer = {}

        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        else:
            print("⚠️ MLflow not available. Experiment tracking disabled.")

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI if specified in config
            if hasattr(self.qframe_api, 'config') and self.qframe_api.config:
                mlflow_config = getattr(self.qframe_api.config, 'mlflow', None)
                if mlflow_config and hasattr(mlflow_config, 'tracking_uri'):
                    mlflow.set_tracking_uri(mlflow_config.tracking_uri)

            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✅ Created MLflow experiment: {self.experiment_name}")
            except Exception:
                # Experiment already exists
                self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
                print(f"✅ Using existing MLflow experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)
            self._mlflow_initialized = True

        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")
            self._mlflow_initialized = False

    def start_run(self, run_name: Optional[str] = None, **tags) -> str:
        """
        Start a new MLflow run

        Args:
            run_name: Optional run name
            **tags: Additional tags for the run

        Returns:
            Run ID
        """
        if not self._mlflow_initialized:
            print("⚠️ MLflow not initialized. Using mock run.")
            self.current_run_id = f"mock_run_{uuid.uuid4()}"
            return self.current_run_id

        try:
            # Default tags
            default_tags = {
                "qframe_version": "0.1.0",
                "created_by": "QFrame Research Platform",
                "timestamp": datetime.now().isoformat()
            }
            default_tags.update(tags)

            self._current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=default_tags
            )

            self.current_run_id = self._current_run.info.run_id
            print(f"🚀 Started MLflow run: {self.current_run_id}")

            return self.current_run_id

        except Exception as e:
            print(f"⚠️ Failed to start MLflow run: {e}")
            self.current_run_id = f"mock_run_{uuid.uuid4()}"
            return self.current_run_id

    def end_run(self):
        """End the current MLflow run"""
        if not self._mlflow_initialized or not self._current_run:
            return

        try:
            mlflow.end_run()
            print(f"✅ Ended MLflow run: {self.current_run_id}")
            self._current_run = None
            self.current_run_id = None
        except Exception as e:
            print(f"⚠️ Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow

        Args:
            params: Parameters to log
        """
        if not self._mlflow_initialized:
            self._params_buffer.update(params)
            return

        try:
            mlflow.log_params(params)
            print(f"📝 Logged {len(params)} parameters")
        except Exception as e:
            print(f"⚠️ Failed to log parameters: {e}")
            self._params_buffer.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow

        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        if not self._mlflow_initialized:
            self._metrics_buffer.update(metrics)
            return

        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step)
            else:
                mlflow.log_metrics(metrics)
            print(f"📊 Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
            self._metrics_buffer.update(metrics)

    def log_model(self, model: Any, model_name: str, **kwargs):
        """
        Log a model to MLflow

        Args:
            model: Model to log
            model_name: Model name
            **kwargs: Additional arguments for model logging
        """
        if not self._mlflow_initialized:
            print(f"⚠️ MLflow not available. Model '{model_name}' not logged.")
            return

        try:
            # Determine model type and log appropriately
            if hasattr(model, '__module__') and 'torch' in str(model.__module__):
                mlflow.pytorch.log_model(model, model_name, **kwargs)
            elif hasattr(model, '__module__') and 'sklearn' in str(model.__module__):
                mlflow.sklearn.log_model(model, model_name, **kwargs)
            else:
                # Generic pickle
                mlflow.log_artifact(model_name)

            print(f"🤖 Logged model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to log model: {e}")

    def log_dataframe(self, df: pd.DataFrame, name: str):
        """
        Log a DataFrame as an artifact

        Args:
            df: DataFrame to log
            name: Artifact name
        """
        if not self._mlflow_initialized:
            print(f"⚠️ MLflow not available. DataFrame '{name}' not logged.")
            return

        try:
            # Save DataFrame to temporary file
            temp_path = f"/tmp/{name}_{uuid.uuid4()}.parquet"
            df.to_parquet(temp_path)

            mlflow.log_artifact(temp_path, f"data/{name}.parquet")
            print(f"📊 Logged DataFrame: {name}")

            # Cleanup
            os.remove(temp_path)
        except Exception as e:
            print(f"⚠️ Failed to log DataFrame: {e}")

    def log_strategy_results(self, strategy_name: str, results: Dict[str, Any]):
        """
        Log QFrame strategy backtest results

        Args:
            strategy_name: Name of the strategy
            results: Backtest results dictionary
        """
        # Extract key metrics from results
        metrics = {}
        params = {"strategy_name": strategy_name}

        # Common performance metrics
        if "total_return" in results:
            metrics["total_return"] = float(results["total_return"])
        if "sharpe_ratio" in results:
            metrics["sharpe_ratio"] = float(results["sharpe_ratio"])
        if "max_drawdown" in results:
            metrics["max_drawdown"] = float(results["max_drawdown"])
        if "win_rate" in results:
            metrics["win_rate"] = float(results["win_rate"])

        # Strategy parameters
        if "parameters" in results:
            for key, value in results["parameters"].items():
                if isinstance(value, (int, float, str, bool)):
                    params[f"param_{key}"] = value

        # Log to MLflow
        self.log_params(params)
        self.log_metrics(metrics)

        # Log full results as artifact
        if self._mlflow_initialized:
            try:
                temp_path = f"/tmp/results_{uuid.uuid4()}.json"
                with open(temp_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                mlflow.log_artifact(temp_path, f"results/{strategy_name}_results.json")
                os.remove(temp_path)
                print(f"📈 Logged strategy results for {strategy_name}")
            except Exception as e:
                print(f"⚠️ Failed to log strategy results: {e}")

    @contextmanager
    def run(self, run_name: Optional[str] = None, **tags):
        """
        Context manager for MLflow runs

        Args:
            run_name: Optional run name
            **tags: Additional tags

        Usage:
            with experiment.run("my_test"):
                # Your experiment code here
                experiment.log_metrics({"accuracy": 0.95})
        """
        run_id = self.start_run(run_name, **tags)
        try:
            yield run_id
        finally:
            self.end_run()

    async def strategy_experiment(
        self,
        strategy_name: str,
        data: Optional[pd.DataFrame] = None,
        parameter_grid: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete strategy experiment

        Args:
            strategy_name: QFrame strategy to test
            data: Market data (optional)
            parameter_grid: Parameters to test
            run_name: MLflow run name

        Returns:
            Experiment results
        """
        run_name = run_name or f"{strategy_name}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.run(run_name, strategy=strategy_name):
            print(f"🧪 Starting experiment: {run_name}")

            if parameter_grid:
                # Parameter sweep experiment
                results = await self._parameter_sweep_experiment(
                    strategy_name, data, parameter_grid
                )
            else:
                # Single run experiment
                results = await self._single_run_experiment(strategy_name, data)

            # Log summary metrics
            self.log_strategy_results(strategy_name, results)

            print(f"✅ Experiment completed: {run_name}")
            return results

    async def _single_run_experiment(
        self,
        strategy_name: str,
        data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Run single strategy experiment"""
        # Use QFrame API for backtesting
        if data is None:
            data = await self.qframe_api.get_data("BTC/USDT", days=30)

        # Add features
        data_with_features = await self.qframe_api.compute_features(data)

        # Run backtest
        results = await self.qframe_api.backtest(strategy_name, data_with_features)

        return results

    async def _parameter_sweep_experiment(
        self,
        strategy_name: str,
        data: Optional[pd.DataFrame],
        parameter_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Run parameter sweep experiment"""
        import itertools

        # Generate parameter combinations
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        combinations = list(itertools.product(*values))

        print(f"🔄 Testing {len(combinations)} parameter combinations")

        best_result = None
        best_params = None
        best_metric = -np.inf
        all_results = []

        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            print(f"📊 Testing combination {i+1}/{len(combinations)}: {params}")

            try:
                # Create child run for this parameter combination
                child_run_name = f"params_{i+1}"
                with self.run(child_run_name, **{f"param_{k}": v for k, v in params.items()}):
                    # Run backtest with these parameters
                    if data is None:
                        test_data = await self.qframe_api.get_data("BTC/USDT", days=30)
                    else:
                        test_data = data

                    test_data_with_features = await self.qframe_api.compute_features(test_data)
                    result = await self.qframe_api.backtest(strategy_name, test_data_with_features, **params)

                    # Log this combination's results
                    self.log_strategy_results(f"{strategy_name}_sweep", result)

                    # Track best result
                    metric_value = result.get("sharpe_ratio", result.get("total_return", 0))
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_result = result
                        best_params = params

                    all_results.append({
                        "parameters": params,
                        "results": result,
                        "metric": metric_value
                    })

            except Exception as e:
                print(f"⚠️ Error in parameter combination {params}: {e}")

        # Create summary
        summary = {
            "strategy": strategy_name,
            "total_combinations": len(combinations),
            "successful_runs": len(all_results),
            "best_parameters": best_params,
            "best_results": best_result,
            "best_metric": best_metric,
            "all_results": all_results
        }

        return summary

    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """
        Compare results across multiple experiments

        Args:
            experiment_names: List of experiment names to compare

        Returns:
            Comparison DataFrame
        """
        if not self._mlflow_initialized:
            print("⚠️ MLflow not available. Cannot compare experiments.")
            return pd.DataFrame()

        try:
            all_runs = []

            for exp_name in experiment_names:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    runs = mlflow.search_runs(experiment.experiment_id)
                    runs["experiment"] = exp_name
                    all_runs.append(runs)

            if all_runs:
                comparison_df = pd.concat(all_runs, ignore_index=True)
                return comparison_df[["experiment", "start_time", "status"] +
                                  [col for col in comparison_df.columns if col.startswith("metrics.")]]
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"⚠️ Failed to compare experiments: {e}")
            return pd.DataFrame()

    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific run

        Args:
            run_id: Run ID (defaults to current run)

        Returns:
            Run information
        """
        if not self._mlflow_initialized:
            return {"status": "MLflow not available", "params": self._params_buffer, "metrics": self._metrics_buffer}

        try:
            run_id = run_id or self.current_run_id
            if not run_id:
                return {"error": "No run ID specified"}

            run = mlflow.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
        except Exception as e:
            return {"error": f"Failed to get run info: {e}"}

    def cleanup(self):
        """Cleanup resources"""
        if self._current_run:
            self.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __repr__(self):
        status = "initialized" if self._mlflow_initialized else "mock mode"
        return f"ExperimentManager('{self.experiment_name}', {status})"