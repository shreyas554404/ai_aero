from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


FEATURE_COLUMNS = ["vibration", "temperature", "pressure", "cycles"]
DEFAULT_COMPONENTS = ["engine", "landing_gear", "hydraulics", "avionics"]
DEFAULT_MANUFACTURERS = ["Airbus", "Boeing", "Embraer"]


@dataclass
class ComponentDecision:
    component: str
    manufacturer: str
    predicted_wear: float
    wear_limit: float
    within_sensor_limits: bool
    airworthy: bool
    reason: str


class PredictiveMaintenanceModel:
    def __init__(self, limits_config: Optional[Dict] = None) -> None:
        self.limits = limits_config or self.default_limits()
        self.models: Dict[str, Dict[str, object]] = {}
        self.best_model_per_component: Dict[str, str] = {}

    @staticmethod
    def default_limits() -> Dict:
        # Placeholder values must be replaced by approved regulator/manufacturer limits in production.
        return {
            "Airbus": {
                "engine": {"vibration_max": 5.0, "temperature_max": 920.0, "pressure_min": 20.0, "pressure_max": 80.0, "wear_max": 0.60},
                "landing_gear": {"vibration_max": 4.0, "temperature_max": 300.0, "pressure_min": 15.0, "pressure_max": 70.0, "wear_max": 0.55},
                "hydraulics": {"vibration_max": 3.5, "temperature_max": 180.0, "pressure_min": 30.0, "pressure_max": 90.0, "wear_max": 0.50},
                "avionics": {"vibration_max": 2.5, "temperature_max": 120.0, "pressure_min": 10.0, "pressure_max": 60.0, "wear_max": 0.45},
            },
            "Boeing": {
                "engine": {"vibration_max": 5.2, "temperature_max": 940.0, "pressure_min": 22.0, "pressure_max": 82.0, "wear_max": 0.62},
                "landing_gear": {"vibration_max": 4.2, "temperature_max": 310.0, "pressure_min": 16.0, "pressure_max": 72.0, "wear_max": 0.57},
                "hydraulics": {"vibration_max": 3.6, "temperature_max": 185.0, "pressure_min": 31.0, "pressure_max": 92.0, "wear_max": 0.52},
                "avionics": {"vibration_max": 2.7, "temperature_max": 125.0, "pressure_min": 11.0, "pressure_max": 62.0, "wear_max": 0.47},
            },
            "Embraer": {
                "engine": {"vibration_max": 4.9, "temperature_max": 900.0, "pressure_min": 19.0, "pressure_max": 78.0, "wear_max": 0.58},
                "landing_gear": {"vibration_max": 3.9, "temperature_max": 295.0, "pressure_min": 14.0, "pressure_max": 68.0, "wear_max": 0.53},
                "hydraulics": {"vibration_max": 3.4, "temperature_max": 175.0, "pressure_min": 29.0, "pressure_max": 88.0, "wear_max": 0.49},
                "avionics": {"vibration_max": 2.4, "temperature_max": 118.0, "pressure_min": 9.0, "pressure_max": 58.0, "wear_max": 0.44},
            },
        }

    def load_data(
        self,
        source: str = "synthetic",
        csv_path: Optional[str] = None,
        live_data: Optional[Iterable[Dict]] = None,
        n_samples: int = 1200,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        if source == "csv":
            if not csv_path:
                raise ValueError("csv_path is required when source='csv'.")
            data = pd.read_csv(csv_path)
        elif source == "live":
            if live_data is None:
                raise ValueError("live_data iterable is required when source='live'.")
            data = pd.DataFrame(live_data)
        elif source == "synthetic":
            data = self.generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)
        else:
            raise ValueError("source must be one of: synthetic, csv, live")

        required_columns = {"flight_id", "manufacturer", "component", *FEATURE_COLUMNS, "wear_index"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return data

    def generate_synthetic_data(self, n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed)
        rows = []
        for i in range(n_samples):
            manufacturer = rng.choice(DEFAULT_MANUFACTURERS)
            component = rng.choice(DEFAULT_COMPONENTS)
            limits = self.limits[manufacturer][component]
            vibration = max(0.2, rng.normal(limits["vibration_max"] * 0.72, 0.7))
            temperature = max(20.0, rng.normal(limits["temperature_max"] * 0.78, limits["temperature_max"] * 0.07))
            pressure = max(5.0, rng.normal((limits["pressure_min"] + limits["pressure_max"]) / 2, 6.0))
            cycles = rng.integers(100, 8000)

            wear = (
                0.25 * (vibration / limits["vibration_max"])
                + 0.30 * (temperature / limits["temperature_max"])
                + 0.20 * (abs(pressure - ((limits["pressure_min"] + limits["pressure_max"]) / 2)) / max(1.0, limits["pressure_max"]))
                + 0.25 * (cycles / 8000)
                + rng.normal(0, 0.03)
            )
            wear = float(np.clip(wear, 0.0, 1.2))

            rows.append(
                {
                    "flight_id": f"FL-{10000 + (i % 400)}",
                    "manufacturer": manufacturer,
                    "component": component,
                    "vibration": float(vibration),
                    "temperature": float(temperature),
                    "pressure": float(pressure),
                    "cycles": int(cycles),
                    "wear_index": wear,
                }
            )

        return pd.DataFrame(rows)

    def _build_model_dict(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=42, max_depth=8),
            "random_forest": RandomForestRegressor(random_state=42, n_estimators=150),
            "deep_learning": MLPRegressor(random_state=42, hidden_layer_sizes=(64, 32), max_iter=900),
        }
        if XGBRegressor is not None:
            models["xgboost"] = XGBRegressor(
                random_state=42,
                n_estimators=180,
                max_depth=5,
                learning_rate=0.05,
                objective="reg:squarederror",
            )
        else:
            models["xgboost_fallback"] = GradientBoostingRegressor(random_state=42)
        return models

    def train_models(self, data: pd.DataFrame) -> pd.DataFrame:
        metrics: List[Dict] = []
        for component in sorted(data["component"].unique()):
            component_df = data[data["component"] == component]
            x = component_df[FEATURE_COLUMNS]
            y = component_df["wear_index"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            model_dict = self._build_model_dict()
            self.models[component] = {}
            best_name = ""
            best_rmse = float("inf")

            for name, model in model_dict.items():
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                self.models[component][name] = model
                metrics.append({"component": component, "model": name, "rmse": rmse})
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_name = name

            self.best_model_per_component[component] = best_name

        return pd.DataFrame(metrics).sort_values(["component", "rmse"])

    def evaluate_components(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.best_model_per_component:
            raise RuntimeError("Models are not trained. Call train_models first.")

        decisions: List[ComponentDecision] = []
        scored_rows = []

        for _, row in data.iterrows():
            component = row["component"]
            manufacturer = row["manufacturer"]
            limits = self.limits[manufacturer][component]
            model_name = self.best_model_per_component[component]
            model = self.models[component][model_name]
            input_row = row[FEATURE_COLUMNS].to_frame().T
            predicted_wear = float(model.predict(input_row)[0])

            within_sensor_limits = (
                row["vibration"] <= limits["vibration_max"]
                and row["temperature"] <= limits["temperature_max"]
                and limits["pressure_min"] <= row["pressure"] <= limits["pressure_max"]
            )
            airworthy = bool(within_sensor_limits and predicted_wear <= limits["wear_max"])
            reason = "within limits" if airworthy else "maintenance required"

            decisions.append(
                ComponentDecision(
                    component=component,
                    manufacturer=manufacturer,
                    predicted_wear=predicted_wear,
                    wear_limit=limits["wear_max"],
                    within_sensor_limits=within_sensor_limits,
                    airworthy=airworthy,
                    reason=reason,
                )
            )
            scored_rows.append(
                {
                    **row.to_dict(),
                    "model": model_name,
                    "predicted_wear": predicted_wear,
                    "wear_limit": limits["wear_max"],
                    "airworthy": airworthy,
                    "reason": reason,
                }
            )

        result_df = pd.DataFrame(scored_rows)

        flight_summary = (
            result_df.groupby("flight_id")
            .agg(
                worst_predicted_wear=("predicted_wear", "max"),
                max_wear_limit=("wear_limit", "max"),
                airworthy=("airworthy", "all"),
            )
            .reset_index()
        )
        flight_summary["final_decision"] = np.where(
            flight_summary["airworthy"],
            "AIRWORTHY",
            "SEND_TO_MAINTENANCE",
        )

        return result_df, flight_summary

    def generate_3d_graphs(self, result_df: pd.DataFrame, output_dir: str) -> List[str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        for component in sorted(result_df["component"].unique()):
            cdf = result_df[result_df["component"] == component]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                cdf["vibration"],
                cdf["temperature"],
                cdf["predicted_wear"],
                c=cdf["predicted_wear"],
                cmap="viridis",
                alpha=0.8,
            )
            ax.set_title(f"{component} Predictive Maintenance 3D Graph")
            ax.set_xlabel("Vibration")
            ax.set_ylabel("Temperature")
            ax.set_zlabel("Predicted Wear")
            fig.colorbar(scatter, ax=ax, label="Predicted Wear")
            file_path = out_dir / f"{component}_3d.png"
            fig.tight_layout()
            fig.savefig(file_path)
            plt.close(fig)
            generated.append(str(file_path))

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for component in sorted(result_df["component"].unique()):
            cdf = result_df[result_df["component"] == component]
            ax.scatter(cdf["vibration"], cdf["pressure"], cdf["predicted_wear"], alpha=0.55, label=component)
        ax.set_title("Master 3D Graph - Combined Components")
        ax.set_xlabel("Vibration")
        ax.set_ylabel("Pressure")
        ax.set_zlabel("Predicted Wear")
        ax.legend(loc="upper left", fontsize=8)
        master_path = out_dir / "master_combined_3d.png"
        fig.tight_layout()
        fig.savefig(master_path)
        plt.close(fig)
        generated.append(str(master_path))
        return generated


def run_pipeline(
    source: str = "synthetic",
    csv_path: Optional[str] = None,
    limits_path: Optional[str] = None,
    output_dir: str = "outputs",
    n_samples: int = 1200,
) -> Dict:
    limits = None
    if limits_path:
        with open(limits_path, "r", encoding="utf-8") as f:
            limits = json.load(f)

    model = PredictiveMaintenanceModel(limits_config=limits)
    data = model.load_data(source=source, csv_path=csv_path, n_samples=n_samples)
    metrics_df = model.train_models(data)
    scored_df, flight_summary = model.evaluate_components(data)
    graphs = model.generate_3d_graphs(scored_df, output_dir=output_dir)

    metrics_file = Path(output_dir) / "model_metrics.csv"
    scored_file = Path(output_dir) / "component_results.csv"
    summary_file = Path(output_dir) / "flight_airworthiness_summary.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_file, index=False)
    scored_df.to_csv(scored_file, index=False)
    flight_summary.to_csv(summary_file, index=False)

    return {
        "metrics": str(metrics_file),
        "component_results": str(scored_file),
        "flight_summary": str(summary_file),
        "graphs": graphs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-based predictive maintenance model for aircraft components")
    parser.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--limits-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-samples", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_pipeline(
        source=args.source,
        csv_path=args.csv_path,
        limits_path=args.limits_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )
    print("Pipeline completed.")
    for name, value in artifacts.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
