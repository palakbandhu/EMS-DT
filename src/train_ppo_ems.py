"""Production-style PPO training and evaluation script for EMS dispatching.

Features
- Uses the existing EMSEnv and Poisson incident generator.
- Supports GPU training with Stable-Baselines3 / SB3-Contrib.
- Saves checkpoints, final model, VecNormalize stats, config, metrics, and plots.
- Runs both deterministic and stochastic evaluation after training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from tqdm.auto import tqdm

import rl_train


LOGGER = logging.getLogger("ems_ppo")


class StreamToLogger:
    """Redirect writes from stdout/stderr into the logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> None:
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            self.logger.log(self.level, line)
        self._buffer = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PPO for EMS dispatching")
    parser.add_argument("--hospitals", default="HOSPITALS_DATA_FINAL.xlsx")
    parser.add_argument("--ambulances", default="final_ambulance_deployment.csv")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--episode-days", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--prefer-external-generator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--avg-speed-kmh", type=float, default=40.0)
    parser.add_argument("--travel-noise-std", type=float, default=0.1)
    parser.add_argument("--enforce-als-for-high", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k-ambulances-low", type=int, default=6)
    parser.add_argument("--top-k-ambulances-medium", type=int, default=4)
    parser.add_argument("--top-k-ambulances-high", type=int, default=3)
    parser.add_argument("--top-k-hospitals-low", type=int, default=8)
    parser.add_argument("--top-k-hospitals-medium", type=int, default=6)
    parser.add_argument("--top-k-hospitals-high", type=int, default=4)
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--plot-rolling-window", type=int, default=10)
    return parser.parse_args()


@dataclass
class RuntimeConfig:
    hospitals: str
    ambulances: str
    output_root: str
    run_name: Optional[str]
    seed: int
    timesteps: int
    episode_days: int
    eval_episodes: int
    checkpoint_freq: int
    log_every_steps: int
    device: str
    prefer_external_generator: bool
    avg_speed_kmh: float
    travel_noise_std: float
    enforce_als_for_high: bool
    top_k_ambulances_low: int
    top_k_ambulances_medium: int
    top_k_ambulances_high: int
    top_k_hospitals_low: int
    top_k_hospitals_medium: int
    top_k_hospitals_high: int
    normalize_observations: bool
    normalize_rewards: bool
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float
    eval_freq: int
    plot_rolling_window: int


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.astype(str).tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_scalar(value: Any) -> Any:
    if isinstance(value, pd.Series):
        non_null = value.dropna().tolist()
        if not non_null:
            return None
        if len(non_null) == 1:
            return sanitize_scalar(non_null[0])
        return " | ".join(str(item) for item in non_null)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_file: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False
    sys.stdout = StreamToLogger(LOGGER, logging.INFO)
    sys.stderr = StreamToLogger(LOGGER, logging.ERROR)


def resolve_device(device_arg: str) -> Dict[str, Any]:
    requested = device_arg
    resolved = rl_train.get_training_device(device_arg)
    cuda_available = bool(torch.cuda.is_available())
    device_info: Dict[str, Any] = {
        "requested_device": requested,
        "resolved_device": resolved,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
    }
    if cuda_available:
        active_index = torch.cuda.current_device()
        device_info["cuda_device_count"] = torch.cuda.device_count()
        device_info["cuda_active_index"] = active_index
        device_info["cuda_active_name"] = torch.cuda.get_device_name(active_index)
    return device_info


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def build_base_env(
    hospital_df: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    seed: int,
    device: str,
    prefer_external_generator: bool,
    episode_days: int,
    avg_speed_kmh: float,
    travel_noise_std: float,
    enforce_als_for_high: bool,
    top_k_ambulances_low: int,
    top_k_ambulances_medium: int,
    top_k_ambulances_high: int,
    top_k_hospitals_low: int,
    top_k_hospitals_medium: int,
    top_k_hospitals_high: int,
) -> rl_train.EMSEnv:
    incident_generator = rl_train.make_incident_generator(
        seed=seed,
        prefer_external=prefer_external_generator,
    )
    return rl_train.EMSEnv(
        hospital_df=hospital_df,
        ambulance_df=ambulance_df,
        incident_generator_fn=incident_generator,
        episode_days=episode_days,
        avg_speed_kmh=avg_speed_kmh,
        travel_noise_std=travel_noise_std,
        enforce_als_for_high_if_available=enforce_als_for_high,
        top_k_ambulances_low=top_k_ambulances_low,
        top_k_ambulances_medium=top_k_ambulances_medium,
        top_k_ambulances_high=top_k_ambulances_high,
        top_k_hospitals_low=top_k_hospitals_low,
        top_k_hospitals_medium=top_k_hospitals_medium,
        top_k_hospitals_high=top_k_hospitals_high,
        seed=seed,
        device=device,
    )


def make_env_factory(
    hospital_df: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    seed: int,
    device: str,
    prefer_external_generator: bool,
    episode_days: int,
    avg_speed_kmh: float,
    travel_noise_std: float,
    enforce_als_for_high: bool,
    top_k_ambulances_low: int,
    top_k_ambulances_medium: int,
    top_k_ambulances_high: int,
    top_k_hospitals_low: int,
    top_k_hospitals_medium: int,
    top_k_hospitals_high: int,
    monitor_path: Optional[Path] = None,
):
    def _factory():
        env = build_base_env(
            hospital_df=hospital_df,
            ambulance_df=ambulance_df,
            seed=seed,
            device=device,
            prefer_external_generator=prefer_external_generator,
            episode_days=episode_days,
            avg_speed_kmh=avg_speed_kmh,
            travel_noise_std=travel_noise_std,
            enforce_als_for_high=enforce_als_for_high,
            top_k_ambulances_low=top_k_ambulances_low,
            top_k_ambulances_medium=top_k_ambulances_medium,
            top_k_ambulances_high=top_k_ambulances_high,
            top_k_hospitals_low=top_k_hospitals_low,
            top_k_hospitals_medium=top_k_hospitals_medium,
            top_k_hospitals_high=top_k_hospitals_high,
        )
        if monitor_path is not None:
            env = Monitor(env, filename=str(monitor_path))
        else:
            env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.action_mask())
        return env

    return _factory


def build_vec_env(
    hospital_df: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    seed: int,
    device: str,
    prefer_external_generator: bool,
    episode_days: int,
    avg_speed_kmh: float,
    travel_noise_std: float,
    enforce_als_for_high: bool,
    top_k_ambulances_low: int,
    top_k_ambulances_medium: int,
    top_k_ambulances_high: int,
    top_k_hospitals_low: int,
    top_k_hospitals_medium: int,
    top_k_hospitals_high: int,
    monitor_path: Optional[Path],
    normalize_observations: bool,
    normalize_rewards: bool,
    training: bool,
    gamma: float,
) -> VecNormalize:
    vec_env = DummyVecEnv(
        [
            make_env_factory(
                hospital_df=hospital_df,
                ambulance_df=ambulance_df,
                seed=seed,
                device=device,
                prefer_external_generator=prefer_external_generator,
                episode_days=episode_days,
                avg_speed_kmh=avg_speed_kmh,
                travel_noise_std=travel_noise_std,
                enforce_als_for_high=enforce_als_for_high,
                top_k_ambulances_low=top_k_ambulances_low,
                top_k_ambulances_medium=top_k_ambulances_medium,
                top_k_ambulances_high=top_k_ambulances_high,
                top_k_hospitals_low=top_k_hospitals_low,
                top_k_hospitals_medium=top_k_hospitals_medium,
                top_k_hospitals_high=top_k_hospitals_high,
                monitor_path=monitor_path,
            )
        ]
    )
    vec_env = VecNormalize(
        vec_env,
        training=training,
        norm_obs=normalize_observations,
        norm_reward=normalize_rewards,
        clip_obs=10.0,
        gamma=gamma,
    )
    return vec_env


class DispatchMetricsCallback(BaseCallback):
    """Persist per-dispatch training metrics without keeping everything in memory."""

    def __init__(
        self,
        output_csv: Path,
        dispatch_log_path: Path,
        log_every_steps: int,
        flush_every: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_csv = output_csv
        self.dispatch_log_path = dispatch_log_path
        self.log_every_steps = max(int(log_every_steps), 1)
        self.flush_every = flush_every
        self.buffer: List[Dict[str, Any]] = []
        self.header_written = output_csv.exists() and output_csv.stat().st_size > 0
        self.last_logged_bucket = -1
        self.latest_dispatch_record: Optional[Dict[str, Any]] = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info or info.get("invalid_action") or "event_id" not in info:
                continue
            record = {
                "training_timestep": int(self.num_timesteps),
                "event_id": sanitize_scalar(info["event_id"]),
                "severity": sanitize_scalar(info["severity"]),
                "specialty": sanitize_scalar(info["specialty"]),
                "ambulance_id": sanitize_scalar(info["ambulance_id"]),
                "ambulance_type": sanitize_scalar(info["ambulance_type"]),
                "hospital_name": sanitize_scalar(info["hospital_name"]),
                "tau_scene_sec": float(sanitize_scalar(info["tau_scene_sec"])),
                "tau_hospital_sec": float(sanitize_scalar(info["tau_hospital_sec"])),
                "tau_total_sec": float(sanitize_scalar(info["tau_total_sec"])),
                "chosen_response_eta_min": float(sanitize_scalar(info["chosen_response_eta_min"])),
                "chosen_hospital_eta_min": float(sanitize_scalar(info["chosen_hospital_eta_min"])),
                "best_feasible_response_eta_min": sanitize_scalar(info.get("best_feasible_response_eta_min")),
                "best_feasible_hospital_eta_min": sanitize_scalar(info.get("best_feasible_hospital_eta_min")),
                "chosen_minus_best_response_eta_min": sanitize_scalar(info.get("chosen_minus_best_response_eta_min")),
                "chosen_minus_best_hospital_eta_min": sanitize_scalar(info.get("chosen_minus_best_hospital_eta_min")),
                "best_feasible_ambulance_id": sanitize_scalar(info.get("best_feasible_ambulance_id")),
                "best_feasible_hospital_name": sanitize_scalar(info.get("best_feasible_hospital_name")),
                "eta_minutes": float(sanitize_scalar(info["tau_total_sec"])) / 60.0,
                "reward": float(sanitize_scalar(info["reward"])),
                "traffic_multiplier": float(sanitize_scalar(info["traffic_multiplier"])),
            }
            for key, value in info.get("reward_components", {}).items():
                record[f"reward_{key}"] = float(sanitize_scalar(value))
            self.buffer.append(record)
            self.latest_dispatch_record = record
            self._append_dispatch_log(record, phase="train")

            episode_summary = info.get("episode_summary")
            if episode_summary:
                LOGGER.info(
                    "TRAIN EPISODE END | dispatches=%s total_reward=%.3f mean_reward_per_dispatch=%.3f "
                    "completed_patients=%s remaining_queue=%s traffic_multiplier=%.3f",
                    episode_summary["dispatches"],
                    episode_summary["total_reward"],
                    episode_summary["mean_reward_per_dispatch"],
                    episode_summary["completed_patients"],
                    episode_summary["remaining_queue"],
                    episode_summary["traffic_multiplier"],
                )

        current_bucket = self.num_timesteps // self.log_every_steps
        if current_bucket > self.last_logged_bucket and self.latest_dispatch_record is not None:
            record = self.latest_dispatch_record
            LOGGER.info(
                "TRAIN PROGRESS | step=%s incident=%s severity=%s specialty=%s ambulance=%s (%s) hospital=%s "
                "chosen_resp_min=%.2f best_resp_min=%s chosen_hosp_min=%.2f best_hosp_min=%s reward=%.3f",
                self.num_timesteps,
                record["event_id"],
                record["severity"],
                record["specialty"],
                record["ambulance_id"],
                record["ambulance_type"],
                record["hospital_name"],
                record["chosen_response_eta_min"],
                record["best_feasible_response_eta_min"],
                record["chosen_hospital_eta_min"],
                record["best_feasible_hospital_eta_min"],
                record["reward"],
            )
            self.last_logged_bucket = current_bucket

        if len(self.buffer) >= self.flush_every:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.output_csv, mode="a", index=False, header=not self.header_written)
        self.header_written = True
        self.buffer.clear()

    def _append_dispatch_log(self, record: Dict[str, Any], phase: str) -> None:
        payload = {"phase": phase, **record}
        with self.dispatch_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=json_default) + "\n")


class TqdmProgressCallback(BaseCallback):
    """Progress bar that stays disabled when using file-only logging."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar: Optional[tqdm] = None
        self.last_n = 0

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training PPO", unit="ts", disable=True)

    def _on_step(self) -> bool:
        if self.progress_bar is not None:
            delta = self.num_timesteps - self.last_n
            if delta > 0:
                self.progress_bar.update(delta)
                self.last_n = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            remaining = self.total_timesteps - self.last_n
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()


def make_model(env: VecNormalize, config: RuntimeConfig, device: str) -> MaskablePPO:
    policy_kwargs = {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [512, 256, 128], "vf": [512, 256, 128]},
        "ortho_init": False,
    }
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.seed,
        device=device,
    )


def aggregate_eval_summary(dispatch_df: pd.DataFrame, episode_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "episodes": int(len(episode_df)),
        "dispatch_rows": int(len(dispatch_df)),
    }
    if not episode_df.empty:
        summary.update({
            "mean_episode_reward": float(episode_df["total_reward"].mean()),
            "std_episode_reward": float(episode_df["total_reward"].std(ddof=0)),
            "mean_dispatches_per_episode": float(episode_df["dispatches"].mean()),
            "mean_reward_per_dispatch": float(episode_df["mean_reward_per_dispatch"].mean()),
            "mean_completed_patients": float(episode_df["completed_patients"].mean()),
        })
    if not dispatch_df.empty:
        summary.update({
            "mean_tau_scene_min": float(dispatch_df["tau_scene_sec"].mean() / 60.0),
            "mean_tau_total_min": float(dispatch_df["tau_total_sec"].mean() / 60.0),
            "median_tau_total_min": float(dispatch_df["tau_total_sec"].median() / 60.0),
            "high_severity_share": float((dispatch_df["severity"] == "HIGH").mean()),
            "golden_hour_rate_high": float(
                ((dispatch_df["severity"] == "HIGH") & (dispatch_df["tau_total_sec"] <= 3600)).sum()
                / max((dispatch_df["severity"] == "HIGH").sum(), 1)
            ),
        })
    return summary


def run_policy_evaluation(
    model: MaskablePPO,
    vec_env: VecNormalize,
    n_episodes: int,
    deterministic: bool,
    dispatch_log_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    dispatch_records: List[Dict[str, Any]] = []
    episode_records: List[Dict[str, Any]] = []
    obs = vec_env.reset()
    completed_episodes = 0

    progress = tqdm(
        total=n_episodes,
        desc=f"Evaluating ({'det' if deterministic else 'stoch'})",
        unit="ep",
        disable=True,
    )

    while completed_episodes < n_episodes:
        masks = get_action_masks(vec_env)
        actions, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        obs, rewards, dones, infos = vec_env.step(actions)

        for done, info in zip(dones, infos):
            if info and not info.get("invalid_action") and "event_id" in info:
                record = {
                    "policy_mode": "deterministic" if deterministic else "stochastic",
                    "reward": float(sanitize_scalar(info["reward"])),
                    "event_id": sanitize_scalar(info["event_id"]),
                    "severity": sanitize_scalar(info["severity"]),
                    "specialty": sanitize_scalar(info["specialty"]),
                    "ambulance_id": sanitize_scalar(info["ambulance_id"]),
                    "ambulance_type": sanitize_scalar(info["ambulance_type"]),
                    "hospital_name": sanitize_scalar(info["hospital_name"]),
                    "tau_scene_sec": float(sanitize_scalar(info["tau_scene_sec"])),
                    "tau_hospital_sec": float(sanitize_scalar(info["tau_hospital_sec"])),
                    "tau_total_sec": float(sanitize_scalar(info["tau_total_sec"])),
                    "chosen_response_eta_min": float(sanitize_scalar(info["chosen_response_eta_min"])),
                    "chosen_hospital_eta_min": float(sanitize_scalar(info["chosen_hospital_eta_min"])),
                    "best_feasible_response_eta_min": sanitize_scalar(info.get("best_feasible_response_eta_min")),
                    "best_feasible_hospital_eta_min": sanitize_scalar(info.get("best_feasible_hospital_eta_min")),
                    "chosen_minus_best_response_eta_min": sanitize_scalar(info.get("chosen_minus_best_response_eta_min")),
                    "chosen_minus_best_hospital_eta_min": sanitize_scalar(info.get("chosen_minus_best_hospital_eta_min")),
                    "best_feasible_ambulance_id": sanitize_scalar(info.get("best_feasible_ambulance_id")),
                    "best_feasible_hospital_name": sanitize_scalar(info.get("best_feasible_hospital_name")),
                    "eta_minutes": float(sanitize_scalar(info["tau_total_sec"])) / 60.0,
                    "traffic_multiplier": float(sanitize_scalar(info["traffic_multiplier"])),
                }
                for key, value in info.get("reward_components", {}).items():
                    record[f"reward_{key}"] = float(sanitize_scalar(value))
                dispatch_records.append(record)
                with dispatch_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=json_default) + "\n")
            if done:
                summary = info.get("episode_summary")
                if summary:
                    episode_records.append(flatten_dict(summary))
                    LOGGER.info(
                        "EVAL EPISODE END | mode=%s episode=%s dispatches=%s total_reward=%.3f "
                        "mean_reward_per_dispatch=%.3f completed_patients=%s remaining_queue=%s",
                        "deterministic" if deterministic else "stochastic",
                        completed_episodes + 1,
                        summary["dispatches"],
                        summary["total_reward"],
                        summary["mean_reward_per_dispatch"],
                        summary["completed_patients"],
                        summary["remaining_queue"],
                    )
                completed_episodes += 1
                progress.update(1)
                if completed_episodes >= n_episodes:
                    break

    progress.close()
    dispatch_df = pd.DataFrame(dispatch_records)
    episode_df = pd.DataFrame(episode_records)
    summary = aggregate_eval_summary(dispatch_df, episode_df)
    return dispatch_df, episode_df, summary


def rolling_mean(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def plot_training_monitor(monitor_csv: Path, output_png: Path, rolling_window: int) -> None:
    if not monitor_csv.exists():
        return
    monitor_df = pd.read_csv(monitor_csv, comment="#")
    if monitor_df.empty or "r" not in monitor_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(monitor_df.index + 1, monitor_df["r"], alpha=0.4, label="Episode reward")
    axes[0].plot(
        monitor_df.index + 1,
        rolling_mean(monitor_df["r"], rolling_window),
        linewidth=2,
        label=f"Rolling mean ({rolling_window})",
    )
    axes[0].set_title("Training Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(monitor_df.index + 1, monitor_df["l"], alpha=0.7, color="tab:orange")
    axes[1].set_title("Training Episode Length")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Dispatch decisions")

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_dispatch_analysis(dispatch_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    if dispatch_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(dispatch_df["reward"], bins=30, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title("Reward Distribution")
    axes[0, 0].set_xlabel("Reward")

    severity_response = (
        dispatch_df.groupby("severity")["tau_total_sec"].mean().reindex(["LOW", "MEDIUM", "HIGH"])
    ) / 60.0
    axes[0, 1].bar(severity_response.index, severity_response.values, color=["#7fc97f", "#fdc086", "#ef3b2c"])
    axes[0, 1].set_title("Mean Total Travel Time by Severity")
    axes[0, 1].set_ylabel("Minutes")

    reward_cols = [col for col in dispatch_df.columns if col.startswith("reward_") and col != "reward"]
    if reward_cols:
        comp_means = dispatch_df[reward_cols].mean().sort_values()
        axes[1, 0].barh(comp_means.index.str.replace("reward_", "", regex=False), comp_means.values, color="tab:purple")
        axes[1, 0].set_title("Mean Reward Components")

    top_hospitals = dispatch_df["hospital_name"].value_counts().head(10).sort_values()
    axes[1, 1].barh(top_hospitals.index, top_hospitals.values, color="tab:green")
    axes[1, 1].set_title("Top Assigned Hospitals")
    axes[1, 1].set_xlabel("Dispatch count")

    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_dispatch_analysis.png", dpi=160)
    plt.close(fig)


def plot_episode_analysis(episode_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    if episode_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(episode_df.index + 1, episode_df["total_reward"], marker="o")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")

    axes[1].plot(episode_df.index + 1, episode_df["dispatches"], marker="o", color="tab:orange")
    axes[1].set_title("Dispatches per Episode")
    axes[1].set_xlabel("Episode")

    axes[2].plot(episode_df.index + 1, episode_df["mean_reward_per_dispatch"], marker="o", color="tab:green")
    axes[2].set_title("Mean Reward per Dispatch")
    axes[2].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_episode_analysis.png", dpi=160)
    plt.close(fig)


def save_dataset_summaries(
    hospital_df: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    hospital_summary = {
        "num_hospitals": int(len(hospital_df)),
        "total_beds": int(hospital_df["total_beds"].fillna(0).sum()),
        "icu_beds": int(hospital_df["icu_beds"].fillna(0).sum()),
        "ventilator_beds": int(hospital_df["ventilator_beds"].fillna(0).sum()),
        "oxygen_beds": int(hospital_df["oxygen_beds"].fillna(0).sum()),
    }
    ambulance_summary = {
        "num_ambulances": int(len(ambulance_df)),
        "ambulance_types": ambulance_df["type"].value_counts().to_dict(),
    }
    payload = {
        "hospital_summary": hospital_summary,
        "ambulance_summary": ambulance_summary,
    }
    save_json(output_dir / "dataset_summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    config = RuntimeConfig(**vars(args))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"ppo_ems_{timestamp}"
    run_dir = ensure_dir(Path(config.output_root) / run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    plots_dir = ensure_dir(run_dir / "plots")
    logs_dir = ensure_dir(run_dir / "logs")
    setup_logging(logs_dir / "run.log")

    save_json(run_dir / "config.json", asdict(config))

    device_info = resolve_device(config.device)
    save_json(run_dir / "device_info.json", device_info)
    resolved_device = device_info["resolved_device"]

    hospital_df = rl_train.load_hospitals(config.hospitals)
    ambulance_df = rl_train.load_ambulances(config.ambulances)
    dataset_summary = save_dataset_summaries(hospital_df, ambulance_df, run_dir)

    train_monitor_csv = logs_dir / "train_monitor.csv"
    eval_monitor_csv = logs_dir / "eval_monitor.csv"
    training_dispatch_csv = logs_dir / "training_dispatch_metrics.csv"
    training_dispatch_log = logs_dir / "training_dispatch_events.jsonl"
    eval_dispatch_log_det = logs_dir / "evaluation_dispatch_deterministic.jsonl"
    eval_dispatch_log_stoch = logs_dir / "evaluation_dispatch_stochastic.jsonl"

    LOGGER.info("Starting PPO EMS run: %s", run_name)
    LOGGER.info("Artifacts directory: %s", run_dir)
    LOGGER.info("Device info: %s", json.dumps(device_info, default=json_default))
    LOGGER.info("Dataset summary: %s", json.dumps(dataset_summary, default=json_default))
    LOGGER.info(
        "Training config | timesteps=%s episode_days=%s log_every_steps=%s n_steps=%s batch_size=%s gamma=%.3f lr=%s eval_episodes=%s",
        config.timesteps,
        config.episode_days,
        config.log_every_steps,
        config.n_steps,
        config.batch_size,
        config.gamma,
        config.learning_rate,
        config.eval_episodes,
    )
    LOGGER.info(
        "Dispatch pruning | enforce_als_for_high=%s top_k_amb(low/med/high)=%s/%s/%s top_k_hosp(low/med/high)=%s/%s/%s",
        config.enforce_als_for_high,
        config.top_k_ambulances_low,
        config.top_k_ambulances_medium,
        config.top_k_ambulances_high,
        config.top_k_hospitals_low,
        config.top_k_hospitals_medium,
        config.top_k_hospitals_high,
    )

    train_env = build_vec_env(
        hospital_df=hospital_df,
        ambulance_df=ambulance_df,
        seed=config.seed,
        device=resolved_device,
        prefer_external_generator=config.prefer_external_generator,
        episode_days=config.episode_days,
        avg_speed_kmh=config.avg_speed_kmh,
        travel_noise_std=config.travel_noise_std,
        enforce_als_for_high=config.enforce_als_for_high,
        top_k_ambulances_low=config.top_k_ambulances_low,
        top_k_ambulances_medium=config.top_k_ambulances_medium,
        top_k_ambulances_high=config.top_k_ambulances_high,
        top_k_hospitals_low=config.top_k_hospitals_low,
        top_k_hospitals_medium=config.top_k_hospitals_medium,
        top_k_hospitals_high=config.top_k_hospitals_high,
        monitor_path=train_monitor_csv,
        normalize_observations=config.normalize_observations,
        normalize_rewards=config.normalize_rewards,
        training=True,
        gamma=config.gamma,
    )

    eval_env = build_vec_env(
        hospital_df=hospital_df,
        ambulance_df=ambulance_df,
        seed=config.seed + 10_000,
        device=resolved_device,
        prefer_external_generator=config.prefer_external_generator,
        episode_days=config.episode_days,
        avg_speed_kmh=config.avg_speed_kmh,
        travel_noise_std=config.travel_noise_std,
        enforce_als_for_high=config.enforce_als_for_high,
        top_k_ambulances_low=config.top_k_ambulances_low,
        top_k_ambulances_medium=config.top_k_ambulances_medium,
        top_k_ambulances_high=config.top_k_ambulances_high,
        top_k_hospitals_low=config.top_k_hospitals_low,
        top_k_hospitals_medium=config.top_k_hospitals_medium,
        top_k_hospitals_high=config.top_k_hospitals_high,
        monitor_path=eval_monitor_csv,
        normalize_observations=config.normalize_observations,
        normalize_rewards=False,
        training=False,
        gamma=config.gamma,
    )

    model = make_model(
        env=train_env,
        config=config,
        device=resolved_device,
    )

    callbacks: List[BaseCallback] = [
        TqdmProgressCallback(total_timesteps=config.timesteps),
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq, 1),
            save_path=str(checkpoints_dir),
            name_prefix="ppo_ems_checkpoint",
            save_vecnormalize=True,
        ),
        DispatchMetricsCallback(
            output_csv=training_dispatch_csv,
            dispatch_log_path=training_dispatch_log,
            log_every_steps=config.log_every_steps,
            flush_every=500,
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval_callback"),
            eval_freq=max(config.eval_freq, 1),
            n_eval_episodes=max(2, min(config.eval_episodes, 5)),
            deterministic=True,
            render=False,
        ),
    ]

    LOGGER.info("Beginning PPO training")
    model.learn(total_timesteps=config.timesteps, callback=CallbackList(callbacks))
    LOGGER.info("Training finished, saving model and normalization statistics")

    model_path = run_dir / "ems_dispatch_maskable_ppo"
    model.save(str(model_path))
    train_env.save(str(run_dir / "vec_normalize.pkl"))

    train_env.training = False
    train_env.norm_reward = False
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.training = False
    eval_env.norm_reward = False

    det_dispatch_df, det_episode_df, det_summary = run_policy_evaluation(
        model=model,
        vec_env=eval_env,
        n_episodes=config.eval_episodes,
        deterministic=True,
        dispatch_log_path=eval_dispatch_log_det,
    )
    stoch_dispatch_df, stoch_episode_df, stoch_summary = run_policy_evaluation(
        model=model,
        vec_env=eval_env,
        n_episodes=config.eval_episodes,
        deterministic=False,
        dispatch_log_path=eval_dispatch_log_stoch,
    )

    det_dispatch_df.to_csv(run_dir / "evaluation_dispatch_deterministic.csv", index=False)
    det_episode_df.to_csv(run_dir / "evaluation_episodes_deterministic.csv", index=False)
    stoch_dispatch_df.to_csv(run_dir / "evaluation_dispatch_stochastic.csv", index=False)
    stoch_episode_df.to_csv(run_dir / "evaluation_episodes_stochastic.csv", index=False)

    save_json(run_dir / "evaluation_summary_deterministic.json", det_summary)
    save_json(run_dir / "evaluation_summary_stochastic.json", stoch_summary)

    plot_training_monitor(train_monitor_csv, plots_dir / "training_monitor.png", config.plot_rolling_window)
    plot_dispatch_analysis(det_dispatch_df, plots_dir, prefix="deterministic")
    plot_dispatch_analysis(stoch_dispatch_df, plots_dir, prefix="stochastic")
    plot_episode_analysis(det_episode_df, plots_dir, prefix="deterministic")
    plot_episode_analysis(stoch_episode_df, plots_dir, prefix="stochastic")

    final_summary = {
        "run_dir": str(run_dir),
        "device_info": device_info,
        "dataset_summary": dataset_summary,
        "deterministic_evaluation": det_summary,
        "stochastic_evaluation": stoch_summary,
        "saved_model": str(model_path) + ".zip",
        "saved_vec_normalize": str(run_dir / "vec_normalize.pkl"),
    }
    save_json(run_dir / "run_summary.json", final_summary)

    LOGGER.info("Deterministic evaluation summary: %s", json.dumps(det_summary, default=json_default))
    LOGGER.info("Stochastic evaluation summary: %s", json.dumps(stoch_summary, default=json_default))
    LOGGER.info("Run complete. Final summary written to %s", run_dir / "run_summary.json")
    LOGGER.info("Final summary: %s", json.dumps(final_summary, default=json_default))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
