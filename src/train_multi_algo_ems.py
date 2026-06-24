"""Train multiple RL algorithms on the EMS candidate-action environment.

Run with:
    python train_multi_algo_ems.py --algorithm dqn --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import rl_train
from custom_rl_agents import DQNAgent, DQNConfig, DiscreteSACAgent, DiscreteSACConfig
from ems_candidate_env import CandidateActionEnv


LOGGER = logging.getLogger("ems_multi_algo")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_file: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class TrainConfig:
    algorithm: str
    hospitals: str
    ambulances: str
    output_root: str
    run_name: Optional[str]
    seed: int
    timesteps: int
    episode_days: int
    eval_episodes: int
    device: str
    prefer_external_generator: bool
    avg_speed_kmh: float
    travel_noise_std: float
    num_candidates: int
    learning_rate: float
    gamma: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    target_update_interval: int
    epsilon_decay_steps: int
    ppo_n_steps: int
    ppo_n_epochs: int
    log_every_steps: int
    eval_every_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple EMS RL algorithms")
    parser.add_argument("--algorithm", required=True, choices=["dqn", "ddqn", "dueling_dqn", "sac", "ppo_variant"])
    parser.add_argument("--hospitals", default="HOSPITALS_DATA_FINAL.xlsx")
    parser.add_argument("--ambulances", default="final_ambulance_deployment.csv")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--episode-days", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefer-external-generator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--avg-speed-kmh", type=float, default=40.0)
    parser.add_argument("--travel-noise-std", type=float, default=0.1)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--ppo-n-epochs", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=2_000)
    parser.add_argument("--eval-every-steps", type=int, default=20_000)
    return parser.parse_args()


def build_base_env(config: TrainConfig, seed_offset: int = 0) -> rl_train.EMSEnv:
    hospital_df = rl_train.load_hospitals(config.hospitals)
    ambulance_df = rl_train.load_ambulances(config.ambulances)
    incident_generator = rl_train.make_incident_generator(
        seed=config.seed + seed_offset,
        prefer_external=config.prefer_external_generator,
    )
    return rl_train.EMSEnv(
        hospital_df=hospital_df,
        ambulance_df=ambulance_df,
        incident_generator_fn=incident_generator,
        episode_days=config.episode_days,
        avg_speed_kmh=config.avg_speed_kmh,
        travel_noise_std=config.travel_noise_std,
        seed=config.seed + seed_offset,
        device=config.device,
    )


def build_candidate_env(config: TrainConfig, seed_offset: int = 0) -> CandidateActionEnv:
    base_env = build_base_env(config, seed_offset=seed_offset)
    return CandidateActionEnv(base_env=base_env, num_candidates=config.num_candidates)


def evaluate_custom_agent(agent, config: TrainConfig, deterministic: bool = True) -> Dict[str, Any]:
    episode_rewards: List[float] = []
    dispatch_counts: List[int] = []
    env = build_candidate_env(config, seed_offset=10000)

    for episode in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.seed + 10000 + episode)
        done = False
        total_reward = 0.0
        dispatches = 0
        while not done:
            action = agent.act(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            dispatches += 1
        episode_rewards.append(total_reward)
        dispatch_counts.append(dispatches)

    env.close()
    return {
        "episodes": config.eval_episodes,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "mean_dispatches": float(np.mean(dispatch_counts)),
        "deterministic": deterministic,
    }


def train_custom_agent(config: TrainConfig, run_dir: Path) -> None:
    env = build_candidate_env(config)
    obs, _ = env.reset(seed=config.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    device = rl_train.get_training_device(config.device)

    if config.algorithm == "sac":
        agent = DiscreteSACAgent(
            DiscreteSACConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                learning_starts=config.learning_starts,
                train_freq=config.train_freq,
            ),
            device=device,
        )
    else:
        agent = DQNAgent(
            DQNConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                learning_starts=config.learning_starts,
                train_freq=config.train_freq,
                target_update_interval=config.target_update_interval,
                epsilon_decay_steps=config.epsilon_decay_steps,
                double_dqn=config.algorithm in {"ddqn"},
                dueling=config.algorithm in {"dueling_dqn"},
            ),
            device=device,
        )

    episode_reward = 0.0
    episode_len = 0
    episode_idx = 0
    episode_records: List[Dict[str, Any]] = []
    best_eval_reward = -float("inf")

    for step in range(1, config.timesteps + 1):
        action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += float(reward)
        episode_len += 1

        if step % max(config.train_freq, 1) == 0:
            train_metrics = agent.train_step()
        else:
            train_metrics = {}

        if step % config.log_every_steps == 0:
            LOGGER.info(
                "TRAIN | algo=%s step=%s reward=%.3f epsilon=%s train=%s",
                config.algorithm,
                step,
                episode_reward,
                f"{train_metrics.get('epsilon', 'n/a'):.4f}" if "epsilon" in train_metrics else "n/a",
                train_metrics,
            )

        if step % config.eval_every_steps == 0:
            eval_summary = evaluate_custom_agent(agent, config, deterministic=True)
            LOGGER.info("EVAL | algo=%s step=%s summary=%s", config.algorithm, step, json.dumps(eval_summary))
            if eval_summary["mean_episode_reward"] > best_eval_reward:
                best_eval_reward = eval_summary["mean_episode_reward"]
                agent.save(str(run_dir / "best_model.pt"))

        if done:
            episode_idx += 1
            episode_records.append(
                {
                    "episode": episode_idx,
                    "reward": episode_reward,
                    "length": episode_len,
                    "step": step,
                }
            )
            LOGGER.info(
                "EPISODE END | algo=%s episode=%s reward=%.3f length=%s",
                config.algorithm,
                episode_idx,
                episode_reward,
                episode_len,
            )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

    final_eval_det = evaluate_custom_agent(agent, config, deterministic=True)
    final_eval_stoch = evaluate_custom_agent(agent, config, deterministic=False)
    agent.save(str(run_dir / f"{config.algorithm}_final.pt"))
    pd.DataFrame(episode_records).to_csv(run_dir / "training_episodes.csv", index=False)
    save_json(run_dir / "evaluation_summary_deterministic.json", final_eval_det)
    save_json(run_dir / "evaluation_summary_stochastic.json", final_eval_stoch)
    save_json(
        run_dir / "run_summary.json",
        {
            "algorithm": config.algorithm,
            "run_dir": str(run_dir),
            "deterministic_evaluation": final_eval_det,
            "stochastic_evaluation": final_eval_stoch,
        },
    )
    env.close()


def train_ppo_variant(config: TrainConfig, run_dir: Path) -> None:
    device = rl_train.get_training_device(config.device)

    def make_env():
        env = build_candidate_env(config)
        return Monitor(env, filename=str(run_dir / "train_monitor.csv"))

    vec_env = VecMonitor(DummyVecEnv([make_env]))
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.batch_size,
        n_epochs=config.ppo_n_epochs,
        gamma=config.gamma,
        verbose=1,
        seed=config.seed,
        device=device,
    )
    model.learn(total_timesteps=config.timesteps)
    model.save(str(run_dir / "ppo_variant_model"))
    vec_env.close()

    def eval_policy(deterministic: bool) -> Dict[str, Any]:
        env = build_candidate_env(config, seed_offset=10000)
        rewards = []
        lengths = []
        for episode in range(config.eval_episodes):
            obs, _ = env.reset(seed=config.seed + 10000 + episode)
            done = False
            total_reward = 0.0
            length = 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += float(reward)
                done = terminated or truncated
                length += 1
            rewards.append(total_reward)
            lengths.append(length)
        env.close()
        return {
            "episodes": config.eval_episodes,
            "deterministic": deterministic,
            "mean_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
            "mean_dispatches": float(np.mean(lengths)),
        }

    det_summary = eval_policy(deterministic=True)
    stoch_summary = eval_policy(deterministic=False)
    save_json(run_dir / "evaluation_summary_deterministic.json", det_summary)
    save_json(run_dir / "evaluation_summary_stochastic.json", stoch_summary)
    save_json(
        run_dir / "run_summary.json",
        {
            "algorithm": config.algorithm,
            "run_dir": str(run_dir),
            "deterministic_evaluation": det_summary,
            "stochastic_evaluation": stoch_summary,
        },
    )


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"{config.algorithm}_{timestamp}"
    run_dir = ensure_dir(Path(config.output_root) / run_name)
    setup_logging(run_dir / "run.log")
    save_json(run_dir / "config.json", asdict(config))

    LOGGER.info("Starting EMS algorithm run: %s", run_name)
    LOGGER.info("Config: %s", json.dumps(asdict(config)))

    if config.algorithm == "ppo_variant":
        train_ppo_variant(config, run_dir)
    else:
        train_custom_agent(config, run_dir)

    LOGGER.info("Run complete: %s", run_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
