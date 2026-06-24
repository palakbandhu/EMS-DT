"""Supervisory agent for validating and explaining EMS dispatch decisions.

This module wraps the trained MaskablePPO policy with:
- hard safety validation from the RL environment,
- ranked feasible alternatives,
- natural-language explanations via LangChain + Hugging Face.

Example:
    python decision_support_agent.py \
        --model-path artifacts/ppo_ems_20260510_043018/best_model \
        --top-k 3 \
        --json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import rl_train


DEFAULT_MODEL_PATH = "artifacts/ppo_ems_20260510_043018/best_model"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def ensure_runtime_dirs(base_dir: Optional[Path] = None) -> Dict[str, str]:
    """Create writable temp/cache directories for torch and HF runtimes."""
    root = (base_dir or Path.cwd()) / ".runtime"
    tmp_dir = root / "tmp"
    torch_cache_dir = root / "torchinductor"
    hf_cache_dir = root / "huggingface"

    for path in (root, tmp_dir, torch_cache_dir, hf_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torch_cache_dir))
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache_dir / "hub"))

    return {
        "runtime_root": str(root),
        "tmp_dir": str(tmp_dir),
        "torch_cache_dir": str(torch_cache_dir),
        "hf_cache_dir": str(hf_cache_dir),
    }


@dataclass
class ActionAssessment:
    action: int
    ambulance_index: int
    ambulance_id: str
    ambulance_type: str
    hospital_index: int
    hospital_name: str
    estimated_response_eta_min: float
    estimated_hospital_eta_min: float
    estimated_total_reward: float
    occupancy_ratio: float
    valid: bool
    reasons: List[str]
    rank: Optional[int] = None


class DispatchDecisionSupportAgent:
    """Supervisory layer on top of the trained dispatch policy."""

    def __init__(
        self,
        model_path: str | Path,
        hospitals_path: Optional[str] = None,
        ambulances_path: Optional[str] = None,
        hf_model_id: str = DEFAULT_HF_MODEL,
        device: str = "auto",
        deterministic_policy: bool = True,
    ):
        load_dotenv()
        self.runtime_dirs = ensure_runtime_dirs(Path.cwd())
        self.model_dir, self.model_file = self._resolve_model_path(Path(model_path))
        self.artifact_dir = self.model_dir.parent
        self.config = self._load_training_config(self.artifact_dir / "config.json")
        self.hospitals_path = hospitals_path or self.config.get("hospitals", "HOSPITALS_DATA_FINAL.xlsx")
        self.ambulances_path = ambulances_path or self.config.get("ambulances", "final_ambulance_deployment.csv")
        self.deterministic_policy = deterministic_policy

        hospital_df = rl_train.load_hospitals(self.hospitals_path)
        ambulance_df = rl_train.load_ambulances(self.ambulances_path)
        env_device = self.config.get("device", device)
        training_device = rl_train.get_training_device(device if device != "auto" else env_device)

        self.vec_env = self._build_inference_env(
            hospital_df=hospital_df,
            ambulance_df=ambulance_df,
            device=env_device,
        )
        self.base_env = self._unwrap_base_env(self.vec_env)
        self.model = MaskablePPO.load(str(self.model_file), env=self.vec_env, device=training_device)
        self.explainer = self._build_explainer(hf_model_id)
        self.obs = self.vec_env.reset()

    @staticmethod
    def _resolve_model_path(model_path: Path) -> Tuple[Path, Path]:
        if model_path.is_dir():
            candidate = model_path / "best_model.zip"
            if candidate.exists():
                return model_path, candidate
        if model_path.is_file():
            return model_path.parent, model_path
        raise FileNotFoundError(f"Could not locate PPO model at {model_path}")

    @staticmethod
    def _load_training_config(config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_inference_env(
        self,
        hospital_df,
        ambulance_df,
        device: str,
    ):
        seed = int(self.config.get("seed", 42))
        episode_days = int(self.config.get("episode_days", rl_train.EMSEnv.DEFAULT_EPISODE_DAYS))
        prefer_external_generator = bool(self.config.get("prefer_external_generator", True))
        avg_speed_kmh = float(self.config.get("avg_speed_kmh", 40.0))
        travel_noise_std = float(self.config.get("travel_noise_std", 0.1))

        def make_env():
            env = rl_train.EMSEnv(
                hospital_df=hospital_df,
                ambulance_df=ambulance_df,
                incident_generator_fn=rl_train.make_incident_generator(
                    seed=seed,
                    prefer_external=prefer_external_generator,
                ),
                episode_days=episode_days,
                avg_speed_kmh=avg_speed_kmh,
                travel_noise_std=travel_noise_std,
                seed=seed,
                device=device,
            )
            env = Monitor(env)
            env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.action_mask())
            return env

        vec_env = DummyVecEnv([make_env])
        vecnormalize_path = self._find_vecnormalize_path()
        if vecnormalize_path is not None:
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        return vec_env

    def _find_vecnormalize_path(self) -> Optional[Path]:
        direct_path = self.artifact_dir / "vec_normalize.pkl"
        if direct_path.exists():
            return direct_path

        checkpoint_dir = self.artifact_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        candidates = sorted(checkpoint_dir.glob("*vecnormalize*.pkl"))
        if candidates:
            return candidates[-1]
        return None

    @staticmethod
    def _unwrap_base_env(vec_env):
        wrapped = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
        return wrapped.envs[0].unwrapped

    @staticmethod
    def _build_explainer(hf_model_id: str):
        hf_token = (
            os.getenv("hf_token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not hf_token:
            return None

        endpoint = HuggingFaceEndpoint(
            repo_id=hf_model_id,
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=320,
            temperature=0.2,
            repetition_penalty=1.05,
        )
        chat_model = ChatHuggingFace(llm=endpoint)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a clinical EMS dispatch supervisor. Explain decisions clearly for a human operator. "
                    "Keep the answer concise, factual, and operationally useful. Mention safety validation, the main decision drivers, "
                    "and whether a fallback option should be considered.",
                ),
                (
                    "human",
                    "Incident summary:\n{incident}\n\n"
                    "Recommended dispatch:\n{recommendation}\n\n"
                    "Validation summary:\n{validation}\n\n"
                    "Top alternatives:\n{alternatives}\n\n"
                    "Write 1 short paragraph plus one final sentence stating whether you would approve this dispatch.",
                ),
            ]
        )
        return prompt | chat_model | StrOutputParser()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.base_env.seed(seed)
        self.obs = self.vec_env.reset()
        return self.current_state_summary()

    def current_state_summary(self) -> Dict[str, Any]:
        incident = self.base_env._select_incident()
        return {
            "current_time_sec": float(self.base_env.current_time),
            "traffic_multiplier": float(self.base_env.traffic_multiplier),
            "queue_size": int(len(self.base_env.incident_queue)),
            "idle_ambulances": int(sum(amb["status"] == "IDLE" for amb in self.base_env.ambulances)),
            "incident": self._serialize_incident(incident, self.base_env.current_time),
        }

    def recommend(self, top_k: int = 3, deterministic: Optional[bool] = None) -> Dict[str, Any]:
        deterministic = self.deterministic_policy if deterministic is None else deterministic
        incident = self.base_env._select_incident()
        if incident is None:
            return {
                "status": "no_active_incident",
                "message": "There is no active incident waiting for dispatch.",
                "state": self.current_state_summary(),
            }

        action_mask = self.base_env.action_mask()
        feasible_actions = np.flatnonzero(action_mask)
        if feasible_actions.size == 0:
            return {
                "status": "no_feasible_action",
                "message": "No feasible dispatch pair is available for the current incident.",
                "state": self.current_state_summary(),
            }

        model_masks = np.asarray([action_mask])
        action, _ = self.model.predict(self.obs, deterministic=deterministic, action_masks=model_masks)
        chosen_action = int(np.asarray(action).reshape(-1)[0])

        ranked_actions = self._rank_feasible_actions(incident, feasible_actions.tolist())
        chosen_assessment = next((item for item in ranked_actions if item.action == chosen_action), None)
        if chosen_assessment is None:
            chosen_assessment = self._assess_action(chosen_action, incident)
            ranked_actions.append(chosen_assessment)
            ranked_actions.sort(key=lambda item: item.estimated_total_reward, reverse=True)
            for idx, item in enumerate(ranked_actions, start=1):
                item.rank = idx

        validation = self._build_validation_summary(chosen_assessment, incident, ranked_actions)
        alternatives = [asdict(item) for item in ranked_actions if item.action != chosen_action][:top_k]
        recommendation = asdict(chosen_assessment)
        explanation = self._generate_explanation(
            incident=incident,
            recommendation=recommendation,
            validation=validation,
            alternatives=alternatives,
        )

        return {
            "status": "ok",
            "state": self.current_state_summary(),
            "recommended_action": recommendation,
            "validation": validation,
            "alternatives": alternatives,
            "explanation": explanation,
        }

    def apply_action(self, action: Optional[int] = None) -> Dict[str, Any]:
        if action is None:
            action_mask = np.asarray([self.base_env.action_mask()])
            action, _ = self.model.predict(
                self.obs,
                deterministic=self.deterministic_policy,
                action_masks=action_mask,
            )
            action = int(np.asarray(action).reshape(-1)[0])

        self.obs, rewards, dones, infos = self.vec_env.step(np.array([action]))
        return {
            "action": int(action),
            "reward": float(np.asarray(rewards).reshape(-1)[0]),
            "done": bool(np.asarray(dones).reshape(-1)[0]),
            "info": infos[0] if infos else {},
        }

    def _rank_feasible_actions(self, incident: Dict[str, Any], actions: Sequence[int]) -> List[ActionAssessment]:
        assessments = [self._assess_action(action, incident) for action in actions]
        assessments.sort(key=lambda item: item.estimated_total_reward, reverse=True)
        for idx, item in enumerate(assessments, start=1):
            item.rank = idx
        return assessments

    def _assess_action(self, action: int, incident: Dict[str, Any]) -> ActionAssessment:
        amb_idx, hosp_idx = divmod(int(action), self.base_env.N_HOSP)
        ambulance = self.base_env.ambulances[amb_idx]
        hospital = self.base_env.hospitals[hosp_idx]

        valid = self.base_env._is_action_valid(amb_idx, hosp_idx, incident)
        reasons: List[str] = []
        if not valid:
            reasons.append("Fails hard environment constraints.")

        tau_scene, tau_hosp, tau_total = self._estimate_travel_times(ambulance, hospital, incident)
        best_metrics = self.base_env._compute_best_feasible_eta_metrics(incident)
        reward, _ = self.base_env._compute_reward(
            incident=incident,
            ambulance=ambulance,
            hospital=hospital,
            tau_scene=tau_scene,
            tau_hosp=tau_hosp,
            tau_total=tau_total,
            best_feasible_metrics=best_metrics,
        )
        occupancy_ratio = self._hospital_occupancy_ratio(hospital, incident["severity"])

        if incident["severity"] == "HIGH" and ambulance["type"] != "ALS":
            reasons.append("High-severity patient assigned to a BLS unit.")
        if incident["severity"] == "HIGH" and (tau_total / 60.0) > 60.0:
            reasons.append("Projected hospital arrival exceeds the golden-hour target.")
        if best_metrics["best_feasible_response_eta_min"] is not None:
            eta_gap = (tau_scene / 60.0) - float(best_metrics["best_feasible_response_eta_min"])
            if eta_gap > 5.0:
                reasons.append(f"Response ETA is {eta_gap:.1f} minutes slower than the best feasible unit.")
        if occupancy_ratio >= 0.9:
            reasons.append("Destination bed occupancy is critically high.")

        return ActionAssessment(
            action=int(action),
            ambulance_index=int(amb_idx),
            ambulance_id=str(ambulance["id"]),
            ambulance_type=str(ambulance["type"]),
            hospital_index=int(hosp_idx),
            hospital_name=str(hospital["name"]),
            estimated_response_eta_min=float(tau_scene / 60.0),
            estimated_hospital_eta_min=float(tau_total / 60.0),
            estimated_total_reward=float(reward),
            occupancy_ratio=float(occupancy_ratio),
            valid=bool(valid),
            reasons=reasons,
        )

    def _build_validation_summary(
        self,
        chosen: ActionAssessment,
        incident: Dict[str, Any],
        ranked_actions: Sequence[ActionAssessment],
    ) -> Dict[str, Any]:
        best = ranked_actions[0]
        risky = (not chosen.valid) or bool(chosen.reasons)
        return {
            "hard_constraints_passed": bool(chosen.valid),
            "risky_assignment": bool(risky),
            "risk_flags": list(chosen.reasons),
            "incident_severity": incident["severity"],
            "policy_rank_among_feasible": int(chosen.rank or -1),
            "best_feasible_action": asdict(best),
            "response_eta_gap_min": float(
                chosen.estimated_response_eta_min - best.estimated_response_eta_min
            ),
            "hospital_eta_gap_min": float(
                chosen.estimated_hospital_eta_min - best.estimated_hospital_eta_min
            ),
            "reward_gap_vs_best": float(
                chosen.estimated_total_reward - best.estimated_total_reward
            ),
        }

    def _generate_explanation(
        self,
        incident: Dict[str, Any],
        recommendation: Dict[str, Any],
        validation: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
    ) -> str:
        if self.explainer is None:
            return self._fallback_explanation(incident, recommendation, validation, alternatives)

        try:
            return self.explainer.invoke(
                {
                    "incident": json.dumps(
                        self._serialize_incident(incident, self.base_env.current_time),
                        indent=2,
                    ),
                    "recommendation": json.dumps(recommendation, indent=2),
                    "validation": json.dumps(validation, indent=2),
                    "alternatives": json.dumps(alternatives, indent=2),
                }
            ).strip()
        except Exception:
            return self._fallback_explanation(incident, recommendation, validation, alternatives)

    @staticmethod
    def _fallback_explanation(
        incident: Dict[str, Any],
        recommendation: Dict[str, Any],
        validation: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
    ) -> str:
        risk_text = "Risk flags: none." if not validation["risk_flags"] else f"Risk flags: {', '.join(validation['risk_flags'])}."
        alt_text = "No alternative dispatches were ranked." if not alternatives else (
            f"Top fallback is ambulance {alternatives[0]['ambulance_id']} to {alternatives[0]['hospital_name']} "
            f"with ETA {alternatives[0]['estimated_hospital_eta_min']:.1f} minutes."
        )
        return (
            f"The policy recommends ambulance {recommendation['ambulance_id']} ({recommendation['ambulance_type']}) "
            f"to {recommendation['hospital_name']} for a {incident['severity']} severity case needing {incident['specialty']}. "
            f"Estimated response ETA is {recommendation['estimated_response_eta_min']:.1f} minutes and hospital arrival ETA is "
            f"{recommendation['estimated_hospital_eta_min']:.1f} minutes. Hard safety checks "
            f"{'passed' if validation['hard_constraints_passed'] else 'failed'}, and the policy rank among feasible actions is "
            f"{validation['policy_rank_among_feasible']}. {risk_text} {alt_text}"
        )

    def _estimate_travel_times(
        self,
        ambulance: Dict[str, Any],
        hospital: Dict[str, Any],
        incident: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        dist_to_scene = rl_train.haversine(
            ambulance["lat"],
            ambulance["lon"],
            incident["lat"],
            incident["lon"],
        )
        dist_to_hospital = rl_train.haversine(
            incident["lat"],
            incident["lon"],
            hospital["lat"],
            hospital["lon"],
        )
        speed_ms = (self.base_env.avg_speed_kmh * 1000.0 / 3600.0) / self.base_env.traffic_multiplier
        tau_scene = dist_to_scene / speed_ms
        tau_hosp = dist_to_hospital / speed_ms
        return tau_scene, tau_hosp, tau_scene + tau_hosp

    def _hospital_occupancy_ratio(self, hospital: Dict[str, Any], severity: str) -> float:
        bed_type = self.base_env.SEV_TO_BEDTYPE[severity]
        if bed_type == "icu":
            occ, cap = hospital["occ_icu"], hospital["icu_beds"]
        elif bed_type == "ventilator":
            occ, cap = hospital["occ_vent"], hospital["ventilator_beds"]
        else:
            occ, cap = hospital["occ_oxygen"], hospital["oxygen_beds"]
        return float(occ / max(cap, 1))

    @staticmethod
    def _serialize_incident(
        incident: Optional[Dict[str, Any]],
        current_time_sec: float,
    ) -> Optional[Dict[str, Any]]:
        if incident is None:
            return None
        return {
            "event_id": int(incident["event_id"]),
            "severity": str(incident["severity"]),
            "specialty": str(incident["specialty"]),
            "lat": float(incident["lat"]),
            "lon": float(incident["lon"]),
            "exception": str(incident["exception"]),
            "current_wait_min": float(max(0.0, current_time_sec - incident.get("time", current_time_sec)) / 60.0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision validation and explanation agent for EMS PPO policy")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hospitals", default=None)
    parser.add_argument("--ambulances", default=None)
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply-action", action="store_true")
    parser.add_argument("--stochastic-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = DispatchDecisionSupportAgent(
        model_path=args.model_path,
        hospitals_path=args.hospitals,
        ambulances_path=args.ambulances,
        hf_model_id=args.hf_model_id,
        device=args.device,
        deterministic_policy=not args.stochastic_policy,
    )
    recommendation = agent.recommend(top_k=args.top_k)
    if args.apply_action and recommendation.get("status") == "ok":
        recommendation["step_result"] = agent.apply_action(
            recommendation["recommended_action"]["action"]
        )

    if args.json:
        print(json.dumps(recommendation, indent=2))
        return

    print("Dispatch Decision Support")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
