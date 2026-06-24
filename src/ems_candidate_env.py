"""Candidate-action wrapper around EMSEnv.

WHY THIS FILE EXISTS
---------------------
`rl_train.EMSEnv` exposes a flat `Discrete(N_AMB * N_HOSP)` action space
(46 * 155 = 7130 actions). That is fine for MaskablePPO, because
sb3-contrib's MaskablePPO accepts an explicit boolean mask at every step
and simply zeroes out the logits of invalid actions before sampling --
the policy network still has 7130 output logits, but the *effective*
branching factor at any decision point is tiny (at most
top_k_ambulances * top_k_hospitals <= 6*8 = 48 feasible pairs).

Standard off-policy algorithms (DQN, Double DQN, Dueling DQN, and the
discrete SAC variant implemented in `custom_rl_agents.py`) have no
first-class concept of an action mask in their textbook formulation:
- DQN's argmax over Q-values would happily pick an action that is
  invalid (ambulance busy, hospital full, wrong specialty) unless we
  either (a) mask the Q-values manually inside the agent, or
  (b) shrink the action space itself so every action the agent can
  emit is guaranteed to be legal.

This module takes approach (b), which is the standard trick used in
dispatch/assignment RL papers when you want to compare a masked
policy-gradient method against vanilla value-based baselines without
reimplementing masking logic three more times: at every decision
point we compute the (small) set of feasible (ambulance, hospital)
pairs from the *real* environment, rank/truncate it down to
`num_candidates` slots, and expose that as a flat
`Discrete(num_candidates)` action space. Index i in the candidate
action space simply means "dispatch using the i-th feasible pair",
and that mapping is recomputed every single step because the set of
feasible pairs changes as ambulances move and beds fill up.

DESIGN CHOICES (useful for explaining in an interview)
-------------------------------------------------------
1. Candidate ranking: feasible (ambulance, hospital) pairs are sorted
   by total ETA (tau_scene + tau_hospital), the same quantity the
   reward function cares most about (eta_gap and golden-hour terms).
   This biases the candidate list towards "obviously reasonable"
   choices, which keeps the action space small without hiding good
   options from the agent.
2. Padding: if fewer than `num_candidates` feasible pairs exist, the
   remaining slots are filled by repeating the best candidate. This
   keeps the action space size constant (required by Discrete) while
   guaranteeing that *every* index in [0, num_candidates) maps to a
   valid action -- the agent can never accidentally "select padding"
   and get an invalid-action penalty.
3. Observation augmentation: the wrapper concatenates the base EMSEnv
   observation with a flattened block describing the current
   candidate list (normalized ETA, ambulance type, hospital
   occupancy ratio for each of the num_candidates slots). Without
   this, a value-based agent looking only at the *global* state would
   have no way to know what "action 3" even refers to on this step,
   since the same integer can map to a different (ambulance, hospital)
   pair from one decision point to the next.
4. No-incident / no-feasible-action edge cases: if the episode horizon
   is reached, or (rarely) no feasible pair exists for the current
   incident even before pruning, we terminate the episode rather than
   stepping the inner env with a meaningless action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback mirrors rl_train.py
    import gym
    from gym import spaces

import rl_train


class CandidateActionEnv(gym.Env):
    """Wraps EMSEnv's (ambulance x hospital) action space into a small,
    fixed-size, always-valid Discrete(num_candidates) action space.

    Parameters
    ----------
    base_env:
        An instantiated `rl_train.EMSEnv`.
    num_candidates:
        Number of feasible (ambulance, hospital) pairs exposed to the
        agent at every decision point. Smaller values make the
        learning problem easier (fewer actions to value) but can
        exclude legitimately good options on busy days; larger values
        do the opposite. 16 is a reasonable middle ground for this
        environment's typical feasible-set sizes (which are bounded by
        top_k_ambulances_by_severity * top_k_hospitals_by_severity,
        i.e. at most 6*8=48 for LOW severity, 4*6=24 for MEDIUM, and
        3*4=12 for HIGH).
    """

    metadata = {"render_modes": ["human"]}

    # Per-candidate feature block:
    #   [tau_scene_norm, tau_hosp_norm, tau_total_norm,
    #    is_ALS, is_BLS, occupancy_ratio, is_best_feasible]
    CANDIDATE_FEATURE_DIM = 7

    def __init__(self, base_env: "rl_train.EMSEnv", num_candidates: int = 16):
        super().__init__()
        self.base_env = base_env
        self.num_candidates = int(num_candidates)

        base_obs_dim = int(np.prod(self.base_env.observation_space.shape))
        candidate_block_dim = self.num_candidates * self.CANDIDATE_FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + candidate_block_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_candidates)

        # Populated by _refresh_candidates(); maps candidate slot -> flat
        # action index understood by the base EMSEnv (amb_idx * N_HOSP + hosp_idx).
        self._candidate_flat_actions: List[int] = []
        self._candidate_meta: List[Dict[str, Any]] = []
        self._base_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._base_obs, info = self.base_env.reset(seed=seed, options=options)
        self._refresh_candidates()
        return self._build_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if not self._candidate_flat_actions:
            # No feasible action existed even before this step was requested
            # (can only happen if reset/refresh produced an empty incident
            # queue at the horizon boundary). Treat as terminal.
            return self._build_obs(), 0.0, True, False, {"invalid_action": True, "reason": "no_candidates"}

        # Clip defensively: Discrete(n) guarantees 0 <= action < n, but we
        # guard anyway since padded slots duplicate the best candidate.
        candidate_idx = min(max(action, 0), len(self._candidate_flat_actions) - 1)
        flat_action = self._candidate_flat_actions[candidate_idx]

        base_obs, reward, terminated, truncated, info = self.base_env.step(flat_action)
        self._base_obs = base_obs
        info["candidate_index"] = candidate_idx
        info["candidate_pool_size"] = len(self._candidate_meta)

        if not (terminated or truncated):
            self._refresh_candidates()
            if not self._candidate_flat_actions:
                # Decision point claimed an incident was ready+feasible,
                # but the world changed by the time we got here (shouldn't
                # normally happen since EMSEnv re-checks at _run_until_decision).
                terminated = True

        return self._build_obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        return self.base_env.render(mode=mode)

    def close(self):
        return self.base_env.close() if hasattr(self.base_env, "close") else None

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def _refresh_candidates(self) -> None:
        """Recompute the feasible (ambulance, hospital) shortlist for the
        incident currently selected by the base environment, sorted by
        total ETA, then store flat action indices + features for the obs.
        """
        incident = self.base_env._select_incident()
        feasible = self.base_env._get_feasible_actions(incident, apply_pruning=True)
        # feasible: List[(amb_idx, hosp_idx, tau_scene, tau_total)]
        feasible = sorted(feasible, key=lambda item: item[3])[: self.num_candidates]

        self._candidate_flat_actions = []
        self._candidate_meta = []

        if not feasible:
            return

        for amb_idx, hosp_idx, tau_scene, tau_total in feasible:
            tau_hosp = tau_total - tau_scene
            flat_action = amb_idx * self.base_env.N_HOSP + hosp_idx
            self._candidate_flat_actions.append(flat_action)
            self._candidate_meta.append(
                {
                    "amb_idx": amb_idx,
                    "hosp_idx": hosp_idx,
                    "tau_scene": tau_scene,
                    "tau_hosp": tau_hosp,
                    "tau_total": tau_total,
                }
            )

        # Pad to a fixed width by repeating the best (lowest-ETA) candidate,
        # which is always index 0 after sorting. This keeps Discrete(n) a
        # constant size while ensuring every index is a legal dispatch.
        if len(self._candidate_flat_actions) < self.num_candidates:
            pad_count = self.num_candidates - len(self._candidate_flat_actions)
            self._candidate_flat_actions.extend([self._candidate_flat_actions[0]] * pad_count)
            self._candidate_meta.extend([dict(self._candidate_meta[0])] * pad_count)

    def _build_obs(self) -> np.ndarray:
        base = (
            self._base_obs
            if self._base_obs is not None
            else np.zeros(self.base_env.observation_space.shape, dtype=np.float32)
        )
        base = np.asarray(base, dtype=np.float32).reshape(-1)

        candidate_block = np.zeros(
            (self.num_candidates, self.CANDIDATE_FEATURE_DIM), dtype=np.float32
        )
        max_travel = float(self.base_env.max_travel_time)

        for slot in range(self.num_candidates):
            if slot >= len(self._candidate_meta):
                continue
            meta = self._candidate_meta[slot]
            amb = self.base_env.ambulances[meta["amb_idx"]]
            hosp = self.base_env.hospitals[meta["hosp_idx"]]
            incident = self.base_env._select_incident()

            tau_scene_norm = min(meta["tau_scene"] / max_travel, 1.0)
            tau_hosp_norm = min(meta["tau_hosp"] / max_travel, 1.0)
            tau_total_norm = min(meta["tau_total"] / max_travel, 1.0)
            is_als = 1.0 if amb["type"] == "ALS" else 0.0
            is_bls = 1.0 if amb["type"] == "BLS" else 0.0

            occupancy_ratio = 0.0
            if incident is not None:
                bed_type = self.base_env.SEV_TO_BEDTYPE[incident["severity"]]
                if bed_type == "icu":
                    occupancy_ratio = hosp["occ_icu"] / max(hosp["icu_beds"], 1)
                elif bed_type == "ventilator":
                    occupancy_ratio = hosp["occ_vent"] / max(hosp["ventilator_beds"], 1)
                else:
                    occupancy_ratio = hosp["occ_oxygen"] / max(hosp["oxygen_beds"], 1)

            is_best = 1.0 if slot == 0 else 0.0

            candidate_block[slot] = [
                tau_scene_norm,
                tau_hosp_norm,
                tau_total_norm,
                is_als,
                is_bls,
                occupancy_ratio,
                is_best,
            ]

        return np.concatenate([base, candidate_block.reshape(-1)]).astype(np.float32)
