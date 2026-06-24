"""Clean RL training environment for EMS dispatch.
EMS Dispatch Environment – Finite-horizon MDP with action masking
Compatible with Gymnasium and Stable-Baselines3 (MaskablePPO).
"""

import numpy as np
import pandas as pd
from collections import deque
import heapq
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, List, Dict, Optional, Any

try:
    import torch
except ImportError:
    torch = None

# gymnasium interface
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class _FallbackEnv:
            metadata = {}

            def reset(self, seed=None, options=None):
                return None

        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackDiscrete:
            def __init__(self, n):
                self.n = n

        class _FallbackSpaces:
            Box = _FallbackBox
            Discrete = _FallbackDiscrete

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

# ------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6371000  # Earth radius in metres
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def resolve_torch_device(device: Optional[str] = None) -> str:
    """Return the requested torch device, falling back safely when unavailable."""
    if torch is None:
        return "cpu"
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device

# ------------------------------------------------------------
# 2.  Load static data (once, outside the class)
# ------------------------------------------------------------
def load_hospitals(excel_path: str) -> pd.DataFrame:
    """Load and clean hospital data, return a DataFrame with all needed columns."""
    df = pd.read_excel(excel_path, header=0)
    # Normalise column names: strip, uppercase
    df.columns = [col.strip().upper() for col in df.columns]
    # Keep the first occurrence when spreadsheets contain duplicate headers.
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    # Rename if necessary (some Excel files may have slightly different names)
    rename_map = {
        'HOSPITAL NAME': 'name',
        'LATITUDE': 'lat',
        'LONGITUDE': 'lon',
        'TOTAL BED COUNT': 'total_beds',
        'ICU BEDS': 'icu_beds',
        'VENTILATOR BEDS': 'ventilator_beds',
        'OXYGEN BEDS': 'oxygen_beds'
    }
    df.rename(columns=rename_map, inplace=True)
    # Make sure lat/lon are float
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df.dropna(subset=['lat', 'lon'], inplace=True)
    return df

def load_ambulances(csv_path: str) -> pd.DataFrame:
    """Load ambulance deployment from CSV."""
    df = pd.read_csv(csv_path)
    # Expected columns: amb_id, type, home_zone, lat, lon, home_lat, home_lon, status, available_at
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ------------------------------------------------------------
# 3.  Environment definition
# ------------------------------------------------------------
class EMSEnv(gym.Env):
    """
    Emergency Medical Service dispatch environment.
    Action space: Discrete(46 * 155) = 7130  (ambulance × hospital)
    State space: flat vector of approx. 4000 dims.
    """

    metadata = {"render_modes": ["human"]}

    # Default episode horizon (3 days in seconds)
    DEFAULT_EPISODE_DAYS = 3

    # Constants for reward function (see formulation)
    REWARD_WEIGHTS = {
        'match': 1.0,
        'travel': 1.0,
        'eta_gap': 1.5,
        'gh': 1.0,
        'wait': 1.0,
        'load': 0.3,
        'opp': 0.1
    }

    # Rebalanced matching reward matrix (rows: BLS=0, ALS=1; cols: LOW=0, MEDIUM=1, HIGH=2)
    MATCH_REWARD = np.array([
        [ 2, -1, -4],   # BLS
        [ 0,  2,  4]    # ALS
    ])

    # Specialty list (order fixed)
    SPECIALTIES = [
        "CARDIOLOGY", "UROLOGY", "BURNS", "CRITICAL", "NEUROLOGY",
        "NEUROSURGERY", "POLYTRAUMA", "PULMONOLOGY", "MEDICINE", "GENERAL",
        "PEDIATRICS", "ORTHOPEDICS", "GYNAECOLOGY", "NEPHROLOGY", "CARDIOTHORACIC"
    ]
    N_SPEC = len(SPECIALTIES)

    # Exceptional event categories (one-hot)
    EXCEPTIONS = ["NONE", "HIGHWAY_CRASH_BURST", "FESTIVAL_CROWDING", "RAIN_DISRUPTION"]
    N_EXCEPT = len(EXCEPTIONS)

    # Severity mapping to bed type for masking and occupancy tracking
    SEV_TO_BEDTYPE = {
        'HIGH': 'icu',
        'MEDIUM': 'ventilator',
        'LOW': 'oxygen'
    }

    # Traffic multiplier bounds (for normalisation)
    TRAFFIC_MULTIPLIER_MEAN = 1.0
    TRAFFIC_MULTIPLIER_STD = 0.15

    def __init__(self,
                 hospital_df: pd.DataFrame,
                 ambulance_df: pd.DataFrame,
                 incident_generator_fn,
                 episode_days=3,
                 avg_speed_kmh=40.0,
                 travel_noise_std=0.1,     # log-normal std for travel time noise
                 enforce_als_for_high_if_available: bool = True,
                 top_k_ambulances_low: int = 6,
                 top_k_ambulances_medium: int = 4,
                 top_k_ambulances_high: int = 3,
                 top_k_hospitals_low: int = 8,
                 top_k_hospitals_medium: int = 6,
                 top_k_hospitals_high: int = 4,
                 seed=None,
                 device: Optional[str] = "auto"):
        super().__init__()

        # Store data
        self.hospital_df = hospital_df.reset_index(drop=True)
        self.ambulance_df = ambulance_df.reset_index(drop=True)
        self.incident_generator_fn = incident_generator_fn
        self.episode_days = int(episode_days)
        self.EPISODE_HORIZON_SEC = self.episode_days * 24 * 3600
        self.avg_speed_kmh = avg_speed_kmh
        self.travel_noise_std = travel_noise_std
        self.enforce_als_for_high_if_available = bool(enforce_als_for_high_if_available)
        self.top_k_ambulances_by_severity = {
            'LOW': max(int(top_k_ambulances_low), 1),
            'MEDIUM': max(int(top_k_ambulances_medium), 1),
            'HIGH': max(int(top_k_ambulances_high), 1),
        }
        self.top_k_hospitals_by_severity = {
            'LOW': max(int(top_k_hospitals_low), 1),
            'MEDIUM': max(int(top_k_hospitals_medium), 1),
            'HIGH': max(int(top_k_hospitals_high), 1),
        }
        self.device = resolve_torch_device(device)
        self.use_torch = torch is not None
        self.use_gpu = self.use_torch and self.device.startswith("cuda")

        # Numbers
        self.N_AMB = len(self.ambulance_df)          # 46
        self.N_HOSP = len(self.hospital_df)          # 155
        self.N_ACTIONS = self.N_AMB * self.N_HOSP    # 7130

        # Episode internal state (filled at reset)
        self.current_time = 0.0          # seconds since episode start
        self.traffic_multiplier = 1.0
        self.incident_queue: List[Dict] = []   # active unassigned incidents
        self.ambulances: List[Dict] = []
        self.hospitals: List[Dict] = []
        self.event_queue = []             # priority queue of (time, seq, event_type, data)
        self._event_seq = 0
        self.future_incidents = []        # pre-generated incidents not yet arrived, sorted
        self.done = False

        # ---- Build gym spaces ----
        # State vector size: incident queue + ambulances + hospitals + context
        self.state_dim = (30 * (2 + 3 + self.N_SPEC + 1 + self.N_EXCEPT)   # incident features
                          + self.N_AMB * (2 + 2 + 4 + 1)                  # ambulance features
                          + self.N_HOSP * (2 + self.N_SPEC + 4)           # hospital features
                          + 2)                                            # context (time, traffic)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        # Set seed for reproducibility
        self.seed(seed)

        # For normalisation: known bounds
        self.lat_bounds = (19.45, 20.85)   # Nashik district
        self.lon_bounds = (73.15, 74.75)
        self.max_wait_time = self.EPISODE_HORIZON_SEC
        self.max_travel_time = 6 * 3600    # 6 hours
        self.max_sim_time = self.EPISODE_HORIZON_SEC
        self.traffic_bounds = (0.5, 1.8)   # clamp
        self._setup_static_acceleration_buffers()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            if self.use_torch:
                torch.manual_seed(seed)
        return [seed]

    def _setup_static_acceleration_buffers(self):
        """Create vectorized CPU/GPU-friendly views of static hospital metadata."""
        self.specialty_to_idx = {name: idx for idx, name in enumerate(self.SPECIALTIES)}
        self.severity_to_idx = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        self.status_to_idx = {
            'IDLE': 0,
            'EN_ROUTE_TO_SCENE': 1,
            'TRANSPORTING': 2,
            'RETURNING': 3,
        }
        self._hospital_specialties_np = np.stack([
            np.array([int(row.get(s, 0)) for s in self.SPECIALTIES], dtype=np.int32)
            for _, row in self.hospital_df.iterrows()
        ]).copy()
        self._hospital_caps_np = {
            'icu': self.hospital_df['icu_beds'].astype(np.int32).to_numpy(copy=True),
            'ventilator': self.hospital_df['ventilator_beds'].astype(np.int32).to_numpy(copy=True),
            'oxygen': self.hospital_df['oxygen_beds'].astype(np.int32).to_numpy(copy=True),
        }
        if self.use_torch:
            self._hospital_specialties_t = torch.as_tensor(
                self._hospital_specialties_np, device=self.device, dtype=torch.bool
            )
            self._hospital_caps_t = {
                bed_type: torch.as_tensor(values, device=self.device, dtype=torch.int32)
                for bed_type, values in self._hospital_caps_np.items()
            }
        else:
            self._hospital_specialties_t = None
            self._hospital_caps_t = {}

    def _current_occupancy_np(self, bed_type: str) -> np.ndarray:
        key = {
            'icu': 'occ_icu',
            'ventilator': 'occ_vent',
            'oxygen': 'occ_oxygen',
        }[bed_type]
        return np.fromiter((h[key] for h in self.hospitals), dtype=np.int32, count=self.N_HOSP)

    def _idle_ambulance_mask_np(self) -> np.ndarray:
        return np.fromiter(
            (amb['status'] == 'IDLE' for amb in self.ambulances),
            dtype=bool,
            count=self.N_AMB
        )

    def reset(self, seed=None, options=None):
        """Initialise a new episode and advance to first decision point."""
        super().reset(seed=seed)
        self.current_time = 0.0
        self.done = False
        self.last_transition_info = {}
        self.episode_stats = {
            'dispatches': 0,
            'invalid_actions': 0,
            'total_reward': 0.0,
            'incidents_generated': 0,
            'high_priority_generated': 0,
            'completed_patients': 0,
            'severity_counts': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
            'reward_components': {
                'match': 0.0,
                'travel': 0.0,
                'eta_gap': 0.0,
                'gh': 0.0,
                'wait': 0.0,
                'load': 0.0,
                'opp': 0.0,
            },
        }
        # Sample traffic multiplier for the whole episode (or could be piecewise constant)
        self.traffic_multiplier = np.clip(
            np.random.normal(self.TRAFFIC_MULTIPLIER_MEAN, self.TRAFFIC_MULTIPLIER_STD),
            *self.traffic_bounds
        )

        # ---- Generate all incidents for the episode ----
        incident_df = self.incident_generator_fn(sim_days=self.episode_days)
        # Sort by incident_time
        incident_df = incident_df.sort_values('incident_time')
        self.future_incidents = []
        for _, row in incident_df.iterrows():
            incident = {
                'event_id': row['event_id'],
                'time': (row['incident_time'] - incident_df['incident_time'].min()).total_seconds(),
                'lat': row['patient_lat'],
                'lon': row['patient_lon'],
                'severity': row['severity'],
                'specialty': row['specialty_requirement'],
                'duration': row['service_duration_min'] * 60,  # convert to seconds
                'exception': row['exceptional_event'],
                'created': False,
                'assigned': False,
                'ambulance': None,
                'hospital': None,
                'admission_time': None,
                'discharge_time': None,
                'scene_arrival_time': None,
                'dispatch_time': None,
                'tau_scene': None,
                'tau_hospital': None,
                'tau_total': None,
            }
            self.future_incidents.append(incident)
        self.episode_stats['incidents_generated'] = len(self.future_incidents)
        self.episode_stats['high_priority_generated'] = sum(
            1 for inc in self.future_incidents if inc['severity'] == 'HIGH'
        )

        # ---- Initialise ambulances ----
        self.ambulances = []
        for _, row in self.ambulance_df.iterrows():
            amb = {
                'id': row['amb_id'],
                'type': row['type'],      # 'BLS' or 'ALS'
                'home_lat': row['home_lat'],
                'home_lon': row['home_lon'],
                'lat': row['home_lat'],   # start at home
                'lon': row['home_lon'],
                'status': 'IDLE',         # IDLE, EN_ROUTE_TO_SCENE, TRANSPORTING, RETURNING
                'remaining_time': 0.0,    # seconds until next status change
                'dest_lat': None,
                'dest_lon': None,
                'current_incident': None,  # which incident is being handled
                'phase': None             # internal: 'to_scene' / 'to_hospital' / 'home'
            }
            self.ambulances.append(amb)

        # ---- Initialise hospitals ----
        self.hospitals = []
        for _, row in self.hospital_df.iterrows():
            hosp = {
                'name': row['name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'specialties': np.array([int(row.get(s, 0)) for s in self.SPECIALTIES], dtype=np.int32),
                'total_beds': int(row['total_beds']),
                'icu_beds': int(row['icu_beds']),
                'ventilator_beds': int(row['ventilator_beds']),
                'oxygen_beds': int(row['oxygen_beds']),
                # current occupancy (beds occupied)
                'occ_total': 0,
                'occ_icu': 0,
                'occ_vent': 0,
                'occ_oxygen': 0,
            }
            self.hospitals.append(hosp)

        # Clear incident queue and event queue
        self.incident_queue = []
        self.event_queue = []
        self._event_seq = 0
        # Schedule all incident arrivals as events
        for inc in self.future_incidents:
            self._add_event(inc['time'], 'new_incident', inc)

        # Advance to the first decision point
        self._run_until_decision()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one dispatching action.
        Args:
            action: integer index = ambulance_idx * N_HOSP + hospital_idx
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Decode action
        amb_idx = action // self.N_HOSP
        hosp_idx = action % self.N_HOSP
        ambulance = self.ambulances[amb_idx]
        hospital = self.hospitals[hosp_idx]

        # Select the highest priority incident (already guaranteed to exist at decision point)
        incident = self._select_incident()   # oldest HIGH > MEDIUM > LOW
        best_feasible_metrics = self._compute_best_feasible_eta_metrics(incident)

        # Safety check (should always be valid due to action mask)
        if not self._is_action_valid(amb_idx, hosp_idx, incident):
            # This shouldn't happen if mask is used, but return a large penalty
            reward = -1000.0
            self.done = True
            self.episode_stats['invalid_actions'] += 1
            info = {
                'invalid_action': True,
                'action': int(action),
                'ambulance_index': int(amb_idx),
                'hospital_index': int(hosp_idx),
            }
            self.last_transition_info = info
            return self._get_obs(), reward, self.done, False, info

        # --- Mark incident as assigned ---
        incident['assigned'] = True
        incident['ambulance'] = amb_idx
        incident['hospital'] = hosp_idx
        incident['dispatch_time'] = self.current_time
        # Remove from queue
        self.incident_queue.remove(incident)

        # --- Compute travel times (with noise) ---
        dist_to_scene = haversine(ambulance['lat'], ambulance['lon'],
                                  incident['lat'], incident['lon'])
        dist_to_hosp = haversine(incident['lat'], incident['lon'],
                                 hospital['lat'], hospital['lon'])
        # Effective speed: base speed / traffic_multiplier (higher traffic -> slower)
        speed_ms = (self.avg_speed_kmh * 1000 / 3600) / self.traffic_multiplier
        tau_scene = dist_to_scene / speed_ms   # seconds
        tau_hosp  = dist_to_hosp / speed_ms

        # Add log-normal noise (multiplicative)
        noise1 = np.random.lognormal(0, self.travel_noise_std)
        noise2 = np.random.lognormal(0, self.travel_noise_std)
        tau_scene *= noise1
        tau_hosp  *= noise2
        tau_total = tau_scene + tau_hosp

        # --- Compute reward ---
        reward, reward_components = self._compute_reward(
            incident, ambulance, hospital, tau_scene, tau_hosp, tau_total, best_feasible_metrics
        )
        incident['tau_scene'] = tau_scene
        incident['tau_hospital'] = tau_hosp
        incident['tau_total'] = tau_total
        self.episode_stats['dispatches'] += 1
        self.episode_stats['total_reward'] += float(reward)
        self.episode_stats['severity_counts'][incident['severity']] += 1
        for key, value in reward_components.items():
            self.episode_stats['reward_components'][key] += float(value)

        # --- Update ambulance and schedule subsequent events ---
        ambulance['status'] = 'EN_ROUTE_TO_SCENE'
        ambulance['remaining_time'] = tau_scene
        ambulance['dest_lat'] = incident['lat']
        ambulance['dest_lon'] = incident['lon']
        ambulance['current_incident'] = incident
        ambulance['phase'] = 'to_scene'

        # Event 1: arrival at scene (after tau_scene)
        arrival_scene_time = self.current_time + tau_scene
        self._add_event(arrival_scene_time, 'ambulance_arrive_scene',
                        {'ambulance': ambulance, 'incident': incident,
                         'next_tau': tau_hosp, 'hospital': hospital})

        # Event 2: arrival at hospital (after tau_scene + tau_hosp)
        arrival_hosp_time = self.current_time + tau_scene + tau_hosp
        self._add_event(arrival_hosp_time, 'ambulance_arrive_hospital',
                        {'ambulance': ambulance, 'incident': incident,
                         'hospital': hospital})

        # --- Advance time to next decision point ---
        self._run_until_decision()
        info = {
            'invalid_action': False,
            'event_id': int(incident['event_id']),
            'severity': incident['severity'],
            'specialty': incident['specialty'],
            'ambulance_index': int(amb_idx),
            'ambulance_id': ambulance['id'],
            'ambulance_type': ambulance['type'],
            'hospital_index': int(hosp_idx),
            'hospital_name': hospital['name'],
            'tau_scene_sec': float(tau_scene),
            'tau_hospital_sec': float(tau_hosp),
            'tau_total_sec': float(tau_total),
            'chosen_response_eta_min': float(tau_scene / 60.0),
            'chosen_hospital_eta_min': float(tau_total / 60.0),
            'dispatch_time_sec': float(incident['dispatch_time']),
            'queue_size_after_dispatch': int(len(self.incident_queue)),
            'reward': float(reward),
            'reward_components': reward_components,
            'traffic_multiplier': float(self.traffic_multiplier),
            'current_time_sec': float(self.current_time),
        }
        info.update(best_feasible_metrics)
        if info['best_feasible_response_eta_min'] is not None:
            info['chosen_minus_best_response_eta_min'] = (
                info['chosen_response_eta_min'] - info['best_feasible_response_eta_min']
            )
        if info['best_feasible_hospital_eta_min'] is not None:
            info['chosen_minus_best_hospital_eta_min'] = (
                info['chosen_hospital_eta_min'] - info['best_feasible_hospital_eta_min']
            )
        if self.done:
            info['episode_summary'] = self.get_episode_summary()
        self.last_transition_info = info
        return self._get_obs(), reward, self.done, False, info

    def action_mask(self) -> np.ndarray:
        """
        Return a boolean mask of shape (7130,) indicating admissible actions.
        Only actions where ambulance is IDLE, hospital offers required specialty,
        and hospital has an appropriate free bed are True.
        If no incident is present, all False.
        """
        mask = np.zeros(self.N_ACTIONS, dtype=bool)
        incident = self._select_incident() if self.incident_queue else None
        if incident is None:
            return mask
        feasible_actions = self._get_feasible_actions(incident, apply_pruning=True)
        if not feasible_actions:
            return mask
        flat_indices = [amb_idx * self.N_HOSP + hosp_idx for amb_idx, hosp_idx, _, _ in feasible_actions]
        mask[np.asarray(flat_indices, dtype=np.int64)] = True
        return mask

    # ====================== Private methods ======================
    def _select_incident(self) -> Optional[Dict]:
        """Pick oldest incident with highest severity (HIGH > MEDIUM > LOW)."""
        if not self.incident_queue:
            return None
        # Severity order: HIGH=0, MEDIUM=1, LOW=2
        sev_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        # Sort by severity (ascending 0 first) then by creation time (oldest)
        return min(self.incident_queue,
                   key=lambda x: (sev_order[x['severity']], x['time']))

    def _is_action_valid(self, amb_idx, hosp_idx, incident) -> bool:
        amb = self.ambulances[amb_idx]
        if amb['status'] != 'IDLE':
            return False
        if (
            self.enforce_als_for_high_if_available
            and incident['severity'] == 'HIGH'
            and amb['type'] != 'ALS'
            and self._has_feasible_als_option(incident)
        ):
            return False
        hosp = self.hospitals[hosp_idx]
        spec_idx = self.specialty_to_idx[incident['specialty'].upper()]
        if not hosp['specialties'][spec_idx]:
            return False
        bed_type = self.SEV_TO_BEDTYPE[incident['severity']]
        if bed_type == 'icu' and hosp['occ_icu'] >= hosp['icu_beds']:
            return False
        if bed_type == 'ventilator' and hosp['occ_vent'] >= hosp['ventilator_beds']:
            return False
        if bed_type == 'oxygen' and hosp['occ_oxygen'] >= hosp['oxygen_beds']:
            return False
        return True

    def _has_feasible_als_option(self, incident: Dict[str, Any]) -> bool:
        for amb_idx, amb in enumerate(self.ambulances):
            if amb['status'] != 'IDLE' or amb['type'] != 'ALS':
                continue
            for hosp_idx in range(self.N_HOSP):
                hosp = self.hospitals[hosp_idx]
                spec_idx = self.specialty_to_idx[incident['specialty'].upper()]
                if not hosp['specialties'][spec_idx]:
                    continue
                bed_type = self.SEV_TO_BEDTYPE[incident['severity']]
                if bed_type == 'icu' and hosp['occ_icu'] >= hosp['icu_beds']:
                    continue
                if bed_type == 'ventilator' and hosp['occ_vent'] >= hosp['ventilator_beds']:
                    continue
                if bed_type == 'oxygen' and hosp['occ_oxygen'] >= hosp['oxygen_beds']:
                    continue
                return True
        return False

    def _get_feasible_actions(
        self,
        incident: Optional[Dict[str, Any]],
        apply_pruning: bool,
    ) -> List[Tuple[int, int, float, float]]:
        if incident is None:
            return []

        speed_ms = (self.avg_speed_kmh * 1000 / 3600) / self.traffic_multiplier
        severity = incident['severity']
        bed_type = self.SEV_TO_BEDTYPE[severity]
        spec_idx = self.specialty_to_idx[incident['specialty'].upper()]

        idle_amb_candidates: List[Tuple[int, float, str]] = []
        enforce_als_only = False
        if self.enforce_als_for_high_if_available and severity == 'HIGH':
            enforce_als_only = self._has_feasible_als_option(incident)

        for amb_idx, amb in enumerate(self.ambulances):
            if amb['status'] != 'IDLE':
                continue
            if enforce_als_only and amb['type'] != 'ALS':
                continue
            dist_to_scene = haversine(amb['lat'], amb['lon'], incident['lat'], incident['lon'])
            idle_amb_candidates.append((amb_idx, dist_to_scene / speed_ms, amb['type']))

        if not idle_amb_candidates:
            return []

        valid_hospital_candidates: List[Tuple[int, float]] = []
        for hosp_idx, hosp in enumerate(self.hospitals):
            if not hosp['specialties'][spec_idx]:
                continue
            if bed_type == 'icu' and hosp['occ_icu'] >= hosp['icu_beds']:
                continue
            if bed_type == 'ventilator' and hosp['occ_vent'] >= hosp['ventilator_beds']:
                continue
            if bed_type == 'oxygen' and hosp['occ_oxygen'] >= hosp['oxygen_beds']:
                continue
            dist_to_hosp = haversine(incident['lat'], incident['lon'], hosp['lat'], hosp['lon'])
            valid_hospital_candidates.append((hosp_idx, dist_to_hosp / speed_ms))

        if not valid_hospital_candidates:
            return []

        idle_amb_candidates.sort(key=lambda item: item[1])
        valid_hospital_candidates.sort(key=lambda item: item[1])

        if apply_pruning:
            amb_limit = self.top_k_ambulances_by_severity[severity]
            hosp_limit = self.top_k_hospitals_by_severity[severity]
            idle_amb_candidates = idle_amb_candidates[:amb_limit]
            valid_hospital_candidates = valid_hospital_candidates[:hosp_limit]

        feasible_actions: List[Tuple[int, int, float, float]] = []
        for amb_idx, tau_scene, _ in idle_amb_candidates:
            for hosp_idx, tau_hosp in valid_hospital_candidates:
                feasible_actions.append((amb_idx, hosp_idx, tau_scene, tau_scene + tau_hosp))
        return feasible_actions

    def _compute_reward(
        self,
        incident,
        ambulance,
        hospital,
        tau_scene,
        tau_hosp,
        tau_total,
        best_feasible_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Calculate the six-component reward for a dispatch."""
        severity = incident['severity']
        sev_idx = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}[severity]
        amb_type = ambulance['type']   # 'BLS' or 'ALS'
        amb_type_idx = 0 if amb_type == 'BLS' else 1

        # 1. Matching reward
        r_match = self.MATCH_REWARD[amb_type_idx, sev_idx]

        # 2. Travel time penalty
        r_travel = -0.3 * np.log(max(tau_total, 0.1) / 60.0 + 1.0)

        # 3. Relative ETA gap penalty against the best feasible option at decision time.
        best_feasible_hospital_eta_min = None
        if best_feasible_metrics is not None:
            best_feasible_hospital_eta_min = best_feasible_metrics.get('best_feasible_hospital_eta_min')
        chosen_hospital_eta_min = tau_total / 60.0
        eta_gap_min = max(0.0, chosen_hospital_eta_min - float(best_feasible_hospital_eta_min or 0.0))
        if severity == 'HIGH':
            r_eta_gap = -min(0.20 * eta_gap_min, 25.0)
        elif severity == 'MEDIUM':
            r_eta_gap = -min(0.08 * eta_gap_min, 10.0)
        else:
            r_eta_gap = -min(0.04 * eta_gap_min, 6.0)

        # 4. Golden hour compliance / severity-sensitive timeliness bonus
        if severity == 'HIGH':
            if tau_total <= 3600:
                r_gh = 20.0
            elif tau_total <= 5400:
                r_gh = 0.0
            else:
                r_gh = -20.0
        elif severity == 'MEDIUM':
            r_gh = 6.0 if tau_total <= 5400 else 0.0
        else:
            r_gh = 4.0 if tau_total <= 7200 else 0.0

        # 5. Waiting time penalty with saturation to avoid dominating the return.
        waiting_seconds = (self.current_time - incident['time']) + tau_scene
        wait_min = waiting_seconds / 60.0
        if severity == 'HIGH':
            r_wait = -10.0 * np.tanh(wait_min / 20.0)
        elif severity == 'MEDIUM':
            r_wait = -3.0 * np.tanh(wait_min / 30.0)
        else:
            r_wait = 0.0

        # 6. Hospital load balancing
        bed_type = self.SEV_TO_BEDTYPE[severity]
        if bed_type == 'icu':
            occ = hospital['occ_icu']
            cap = hospital['icu_beds']
        elif bed_type == 'ventilator':
            occ = hospital['occ_vent']
            cap = hospital['ventilator_beds']
        else:
            occ = hospital['occ_oxygen']
            cap = hospital['oxygen_beds']
        occupancy_ratio = occ / max(cap, 1)   # avoid division by zero
        r_load = 2.0 * (1.0 - occupancy_ratio)

        # 7. Resource opportunity cost (distance from home base to incident scene)
        home_lat, home_lon = ambulance['home_lat'], ambulance['home_lon']
        dist_from_home = haversine(home_lat, home_lon, incident['lat'], incident['lon'])
        r_opp = -min(dist_from_home / 20000.0, 2.0) if dist_from_home > 5000 else 0.0

        # Weighted sum
        reward_components = {
            'match': float(self.REWARD_WEIGHTS['match'] * r_match),
            'travel': float(self.REWARD_WEIGHTS['travel'] * r_travel),
            'eta_gap': float(self.REWARD_WEIGHTS['eta_gap'] * r_eta_gap),
            'gh': float(self.REWARD_WEIGHTS['gh'] * r_gh),
            'wait': float(self.REWARD_WEIGHTS['wait'] * r_wait),
            'load': float(self.REWARD_WEIGHTS['load'] * r_load),
            'opp': float(self.REWARD_WEIGHTS['opp'] * r_opp),
        }
        total = float(sum(reward_components.values()))

        # Debug print
        if np.isnan(total):
            print("NaN reward detected. Components:")
            print(f"  r_match={r_match}, r_travel={r_travel}, r_gh={r_gh}, r_wait={r_wait}, r_load={r_load}, r_opp={r_opp}")
            print(f"  tau_total={tau_total}, occupancy_ratio={occupancy_ratio if 'occupancy_ratio' in locals() else 'N/A'}")
        return total, reward_components

    def _compute_best_feasible_eta_metrics(self, incident: Optional[Dict]) -> Dict[str, Any]:
        """Compute best feasible response and hospital ETA for the current incident."""
        metrics = {
            'best_feasible_response_eta_min': None,
            'best_feasible_hospital_eta_min': None,
            'best_feasible_ambulance_id': None,
            'best_feasible_hospital_name': None,
            'chosen_minus_best_response_eta_min': None,
            'chosen_minus_best_hospital_eta_min': None,
        }
        if incident is None:
            return metrics

        best_scene_sec = None
        best_total_sec = None
        best_amb_id = None
        best_hosp_name = None

        for amb_idx, hosp_idx, tau_scene, tau_total in self._get_feasible_actions(incident, apply_pruning=False):
            if best_total_sec is None or tau_total < best_total_sec:
                best_total_sec = tau_total
                best_scene_sec = tau_scene
                best_amb_id = self.ambulances[amb_idx]['id']
                best_hosp_name = self.hospitals[hosp_idx]['name']

        if best_scene_sec is not None and best_total_sec is not None:
            metrics['best_feasible_response_eta_min'] = float(best_scene_sec / 60.0)
            metrics['best_feasible_hospital_eta_min'] = float(best_total_sec / 60.0)
            metrics['best_feasible_ambulance_id'] = best_amb_id
            metrics['best_feasible_hospital_name'] = best_hosp_name
        return metrics

    # ---- Event queue handling ----
    def _add_event(self, time_sec, event_type, data):
        self._event_seq += 1
        heapq.heappush(self.event_queue, (time_sec, self._event_seq, event_type, data))

    def _run_until_decision(self):
        """Advance simulation until the next decision epoch or episode end."""
        while True:
            # Check if we should stop the episode
            if self.current_time >= self.EPISODE_HORIZON_SEC:
                self.done = True
                return

            # If no more events, episode done (all incidents processed, all ambulances idle)
            if not self.event_queue:
                self.done = True
                return

            # Peek at next event
            next_time, _, ev_type, ev_data = self.event_queue[0]
            # If we are already at a decision epoch (incident + feasible action), stop
            if self.incident_queue and any(self.action_mask()):
                return

            # Otherwise process the next event
            heapq.heappop(self.event_queue)
            self.current_time = next_time

            # Process event
            if ev_type == 'new_incident':
                incident = ev_data
                incident['created'] = True
                incident['time'] = self.current_time   # set exact arrival time
                self.incident_queue.append(incident)

            elif ev_type == 'ambulance_arrive_scene':
                amb = ev_data['ambulance']
                incident = ev_data['incident']
                next_tau = ev_data['next_tau']
                hospital = ev_data['hospital']
                # Ambulance reaches scene, now transporting
                amb['lat'] = incident['lat']
                amb['lon'] = incident['lon']
                amb['status'] = 'TRANSPORTING'
                amb['remaining_time'] = next_tau
                amb['dest_lat'] = hospital['lat']
                amb['dest_lon'] = hospital['lon']
                amb['phase'] = 'to_hospital'
                incident['scene_arrival_time'] = self.current_time

            elif ev_type == 'ambulance_arrive_hospital':
                amb = ev_data['ambulance']
                incident = ev_data['incident']
                hospital = ev_data['hospital']
                amb['lat'] = hospital['lat']
                amb['lon'] = hospital['lon']
                # Patient admitted to hospital
                admission_time = self.current_time
                incident['admission_time'] = admission_time
                # Occupy appropriate bed
                severity = incident['severity']
                bed_type = self.SEV_TO_BEDTYPE[severity]
                if bed_type == 'icu':
                    hospital['occ_icu'] += 1
                    hospital['occ_total'] += 1
                elif bed_type == 'ventilator':
                    hospital['occ_vent'] += 1
                    hospital['occ_total'] += 1
                elif bed_type == 'oxygen':
                    hospital['occ_oxygen'] += 1
                    hospital['occ_total'] += 1
                # Schedule patient discharge after service duration
                discharge_time = admission_time + incident['duration']
                self._add_event(discharge_time, 'patient_discharge',
                                {'incident': incident, 'hospital': hospital})
                # Ambulance starts returning to home base
                dist_to_home = haversine(amb['dest_lat'], amb['dest_lon'],
                                        amb['home_lat'], amb['home_lon'])
                speed_ms = (self.avg_speed_kmh * 1000 / 3600) / self.traffic_multiplier
                tau_return = dist_to_home / speed_ms * np.random.lognormal(0, self.travel_noise_std)
                amb['status'] = 'RETURNING'
                amb['remaining_time'] = tau_return
                amb['dest_lat'] = amb['home_lat']
                amb['dest_lon'] = amb['home_lon']
                amb['phase'] = 'home'
                self._add_event(self.current_time + tau_return, 'ambulance_arrive_home',
                                {'ambulance': amb})

            elif ev_type == 'patient_discharge':
                incident = ev_data['incident']
                hospital = ev_data['hospital']
                incident['discharge_time'] = self.current_time
                self.episode_stats['completed_patients'] += 1
                severity = incident['severity']
                bed_type = self.SEV_TO_BEDTYPE[severity]
                if bed_type == 'icu':
                    hospital['occ_icu'] = max(0, hospital['occ_icu'] - 1)
                    hospital['occ_total'] = max(0, hospital['occ_total'] - 1)
                elif bed_type == 'ventilator':
                    hospital['occ_vent'] = max(0, hospital['occ_vent'] - 1)
                    hospital['occ_total'] = max(0, hospital['occ_total'] - 1)
                elif bed_type == 'oxygen':
                    hospital['occ_oxygen'] = max(0, hospital['occ_oxygen'] - 1)
                    hospital['occ_total'] = max(0, hospital['occ_total'] - 1)

            elif ev_type == 'ambulance_arrive_home':
                amb = ev_data['ambulance']
                amb['lat'] = amb['home_lat']
                amb['lon'] = amb['home_lon']
                amb['status'] = 'IDLE'
                amb['remaining_time'] = 0.0
                amb['dest_lat'] = None
                amb['dest_lon'] = None
                amb['current_incident'] = None
                amb['phase'] = None

            # After event processing, loop to check for decision point again

    # ---- State representation ----
    def _get_obs(self) -> np.ndarray:
        """Build the flattened normalised state vector."""
        # 1. Incident queue (fixed size 30, pad with zeros)
        max_inc = 30
        inc_vec = []
        # Take sorted by priority rule? We'll just take the whole list, padding.
        # To keep consistent, we can sort by (severity, creation_time) descending priority.
        sev_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_inc = sorted(self.incident_queue,
                            key=lambda x: (sev_order[x['severity']], x['time']))
        for i in range(max_inc):
            if i < len(sorted_inc):
                inc = sorted_inc[i]
                lat_norm = (inc['lat'] - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
                lon_norm = (inc['lon'] - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])
                sev_onehot = np.zeros(3)
                sev_onehot[sev_order[inc['severity']]] = 1.0
                spec_onehot = np.zeros(self.N_SPEC)
                spec_onehot[self.specialty_to_idx[inc['specialty'].upper()]] = 1.0
                wait_seconds = max(self.current_time - inc.get('time', self.current_time), 0.0)
                wait_norm = wait_seconds / self.max_wait_time
                exc_onehot = np.zeros(self.N_EXCEPT)
                exc_idx = self.EXCEPTIONS.index(inc['exception']) if inc['exception'] in self.EXCEPTIONS else 0
                exc_onehot[exc_idx] = 1.0
                features = np.concatenate([[lat_norm, lon_norm], sev_onehot, spec_onehot,
                                           [wait_norm], exc_onehot])
                inc_vec.append(features)
            else:
                inc_vec.append(np.zeros(2+3+self.N_SPEC+1+self.N_EXCEPT))
        inc_vec = np.concatenate(inc_vec)

        # 2. Ambulances
        amb_vec = []
        status_list = ['IDLE', 'EN_ROUTE_TO_SCENE', 'TRANSPORTING', 'RETURNING']
        for amb in self.ambulances:
            lat_n = (amb['lat'] - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
            lon_n = (amb['lon'] - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])
            type_onehot = np.array([1, 0]) if amb['type'] == 'BLS' else np.array([0, 1])
            status_onehot = np.zeros(4)
            if amb['status'] in status_list:
                status_onehot[self.status_to_idx[amb['status']]] = 1.0
            rem_time_n = amb['remaining_time'] / self.max_travel_time
            amb_vec.append(np.concatenate([[lat_n, lon_n], type_onehot, status_onehot, [rem_time_n]]))
        amb_vec = np.concatenate(amb_vec)

        # 3. Hospitals
        hosp_vec = []
        for hosp in self.hospitals:
            lat_n = (hosp['lat'] - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
            lon_n = (hosp['lon'] - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])
            spec_vec = hosp['specialties'].astype(np.float32)
            occ_total_r = hosp['occ_total'] / max(hosp['total_beds'], 1)
            occ_icu_r = hosp['occ_icu'] / max(hosp['icu_beds'], 1)
            occ_vent_r = hosp['occ_vent'] / max(hosp['ventilator_beds'], 1)
            occ_oxygen_r = hosp['occ_oxygen'] / max(hosp['oxygen_beds'], 1)
            hosp_vec.append(np.concatenate([[lat_n, lon_n], spec_vec,
                                            [occ_total_r, occ_icu_r, occ_vent_r, occ_oxygen_r]]))
        hosp_vec = np.concatenate(hosp_vec)

        # 4. Context
        time_norm = self.current_time / self.max_sim_time
        traffic_norm = (self.traffic_multiplier - self.traffic_bounds[0]) / (self.traffic_bounds[1] - self.traffic_bounds[0])
        ctx = np.array([time_norm, traffic_norm])

        return np.concatenate([inc_vec, amb_vec, hosp_vec, ctx]).astype(np.float32)

    def render(self, mode='human'):
        """Simple text render."""
        if mode == 'human':
            print(f"Time: {self.current_time/3600:.2f}h | Queue: {len(self.incident_queue)} | "
                  f"Idle Amb: {sum(1 for a in self.ambulances if a['status']=='IDLE')}")

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a compact summary of the current episode for analysis."""
        summary = {
            'dispatches': int(self.episode_stats['dispatches']),
            'invalid_actions': int(self.episode_stats['invalid_actions']),
            'total_reward': float(self.episode_stats['total_reward']),
            'incidents_generated': int(self.episode_stats['incidents_generated']),
            'high_priority_generated': int(self.episode_stats['high_priority_generated']),
            'completed_patients': int(self.episode_stats['completed_patients']),
            'severity_counts': dict(self.episode_stats['severity_counts']),
            'reward_components': dict(self.episode_stats['reward_components']),
            'episode_time_sec': float(self.current_time),
            'traffic_multiplier': float(self.traffic_multiplier),
            'remaining_queue': int(len(self.incident_queue)),
            'idle_ambulances_end': int(sum(1 for a in self.ambulances if a['status'] == 'IDLE')),
        }
        if summary['dispatches'] > 0:
            summary['mean_reward_per_dispatch'] = summary['total_reward'] / summary['dispatches']
        else:
            summary['mean_reward_per_dispatch'] = 0.0
        return summary

# ------------------------------------------------------------
# 4.  Synthetic incident generator adapter (example)
# ------------------------------------------------------------
# Assuming you have your Stage 3 generator defined as a function that returns a DataFrame.
# Wrap it into a callable that the environment can use.
def make_incident_generator(target_district="NASHIK", seed=42, prefer_external=False):
    """
    Return an incident generator callable for the environment.
    Falls back to a lightweight built-in generator when pat_gen or its data
    dependencies are unavailable.
    """
    rng = np.random.default_rng(seed)
    severity_probs = np.array([0.2, 0.35, 0.45])  # HIGH, MEDIUM, LOW
    severity_labels = np.array(["HIGH", "MEDIUM", "LOW"])
    specialty_by_severity = {
        "HIGH": np.array(["CRITICAL", "CARDIOLOGY", "NEUROSURGERY", "POLYTRAUMA", "BURNS"]),
        "MEDIUM": np.array(["MEDICINE", "GENERAL", "ORTHOPEDICS", "PULMONOLOGY", "NEUROLOGY"]),
        "LOW": np.array(["GENERAL", "MEDICINE", "PEDIATRICS", "GYNAECOLOGY", "UROLOGY"]),
    }
    exception_labels = np.array(["NONE", "HIGHWAY_CRASH_BURST", "FESTIVAL_CROWDING", "RAIN_DISRUPTION"])
    exception_probs = np.array([0.84, 0.05, 0.06, 0.05])

    external_generator = None
    if prefer_external:
        try:
            from pat_gen import generate_synthetic_incidents_poisson as external_generator
        except Exception:
            external_generator = None

    def generator(sim_days=7):
        if external_generator is not None:
            return external_generator(sim_days=sim_days)

        start_date = pd.Timestamp.today().normalize()
        rows = []
        event_id = 1
        daily_events = max(20, int(32 * sim_days))

        for day_offset in range(sim_days):
            day_start = start_date + pd.Timedelta(days=day_offset)
            n_events = rng.poisson(daily_events)
            if n_events == 0:
                continue

            offsets = np.sort(rng.integers(0, 24 * 3600, size=n_events))
            for sec in offsets:
                incident_time = day_start + pd.Timedelta(seconds=int(sec))
                severity = rng.choice(severity_labels, p=severity_probs)
                specialty = rng.choice(specialty_by_severity[severity])
                service_duration_min = float(np.clip(rng.normal(loc=90, scale=30), 20, 240))
                rows.append({
                    "event_id": event_id,
                    "incident_time": incident_time,
                    "patient_lat": float(rng.uniform(19.45, 20.85)),
                    "patient_lon": float(rng.uniform(73.15, 74.75)),
                    "severity": severity,
                    "specialty_requirement": specialty,
                    "service_duration_min": service_duration_min,
                    "exceptional_event": str(rng.choice(exception_labels, p=exception_probs)),
                })
                event_id += 1

        return pd.DataFrame(rows)
    return generator


def get_training_device(preferred: str = "auto") -> str:
    """Device string suitable for torch/SB3 training."""
    return resolve_torch_device(preferred)

# ------------------------------------------------------------
# 5.  Usage with Maskable PPO (Stable-Baselines3 + SB3-Contrib)
# ------------------------------------------------------------


if __name__ == "__main__":
    # Load data
    hospital_df = load_hospitals("HOSPITALS_DATA_FINAL.xlsx")
    ambulance_df = load_ambulances("final_ambulance_deployment.csv")

    # Instantiate environment
    env = EMSEnv(
        hospital_df=hospital_df,
        ambulance_df=ambulance_df,
        incident_generator_fn=make_incident_generator(seed=42, prefer_external=False),
        avg_speed_kmh=40.0,
        travel_noise_std=0.1,
        seed=42,
        device="auto",
    )

    print(f"Environment acceleration device: {env.device}")
    if torch is not None:
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # Example of using MaskablePPO (requires sb3-contrib)
    # from sb3_contrib import MaskablePPO
    # model = MaskablePPO("MlpPolicy", env, verbose=1, device=get_training_device("auto"))
    # model.learn(total_timesteps=500_000)
    # model.save("ems_dispatch_maskable_ppo")

    # Quick test
    obs, _ = env.reset()
    print("Initial observation shape:", obs.shape)
    mask = env.action_mask()
    print("Valid actions:", mask.sum(), "out of", env.N_ACTIONS)
    # Take a random valid action
    valid_acts = np.where(mask)[0]
    if len(valid_acts) > 0:
        action = np.random.choice(valid_acts)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}, Done: {done}")
