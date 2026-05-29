"""
PHINO MuJoCo physics engine.

Replaces the old hand-written `AnalyticalPhysicsEngine` (point-mass Newtonian
integrator) with real MuJoCo simulation of MolmoSpaces object assets.

For each asset we compose a scene = the asset's MJCF body (with its meshes /
collision geoms) + a ground plane + a light, override the body's
mass / friction / restitution from the predicted `PhysicsConfig`, settle the
object onto the plane, then apply each of the 5 diagnostic actions as an
external wrench (`data.xfrc_applied`) on the free body and record a 13-D
trajectory `[pos(3), quat(4), linvel(3), angvel(3)]` per frame at 20 Hz.

The trajectory layout matches the old engine exactly so the downstream
similarity / dataset / training code is unchanged.

MuJoCo has no direct restitution coefficient, so restitution is approximated by
the contact `solref` damping ratio (lower damping ratio -> bouncier).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================================
# Diagnostic actions (ported from the analytical engine, retuned for MuJoCo)
# ==========================================================================
@dataclass
class DiagnosticAction:
    """A short scripted external wrench applied to the free body.

    `force` / `torque` are *fixed* magnitudes (Newtons / Newton-metres) applied
    in the world frame, so lighter objects accelerate more (mass-sensitive),
    high-friction objects decelerate faster (friction-sensitive), and bouncy
    objects rebound (restitution-sensitive).
    """

    name: str
    action_type: str
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    force: float = 8.0
    torque: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


DIAGNOSTIC_ACTIONS: List[DiagnosticAction] = [
    DiagnosticAction("push_x", "push", _unit([1.0, 0.0, 0.0]), force=8.0, torque=0.2),
    DiagnosticAction("push_y", "push", _unit([0.0, 1.0, 0.0]), force=8.0, torque=0.2),
    DiagnosticAction("grasp_lift_release", "grasp_lift", _unit([0.0, 0.0, 1.0]), force=6.0),
    DiagnosticAction("lateral_flick", "flick", _unit([1.0, 0.5, 0.0]), force=25.0, torque=5.0),
    DiagnosticAction("slow_press_down", "press", _unit([0.0, 0.0, -1.0]), force=8.0),
]


# ==========================================================================
# Engine
# ==========================================================================
class MuJoCoPhysicsEngine:
    """Simulate MolmoSpaces assets under diagnostic actions in MuJoCo."""

    def __init__(
        self,
        frame_dt: float = 0.05,        # 20 Hz recording (matches old engine)
        timesteps: int = 50,           # frames per action
        sim_timestep: float = 0.005,   # MuJoCo integrator step
        settle_steps: int = 200,       # substeps to settle onto the plane
        spawn_height: float = 0.12,    # initial drop height (m)
        pos_clip: float = 3.0,         # clip displacement (m) — object left the workspace
        lin_vel_clip: float = 15.0,    # clip linear velocity (m/s)
        ang_vel_clip: float = 50.0,    # clip angular velocity (rad/s)
    ):
        self.frame_dt = frame_dt
        self.timesteps = timesteps
        self.sim_timestep = sim_timestep
        self.settle_steps = settle_steps
        self.spawn_height = spawn_height
        self.pos_clip = pos_clip
        self.lin_vel_clip = lin_vel_clip
        self.ang_vel_clip = ang_vel_clip
        self.substeps = max(1, round(frame_dt / sim_timestep))

    # ----- scene construction -------------------------------------------
    def build_scene(self, mjcf_path: str):
        """Compose asset + ground plane + light into a compiled MjModel.

        Returns (model, data, info) where info has the free-joint body id and
        the qpos/qvel addresses plus the added plane geom id.
        """
        import mujoco

        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        spec.option.gravity = [0.0, 0.0, -9.81]
        spec.option.timestep = self.sim_timestep

        wb = spec.worldbody
        wb.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
        plane = wb.add_geom()
        plane.type = mujoco.mjtGeom.mjGEOM_PLANE
        plane.size = [5.0, 5.0, 0.1]
        plane.pos = [0.0, 0.0, 0.0]
        plane.rgba = [0.6, 0.6, 0.6, 1.0]

        model = spec.compile()
        data = mujoco.MjData(model)

        # locate the (first) free joint = the object's root body
        free = [j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise RuntimeError(f"No free joint found in asset {mjcf_path}")
        jid = free[0]
        info = {
            "body_id": int(model.jnt_bodyid[jid]),
            "qadr": int(model.jnt_qposadr[jid]),
            "vadr": int(model.jnt_dofadr[jid]),
            "plane_gid": model.ngeom - 1,  # the plane we appended last
        }
        return model, data, info

    def apply_physics(self, model, info: dict, physics) -> None:
        """Override mass / friction / restitution on a compiled model in place."""
        # --- mass: scale all (non-world) body masses+inertias to hit target ---
        bid = info["body_id"]
        cur_mass = float(model.body_subtreemass[bid])
        if cur_mass > 1e-9 and physics.mass > 0:
            ratio = physics.mass / cur_mass
            model.body_mass[1:] *= ratio
            model.body_inertia[1:] *= ratio

        # --- friction + restitution on collision geoms (+ the plane) ---
        # restitution -> contact solref damping ratio (lower => bouncier)
        dampratio = float(np.clip(1.0 - physics.restitution, 0.08, 1.0))
        timeconst = max(2.0 * self.sim_timestep, 0.01)
        for g in range(model.ngeom):
            is_collision = (model.geom_contype[g] != 0
                            or model.geom_conaffinity[g] != 0)
            if not is_collision:
                continue
            model.geom_friction[g, 0] = float(physics.static_friction)
            model.geom_friction[g, 1] = max(0.005, 0.1 * physics.static_friction)
            model.geom_solref[g, 0] = timeconst
            model.geom_solref[g, 1] = dampratio

    def _settle(self, model, data, info) -> np.ndarray:
        """Drop the object and let it come to rest; return settled qpos/qvel snapshot."""
        import mujoco

        mujoco.mj_resetData(model, data)
        qadr = info["qadr"]
        data.qpos[qadr + 2] = self.spawn_height
        for _ in range(self.settle_steps):
            mujoco.mj_step(model, data)
        snap = (data.qpos.copy(), data.qvel.copy())
        return snap

    # ----- action wrench schedule ---------------------------------------
    def _wrench(self, action: DiagnosticAction, frame: int, T: int,
                obj_mass: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (force(3), torque(3)) in world frame for a given frame."""
        f = np.zeros(3)
        tau = np.zeros(3)
        half = T // 2
        if action.action_type == "push":
            if frame < half:
                f = action.direction * action.force
                tau[2] = action.torque
        elif action.action_type == "grasp_lift":
            # lift with a fixed upward force (heavier => rises less), then release
            if frame < int(0.3 * T):
                f = action.direction * action.force
            elif frame < int(0.6 * T):
                f = action.direction * (obj_mass * 9.81)  # hold against gravity
            # else: released -> falls (tests restitution on impact)
        elif action.action_type == "flick":
            if frame < 4:
                f = action.direction * action.force
                tau[2] = action.torque
        elif action.action_type == "press":
            if frame < int(0.6 * T):
                f = action.direction * action.force
        return f, tau

    # ----- public API ----------------------------------------------------
    def simulate_actions(
        self,
        mjcf_path: str,
        physics,
        actions: Optional[List[DiagnosticAction]] = None,
    ) -> np.ndarray:
        """Run all diagnostic actions for one (asset, physics); return concatenated
        trajectory of shape (len(actions) * timesteps, 13).
        """
        import mujoco

        actions = actions or DIAGNOSTIC_ACTIONS
        model, data, info = self.build_scene(mjcf_path)
        self.apply_physics(model, info, physics)
        qpos0, qvel0 = self._settle(model, data, info)

        bid, qadr, vadr = info["body_id"], info["qadr"], info["vadr"]
        obj_mass = float(physics.mass)
        start_pos = qpos0[qadr:qadr + 3].copy()

        all_traj = []
        for action in actions:
            data.qpos[:] = qpos0
            data.qvel[:] = qvel0
            mujoco.mj_forward(model, data)
            traj = np.zeros((self.timesteps, 13), dtype=np.float64)
            for t in range(self.timesteps):
                f, tau = self._wrench(action, t, self.timesteps, obj_mass)
                for _ in range(self.substeps):
                    data.xfrc_applied[:] = 0.0
                    data.xfrc_applied[bid, :3] = f
                    data.xfrc_applied[bid, 3:] = tau
                    mujoco.mj_step(model, data)
                # record: position relative to settled start (comparable across assets).
                # Clip to a workspace bound so a near-massless object accelerated by a
                # fixed force doesn't fly to ~1km and blow up the trajectory MSE
                # (which would saturate exp(-MSE/sigma) to 0 and kill the label).
                traj[t, :3] = np.clip(data.qpos[qadr:qadr + 3] - start_pos,
                                      -self.pos_clip, self.pos_clip)
                traj[t, 3:7] = data.qpos[qadr + 3:qadr + 7]
                traj[t, 7:10] = np.clip(data.qvel[vadr:vadr + 3],
                                        -self.lin_vel_clip, self.lin_vel_clip)
                traj[t, 10:13] = np.clip(data.qvel[vadr + 3:vadr + 6],
                                         -self.ang_vel_clip, self.ang_vel_clip)
            all_traj.append(traj)
        return np.concatenate(all_traj, axis=0)

    def execute_diagnostic_action(
        self, mjcf_path: str, action: DiagnosticAction, physics, timesteps: int = 50,
    ) -> np.ndarray:
        """Single-action variant (kept for parity with the old engine API)."""
        return self.simulate_actions(mjcf_path, physics, actions=[action])


# ==========================================================================
# Dynamics similarity (full-trajectory MSE -> soft contrastive label)
# ==========================================================================
def compute_dynamics_similarity_mse(traj1: np.ndarray, traj2: np.ndarray) -> float:
    mse = np.mean((traj1 - traj2) ** 2)
    return float(np.exp(-mse / 5.0))


# alias kept for backwards-compatible imports
compute_dynamics_similarity_l2 = compute_dynamics_similarity_mse

SIMILARITY_METRICS = {
    "mse": compute_dynamics_similarity_mse,
    "l2": compute_dynamics_similarity_mse,
}


class TrajectoryNormalizer:
    """Standardize trajectories per (action, feature) and auto-calibrate sigma.

    Different diagnostic actions live at very different scales (a flick flings the
    object metres, a press barely moves it), and a fixed-force flick on a light
    object can still produce a much larger displacement than on a heavy one. A
    single raw concatenated MSE is therefore dominated by whichever action/feature
    has the largest magnitude, and `exp(-MSE/sigma)` saturates to 0 for many pairs
    (information loss).

    Fix: z-score each (action, feature) using dataset statistics so every action
    contributes comparably, then pick sigma from the *distribution* of pairwise
    distances (so the resulting similarities are well spread, median ~0.5) rather
    than a hand-tuned constant.
    """

    def __init__(self, n_actions: int = 5, timesteps: int = 50):
        self.n_actions = n_actions
        self.timesteps = timesteps
        self.mean = None   # (n_actions, 13)
        self.std = None    # (n_actions, 13)
        self.sigma = 1.0

    def _reshape(self, fp: np.ndarray) -> np.ndarray:
        return fp.reshape(self.n_actions, self.timesteps, 13)

    def fit(self, fingerprints, rng=None, n_calib_pairs: int = 20000,
            target_median_sim: float = 0.5):
        A = np.stack(fingerprints).reshape(
            len(fingerprints), self.n_actions, self.timesteps, 13)
        self.mean = A.mean(axis=(0, 2))               # (n_actions, 13)
        self.std = A.std(axis=(0, 2)) + 1e-6
        # calibrate sigma so the median pairwise similarity ~= target_median_sim
        rng = rng or np.random.default_rng(0)
        n = len(fingerprints)
        Z = (A - self.mean[None, :, None, :]) / self.std[None, :, None, :]
        Z = Z.reshape(n, -1)
        m = min(n_calib_pairs, n * (n - 1) // 2) if n > 1 else 0
        dists = []
        for _ in range(m):
            i, j = rng.choice(n, size=2, replace=False)
            dists.append(np.mean((Z[i] - Z[j]) ** 2))
        med = float(np.median(dists)) if dists else 1.0
        self.sigma = max(med / np.log(1.0 / target_median_sim), 1e-6)
        return self

    def transform(self, fp: np.ndarray) -> np.ndarray:
        f = self._reshape(fp)
        return ((f - self.mean[:, None, :]) / self.std[:, None, :]).reshape(-1)

    def similarity(self, fp_i: np.ndarray, fp_j: np.ndarray) -> float:
        mse = np.mean((self.transform(fp_i) - self.transform(fp_j)) ** 2)
        return float(np.exp(-mse / self.sigma))

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std,
                 sigma=self.sigma, n_actions=self.n_actions, timesteps=self.timesteps)

    @staticmethod
    def load(path: str) -> "TrajectoryNormalizer":
        d = np.load(path)
        tn = TrajectoryNormalizer(int(d["n_actions"]), int(d["timesteps"]))
        tn.mean, tn.std, tn.sigma = d["mean"], d["std"], float(d["sigma"])
        return tn
