"""G1 29-DoF + BrainCo Revo2 right hand (fixed grasp) + epee, built by editing a G1 MjSpec.

Everything is merged INTO `right_wrist_yaw_link` (no new bodies, no new joints): the stock G1 body list, qpos (7 + 29) and
qvel stay unchanged, so motion files, body indices and mjlab / SuperMimic code written for the stock G1 keep working.
What changes: the inertia of that link (bare link + hand + epee, 1.24 kg instead of the 0.25 kg rubber hand), its visual
geoms, its collision geoms (fist capsule replaces `right_hand_collision`, plus guard cylinder and blade capsule, all named
`*_collision` so mjlab's collision regexes pick them up), and two sites (`epee_tip`, `epee_guard_apex`).

Usage with mjlab (e.g. inside SuperMimic-Reproduced):
    from fencingrobot.robot_spec import get_g1_epee_robot_cfg
    robot_cfg = get_g1_epee_robot_cfg()          # = mjlab get_g1_robot_cfg() with spec_fn replaced
Standalone:
    spec = mujoco.MjSpec.from_file(g1_xml); apply_grasp(spec); model = spec.compile()
Actuators / effort limits are NOT touched: mjlab's G1 config already uses the Unitree motor limits (wrist pitch / yaw 5 N m),
which is what the project decided to train with (2026-11-10).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC_JSON = REPO / "robot" / "grasp" / "grasp_spec.json"
MESH_DIR = REPO / "robot" / "grasp" / "meshes"


def load_grasp_spec(path: Path = SPEC_JSON) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_grasp(spec: mujoco.MjSpec, gs: dict | None = None, mesh_dir: Path = MESH_DIR, visual: bool = True) -> mujoco.MjSpec:
    """Edit `spec` in place (and return it)."""
    gs = gs or load_grasp_spec()
    body = spec.body(gs["link"])
    if body is None:
        raise KeyError(f"body {gs['link']} not in spec")
    # 1. remove the stock hand geoms (rubber hand visual + its collision capsule)
    for g in list(body.geoms):
        tag = (g.name or "") + "|" + (getattr(g, "meshname", "") or "")
        if any(k in tag for k in gs["remove_geoms_matching"]):
            spec.delete(g)
    # 2. inertia of the merged link
    I = gs["inertial"]
    body.explicitinertial = True
    body.mass = float(I["mass"])
    body.ipos = np.asarray(I["pos"], float)
    ixx, iyy, izz, ixy, ixz, iyz = I["fullinertia"]
    w, V = np.linalg.eigh(np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], float))
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = np.zeros(4); mujoco.mju_mat2Quat(q, V.reshape(-1))
    body.inertia, body.iquat = w, q                  # principal moments + frame (avoids fullinertia vs diaginertia clash)
    # 3. visual meshes (no contact, no mass)
    if visual:
        for v in gs["visual"]:
            mname = "fr_" + Path(v["mesh"]).stem
            if spec.mesh(mname) is None:
                spec.add_mesh(name=mname, file=str((mesh_dir / v["mesh"]).resolve()))
            g = body.add_geom(name=v["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                              pos=np.asarray(v["pos"], float), quat=np.asarray(v["quat_wxyz"], float))
            g.contype, g.conaffinity, g.group, g.density, g.mass = 0, 0, 2, 0.0, 0.0
            g.rgba = np.asarray(v.get("rgba", [0.8, 0.8, 0.8, 1]), float)
    # 4. collision primitives (massless: the inertia above already contains everything)
    for c in gs["collision"]:
        kw = dict(name=c["name"], size=np.r_[np.asarray(c["size"], float), np.zeros(3 - len(c["size"]))])
        if c["type"] == "capsule":
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.asarray(c["fromto"], float), **kw)
        elif c["type"] == "cylinder":
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=np.asarray(c["pos"], float),
                              quat=np.asarray(c["quat_wxyz"], float), **kw)
        else:
            raise ValueError(c["type"])
        g.density, g.mass, g.group = 0.0, 0.0, 3
    # 5. sites
    for s in gs["sites"]:
        body.add_site(name=s["name"], pos=np.asarray(s["pos"], float), size=np.array([0.006, 0, 0]))
    return spec


# ------------------------------------------------------------------------------------------- soft blade contact
# A real epee blade bends; the model's blade is rigid (no extra bodies / joints: qpos stays 7 + 29). To get contact FORCES
# of the right size, the blade capsule is cut into segments and each segment gets the contact stiffness a cantilever
# clamped at the guard has at that distance x from the guard: k(x) = 3 EI / x^3. The bend shape / whip is NOT modelled
# (the blade sinks into what it touches instead of bending around it).
# EI: FIE material rules m.16.5 - blade clamped 70 cm from the tip, 200 g hung 3 cm from the tip, tip deflection 4.5 to
# 7 cm -> uniform-beam EI = F a^2 (3 L - a) / (6 delta) = 0.2099 / delta = 3.0 to 4.66 N m^2; middle of the range
# (delta 5.75 cm). A real blade tapers, so it is stiffer than this near the guard and softer near the tip.
BLADE_EI = 0.2099 / 0.0575            # N m^2
BLADE_SEGMENTS = 4
BLADE_IMPEDANCE = 0.5                 # constant solimp d (no ramp with depth: penetrations here are centimetres)
# MuJoCo's direct contact stiffness solref = (-K, -B) is in acceleration units: with constant impedance d, the contact
# that statically balances a force F at penetration r satisfies  F = K d / ((1 - d) W) r,  where W is the solver's
# diagonal approximation of the inverse inertia at the contact - a MODEL CONSTANT, not pose dependent. W is measured,
# not derived: tools/check_blade_contact.py pushes each segment into the floor and reads F / r (CPU MuJoCo and
# mujoco_warp agree to 4 digits); in the task model W = 0.25 / kg (half the wrist link's body_invweight0 of 0.4999).
# Against the robot's own bodies the other body's inverse weight adds, so self-contacts come out softer than k(x);
# that case is not measured.
BLADE_INVWEIGHT = 0.25  # 1 / kg


def blade_segments(gs: dict, n: int = BLADE_SEGMENTS) -> list[dict]:
    """Segments of the blade capsule, guard to tip, in the link frame, with the target stiffness at their centre."""
    c = next(c for c in gs["collision"] if c["name"] == "epee_blade_collision")
    p0, p1 = np.asarray(c["fromto"][:3], float), np.asarray(c["fromto"][3:], float)
    L = float(np.linalg.norm(p1 - p0))
    out = []
    for i in range(n):
        a, b = p0 + (p1 - p0) * i / n, p0 + (p1 - p0) * (i + 1) / n
        x = L * (i + 0.5) / n
        out.append({"name": f"epee_blade{i}_collision", "p0": a, "p1": b, "radius": float(c["size"][0]), "x_mid": x,
                    "k": 3.0 * BLADE_EI / x ** 3})
    return out


def blade_contact_params(n: int = BLADE_SEGMENTS) -> dict[str, tuple[tuple[float, float], tuple[float, ...]]]:
    """geom name -> (solref, solimp) realising k(x) for a static push against the floor."""
    assert BLADE_INVWEIGHT > 0, "run tools/check_blade_contact.py and fill BLADE_INVWEIGHT"
    d = BLADE_IMPEDANCE
    out = {}
    for s in blade_segments(load_grasp_spec(), n):
        K = s["k"] * (1.0 - d) / d * BLADE_INVWEIGHT
        out[s["name"]] = ((-K, -2.0 * np.sqrt(K)), (d, d, 0.001, 0.5, 2.0))   # damping: critical for the reference dynamics
    return out


def soften_blade(spec: mujoco.MjSpec, n: int = BLADE_SEGMENTS) -> mujoco.MjSpec:
    """Replace `epee_blade_collision` by n capsule segments `epee_blade{i}_collision` (same line, same radius, massless).
    Their solref / solimp / priority are set by the collision cfg (mjlab applies it after the spec function)."""
    gs = load_grasp_spec()
    body = spec.body(gs["link"])
    for g in list(body.geoms):
        if g.name == "epee_blade_collision":
            spec.delete(g)
    for s in blade_segments(gs, n):
        g = body.add_geom(name=s["name"], type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.r_[s["p0"], s["p1"]],
                          size=np.array([s["radius"], 0, 0]))
        g.density, g.mass, g.group = 0.0, 0.0, 3
    return spec


def get_g1_epee_spec() -> mujoco.MjSpec:
    """mjlab's G1 spec with the grasp applied (requires mjlab)."""
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as C
    return apply_grasp(C.get_spec())


def get_g1_epee_soft_spec() -> mujoco.MjSpec:
    return soften_blade(get_g1_epee_spec())


def get_g1_epee_robot_cfg(soft_blade: bool = False):
    """mjlab EntityCfg identical to get_g1_robot_cfg() except for the robot spec (requires mjlab).

    soft_blade: blade capsule cut into segments with bending-equivalent contact stiffness (see above). The blade
    segments get priority 2 (feet have 1, everything else 0) so that THEIR solref / solimp decide the contact instead
    of being mixed with the other geom's."""
    from dataclasses import replace

    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as C
    cfg = C.get_g1_robot_cfg()
    if not soft_blade:
        cfg.spec_fn = get_g1_epee_spec
        return cfg
    cfg.spec_fn = get_g1_epee_soft_spec
    assert cfg.collisions == (C.FULL_COLLISION,), cfg.collisions
    par = blade_contact_params()
    base = C.FULL_COLLISION
    cfg.collisions = (replace(
        base,
        priority={**base.priority, **{f"^{n}$": 2 for n in par}},
        solref={f"^{n}$": tuple(float(x) for x in v[0]) for n, v in par.items()},
        solimp={f"^{n}$": v[1] for n, v in par.items()},
    ),)
    return cfg
