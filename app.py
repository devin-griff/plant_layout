# =============================================================================
# Plant Layout Optimizer — a Streamlit tutorial app.
#
# Process plant layout problem solved via Pyomo GDP. Place rectangular
# blocks in 2D space to minimize:
#   - plant bounding-box dimensions  (l_f + w_f), plus
#   - cost-weighted Manhattan pipe distances between blocks
#                                       (Σ c_ij · (t_ij + s_ij))
#
# Library roadmap:
#   - streamlit  — UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — algebraic modeling, including the `pyomo.gdp` submodule
#                  for native Disjunction blocks.
#   - Gurobi     — MIP solver, called via Pyomo's native appsi Gurobi
#                  interface. Ships as a pip wheel (`gurobipy`); needs a
#                  Gurobi license (WLS in production via Fly secrets, or a
#                  local license file pointed to by GRB_LICENSE_FILE).
#   - pandas     — DataFrames for the editable block-dimensions and
#                  cost-matrix tables.
#   - altair     — interactive layout figure (rectangles + pipe lines +
#                  hover tooltips).
#
# Model structure:
#   The non-overlap structure is naturally a 4-way disjunction per block
#   pair (i is left / right / above / below j); rotation (when enabled) is
#   a 2-way disjunction per block (default vs. 90° rotated). Both are
#   written as `pyomo.gdp.Disjunction` blocks and reformulated to a MILP
#   via the Big-M GDP transformation, then solved with Gurobi.
#   Pipe distances are rectilinear center-to-center (the literature
#   convention), computed by always-on dx/dy constraints kept OUT of the
#   disjunction so the objective never depends on which spatial relation is
#   chosen — this avoids the costly continuous degeneracy that coupling
#   distance into the disjuncts would create.
#
# Symmetry breaking:
#   `sym=1` is hardcoded. Pinning block 1 (the rack) to be "left of" block 2
#   kills the left/right mirror and speeds up the solve. Only the horizontal
#   mirror is a symmetry: the top/bottom mirror is broken by each object's fixed
#   north/south tie-in, so there is no vertical cut. See `sym_1` in `build_model`.
#
# Time limit + incumbent handling:
#   At n=10 the solve can blow past 10 s. We set a wall-clock time limit
#   and use Pyomo's `load_solutions=False` path to optionally load the best
#   feasible solution found before the cutoff. The Layout tab annotates
#   "Optimal" vs "Incumbent (suboptimal)" accordingly.
#
# File roadmap (matching section banners below):
#   1. Page config + CSS + home-logo.
#   2. Constants and defaults.
#   3. State helpers — object-list model, add/delete, reset, randomize.
#   4. Solver — build_model + log-capturing solve + incumbent loader.
#   5. Visualization — Altair layout figure with rectangles + pipe overlay.
#   6. Tab renderers — Layout, Data, Formulation, Logs.
#   7. Main — sidebar widgets + tab assembly.
# =============================================================================

import base64
import contextlib
import io
import math
import os
import random
import time
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition


def _materialize_gurobi_license():
    """Production license shim. Fly secrets surface as environment
    variables, but gurobipy wants a license FILE — so if the three WLS
    values are present and no license file is configured, write one to
    the home directory and point GRB_LICENSE_FILE at it. Local dev is
    untouched: there GRB_LICENSE_FILE already points at a file on disk.
    The values never enter the repo or image — only Fly's secret store
    and the container's private filesystem."""
    if os.environ.get("GRB_LICENSE_FILE"):
        return
    access = os.environ.get("GRB_WLSACCESSID")
    secret = os.environ.get("GRB_WLSSECRET")
    license_id = os.environ.get("GRB_LICENSEID")
    if not (access and secret and license_id):
        return
    lic_path = Path.home() / "gurobi.lic"
    if not lic_path.exists():
        lic_path.write_text(
            f"WLSACCESSID={access}\n"
            f"WLSSECRET={secret}\n"
            f"LICENSEID={license_id}\n",
            encoding="utf-8",
        )
    os.environ["GRB_LICENSE_FILE"] = str(lic_path)


_materialize_gurobi_license()


# ── 1. Page config + CSS + home-logo ──────────────────────────────────────────

st.set_page_config(
    page_title="Plant Layout",
    page_icon="favicon.png",
    layout="wide",
)

# Fixed-corner home logo (no sidebar — all controls are inline on the
# Optimizer tab).
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
    <style>
    .home-logo-corner {
        position: fixed;
        top: 0.5rem;
        left: 0.75rem;
        z-index: 999999;
    }
    .home-logo-corner img {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: block;
    }
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem !important;
    }
    </style>
    """
    f'<a href="https://griffith-pse.com" target="_self" '
    f'class="home-logo-corner">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
    f"</a>",
    unsafe_allow_html=True,
)


# ── 2. Constants and defaults ─────────────────────────────────────────────────

MAX_OBJECTS = 25           # rack + up to 24 others (default instance stays 15)
MIN_OBJECTS = 2            # rack + ≥1 object (sym_1/sym_2 reference block 2)

DIM_MIN, DIM_MAX = 1, 9    # editable length / width range
DIM_RAND_MAX = 3           # Randomize draws dimensions from [1, 3]
COST_MIN, COST_MAX = 1, 9  # editable pipe-cost range (each object pipes to its assigned rack end)
COST_RAND_MAX = 3          # Randomize draws costs from [1, 3]

# Weight on the plant bounding box (l_f + w_f) relative to the cost-weighted
# piping in the objective. The two terms are not in the same natural units, so
# this is the size-vs-piping knob. Default 1, no UI control.
FOOTPRINT_WEIGHT = 1.0

# The rack (object 1) spans the plant length: fixed long-and-thin dims, and
# always the longest object so every instance stays feasible. Reset and
# Randomize both keep these — only the other objects' dims/costs change.
RACK_LEN, RACK_WID = 9, 1
DEFAULT_N = 15             # objects present on first load / after Reset

# Direct unit-to-unit connections (close-coupled pairs). The instance rolls a
# full disjoint pairing of the non-rack objects; the "Connections" control
# picks how many of those pairs are active, and "Cost ×" scales their pipe
# cost relative to the rack tie-in range (dedicated large-bore routing).
DEFAULT_PAIRS = 3
PAIR_WEIGHT_MIN, PAIR_WEIGHT_MAX, PAIR_WEIGHT_DEFAULT = 1, 10, 10

# Minimum separation distance (integer stepper).
D_MIN, D_MAX, D_DEFAULT = 0, 3, 1

# Time-limit presets for the inline radio (label → seconds).
_TIME_LIMITS = {"10 s": 10, "30 s": 30, "60 s": 60}

# Solution pool: ask Gurobi for the best POOL_SIZE feasible layouts, then show up
# to MAX_SOLUTIONS distinct ones in the layout selector. The degeneracy
# breaking constraint in build_model is what makes pooled solutions distinct
# physical layouts rather than duplicate indicator encodings of one layout.
POOL_SIZE = 40      # candidates pulled from Gurobi's pool before diverse pick.
                    # PoolSearchMode=2 keeps the n BEST solutions, which
                    # cluster around similar geometry — a larger pool retains
                    # the worse-but-different layouts the diverse pick needs.
MAX_SOLUTIONS = 5   # distinct layouts shown in the selector

# RNG seed for Randomize; bumped each click for a fresh instance.
DEFAULT_SEED = 1

# Categorical palette — each object's index drives BOTH its editor badge color
# and its block fill in the layout, so the two views stay visually linked
# (object 1, the rack, gets the first color).
# At least MAX_OBJECTS distinct colors so no two objects ever share one (the
# index→color map is `_PALETTE[(i-1) % len]`, so the list must be >= the object
# count to avoid repeats).
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#9467BD",
    "#2CA02C", "#D62728", "#8C564B", "#E377C2", "#17BECF", "#BCBD22",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#C5B0D5", "#C49C94", "#7F7F7F",
    "#DBDB8D",
]


# ── 3. State helpers ──────────────────────────────────────────────────────────
#
# The instance is an ordered list of objects. objs[0] is the rack (object 1):
# every other object has a single pipe cost to the rack and a north/south end it
# ties into, and objects don't connect to each other. Objects carry stable
# integer ids so per-row editor widgets keep their state across add/delete;
# length/width/cost/side map id → value (the rack's cost and side are unused).

def _gen_objects(seed, objs):
    """Roll dimensions and pipe costs for the NON-rack objects, leaving the
    rack (first object) pinned at its fixed RACK_LEN×RACK_WID. Used by both the
    default instance and Randomize, so neither ever resizes the rack."""
    rng = random.Random(seed)
    objs = list(objs)
    rack_id = objs[0]
    length = {rack_id: RACK_LEN}
    width = {rack_id: RACK_WID}
    cost = {rack_id: 0}
    for oid in objs[1:]:
        length[oid] = rng.randint(DIM_MIN, DIM_RAND_MAX)
        width[oid] = rng.randint(DIM_MIN, DIM_RAND_MAX)
        cost[oid] = rng.randint(1, COST_RAND_MAX)
    # Each non-rack object ties into the north ("N") or south ("S") end of the
    # rack; the side is fixed instance data. Drawn after the dims/costs so adding
    # it does not disturb their RNG sequence.
    side = {oid: rng.choice(["N", "S"]) for oid in objs[1:]}
    return objs, length, width, cost, side


def _gen_pairs(seed, objs):
    """Roll a full disjoint pairing of the non-rack objects. Every pair has a
    base cost of 1, so the 'Cost ×' control IS each pair's pipe cost. The
    'Connections' control slices the first k pairs, so raising the count
    extends the active set without re-rolling existing pairs. Seeded
    independently of _gen_objects so the two draws don't perturb each
    other."""
    rng = random.Random(seed + 10_000_019)
    units = list(objs[1:])
    rng.shuffle(units)
    all_pairs = [(units[k], units[k + 1]) for k in range(0, len(units) - 1, 2)]
    pair_base = {p: 1 for p in all_pairs}
    return all_pairs, pair_base


def _default_data():
    """Initial / Reset instance: the rack plus DEFAULT_N-1 small objects."""
    return _gen_objects(DEFAULT_SEED, list(range(1, DEFAULT_N + 1)))


def _randomize_data(seed, objs):
    """Re-roll only the non-rack objects, preserving the current object count
    and row ids; the rack keeps its fixed dimensions."""
    return _gen_objects(seed, objs)


def _block_label(i):
    """Display name for block index i: the rack (object 1) reads 'Rack', and
    every other object is renumbered from 1 (so block 2 → '1', block 3 → '2')."""
    return "Rack" if int(i) == 1 else str(int(i) - 1)


def _set_data(objs, length, width, cost, side):
    ss = st.session_state
    ss["objs"], ss["length"], ss["width"], ss["cost"], ss["side"] = (
        list(objs), dict(length), dict(width), dict(cost), dict(side)
    )


def _init_state():
    ss = st.session_state
    ss.setdefault("rotate", False)
    ss.setdefault("d_min", D_DEFAULT)
    ss.setdefault("seed", DEFAULT_SEED)
    ss.setdefault("_obj_ver", 0)
    ss.setdefault("side", {})
    ss.setdefault("n_pairs", DEFAULT_PAIRS)
    ss.setdefault("pair_weight", PAIR_WEIGHT_DEFAULT)
    if "objs" not in ss:
        _set_data(*_default_data())
    if "all_pairs" not in ss:
        ss["all_pairs"], ss["pair_base"] = _gen_pairs(ss["seed"], ss["objs"])
    # Reset / Randomize set a one-shot flag and rerun; we apply it here, before
    # any editor widget is instantiated, so widget-backed keys don't clash.
    if ss.pop("_pending_reset", False):
        _set_data(*_default_data())
        ss["all_pairs"], ss["pair_base"] = _gen_pairs(DEFAULT_SEED, ss["objs"])
        ss["n_pairs"] = DEFAULT_PAIRS
        ss["pair_weight"] = PAIR_WEIGHT_DEFAULT
        ss["_obj_ver"] += 1
        ss.pop("res", None)
    if ss.pop("_pending_random", False):
        ss["seed"] += 1
        _set_data(*_randomize_data(ss["seed"], ss["objs"]))
        ss["all_pairs"], ss["pair_base"] = _gen_pairs(ss["seed"], ss["objs"])
        ss["_obj_ver"] += 1
        ss.pop("res", None)


def _active_pairs(ss):
    """The first n_pairs of the rolled pairing, weight-scaled: list of
    (id_a, id_b, cost)."""
    k = min(int(ss["n_pairs"]), len(ss["all_pairs"]))
    w = int(ss["pair_weight"])
    return [(a, b, w * ss["pair_base"][(a, b)])
            for a, b in ss["all_pairs"][:k]]


def add_object():
    ss = st.session_state
    if len(ss["objs"]) >= MAX_OBJECTS:
        return
    new_id = (max(ss["objs"]) + 1) if ss["objs"] else 1
    ss["objs"] = ss["objs"] + [new_id]
    ss["length"] = {**ss["length"], new_id: 2}
    ss["width"] = {**ss["width"], new_id: 2}
    ss["cost"] = {**ss["cost"], new_id: 1}
    ss["side"] = {**ss["side"], new_id: "N"}
    ss.pop("res", None)


def _delete_object(oid):
    """on_click for a per-row delete button. The rack (first object) can't be
    deleted, nor can the list drop below MIN_OBJECTS."""
    ss = st.session_state
    if oid == ss["objs"][0] or len(ss["objs"]) <= MIN_OBJECTS:
        return
    ss["objs"] = [i for i in ss["objs"] if i != oid]
    for key in ("length", "width", "cost", "side"):
        ss[key] = {i: v for i, v in ss[key].items() if i != oid}
    # Drop any unit-to-unit pair touching the deleted object.
    ss["all_pairs"] = [p for p in ss["all_pairs"] if oid not in p]
    ss["pair_base"] = {p: c for p, c in ss["pair_base"].items() if oid not in p}
    ss.pop("res", None)


def _objs_to_inputs(ss):
    """Map the object list onto build_model's (n, l0, w0, cmat).

    Two implicit tie-in headers are appended after the user objects: a north
    header (index nu+1) and a south header (nu+2), each zero length and the
    rack's width, which build_model pins to the rack's top and bottom ends. Each
    non-rack object's pipe cost goes to its assigned header — north or south per
    the instance data — rather than to the main rack, whose column stays zero."""
    objs = ss["objs"]
    nu = len(objs)
    north, south = nu + 1, nu + 2
    n = nu + 2
    l0 = {p: int(ss["length"][objs[p - 1]]) for p in range(1, nu + 1)}
    w0 = {p: int(ss["width"][objs[p - 1]]) for p in range(1, nu + 1)}
    l0[north] = l0[south] = 0
    w0[north] = w0[south] = RACK_WID
    cmat = [[0.0] * n for _ in range(n)]
    for p in range(2, nu + 1):
        h = north if ss["side"].get(objs[p - 1], "N") == "N" else south
        cmat[h - 1][p - 1] = float(ss["cost"][objs[p - 1]])
    # Direct unit-to-unit connections: weight-scaled pipe costs between the
    # active close-coupled pairs (lower-triangular, matching build_model).
    pos = {oid: k + 1 for k, oid in enumerate(objs)}
    for a, b, c in _active_pairs(ss):
        pa, pb = pos.get(a), pos.get(b)
        if pa is None or pb is None:
            continue
        cmat[max(pa, pb) - 1][min(pa, pb) - 1] += float(c)
    return n, l0, w0, cmat


# ── 4. Solver ─────────────────────────────────────────────────────────────────

def build_model(n, l0, w0, cmat, d_uniform, rotate, sym):
    """Construct the GDP plant-layout model.

    Args:
        n         : int, number of blocks
        l0, w0    : {1..n: float} block default length/width
        cmat      : n×n list-of-lists, lower-triangular pipe costs (cmat[i-1][j-1] for i>j)
        d_uniform : float, min separation distance applied to every pair
        rotate    : bool, allow 90° rotation per block
        sym       : 0 or 1, enable symmetry-breaking on blocks 1 and 2

    Returns the unsolved Pyomo `ConcreteModel`. The caller applies the GDP
    transformation and runs the solver.
    """
    m = pyo.ConcreteModel()

    # Blocks indexed 1..n; pair set is the strict lower triangle (i > j).
    m.n = pyo.Set(ordered=True, initialize=pyo.RangeSet(1, n))
    m.p = pyo.Set(initialize=m.n * m.n, dimen=2,
                  filter=lambda m, i, j: i > j)

    # Default-orientation dimensions.
    m.w0 = pyo.Param(m.n, initialize=w0)
    m.l0 = pyo.Param(m.n, initialize=l0)

    # Pair parameters: pipe cost and minimum required separation.
    c_dict = {(i, j): float(cmat[i - 1][j - 1]) for i, j in m.p}
    d_dict = {(i, j): float(d_uniform) for i, j in m.p}
    # Tie-in headers (zero-length objects) are virtual pipe targets, not physical
    # equipment, so they impose no separation: every pair involving a header gets
    # zero minimum distance. Header–rack needs it (the header sits flush on the
    # rack's end), and header–object needs it too, otherwise the invisible header
    # line would push real blocks 1 unit away from the rack's ends.
    _hdr = {i for i in m.n if l0[i] == 0}
    if _hdr:
        for (i, j) in d_dict:
            if i in _hdr or j in _hdr:
                d_dict[(i, j)] = 0.0
    m.c = pyo.Param(m.p, initialize=c_dict)
    m.d = pyo.Param(m.p, initialize=d_dict)

    # Conservative upper bound on placement coordinates: stack all blocks
    # along one axis at their longest dimension. Keeps the LP relaxation
    # bounded without being too loose to be useful.
    m.UB = pyo.Param(initialize=sum(max(m.l0[i], m.w0[i]) for i in m.n))

    # Decision variables.
    m.x = pyo.Var(m.n, bounds=(0, m.UB))      # lower-left x
    m.y = pyo.Var(m.n, bounds=(0, m.UB))      # lower-left y
    m.l = pyo.Var(m.n, bounds=(0, m.UB))      # block length (= l0 unless rotated)
    m.w = pyo.Var(m.n, bounds=(0, m.UB))      # block width  (= w0 unless rotated)
    m.dx = pyo.Var(m.p, bounds=(0, m.UB))     # x-axis (horizontal) edge gap
    m.dy = pyo.Var(m.p, bounds=(0, m.UB))     # y-axis (vertical) edge gap
    m.l_f = pyo.Var(within=pyo.NonNegativeReals)  # plant length
    m.w_f = pyo.Var(within=pyo.NonNegativeReals)  # plant width

    # Plant bounds: every block lies inside the plant's bounding box.
    # Length is the vertical (y) axis; width the horizontal (x) axis.
    @m.Constraint(m.n)
    def plant_length(m, i):
        return m.l_f >= m.y[i] + m.l[i]

    @m.Constraint(m.n)
    def plant_width(m, i):
        return m.w_f >= m.x[i] + m.w[i]

    # Pipe rack (block 1) spans the plant length (the vertical y-axis):
    # pinned at y=0 with the plant length fixed to the rack's length. Every
    # other object then fits within [0, l_1] in y and sits to the LEFT or RIGHT
    # of the rack (in x). The rack's x is free; only the WIDTH (horizontal x)
    # is minimized.
    m.rack_at_origin = pyo.Constraint(expr=m.y[1] == 0)
    m.plant_len_eq_rack = pyo.Constraint(expr=m.l_f == m.l[1])

    # Implicit pipe-rack tie-in headers: each zero-length object is pinned flush
    # to one end of the main rack and aligned to its x. The lowest-index header
    # ties to the north (top) end at y = l_f; the next to the south (bottom) end
    # at y = 0.
    _headers = sorted(i for i in m.n if l0[i] == 0)
    if len(_headers) >= 1:
        _hn = _headers[0]
        m.pin_north_x = pyo.Constraint(expr=m.x[_hn] == m.x[1])
        m.pin_north_y = pyo.Constraint(expr=m.y[_hn] == m.l_f)
    if len(_headers) >= 2:
        _hs = _headers[1]
        m.pin_south_x = pyo.Constraint(expr=m.x[_hs] == m.x[1])
        m.pin_south_y = pyo.Constraint(expr=m.y[_hs] == 0)

    # Rectilinear center-to-center distances, defined GLOBALLY (not inside the
    # disjunction): dx_ij >= |center_x(i) - center_x(j)| and dy_ij the same in
    # y — the literature-standard distance convention. They're minimized in
    # the objective, so each settles to the true center distance. Keeping them
    # out of the disjuncts makes the objective independent of which spatial
    # relation is chosen — the disjunction below decides only non-overlap.
    @m.Constraint(m.p)
    def dx_lb_a(m, i, j):
        return m.dx[i, j] >= (m.x[i] + m.w[i] / 2) - (m.x[j] + m.w[j] / 2)

    @m.Constraint(m.p)
    def dx_lb_b(m, i, j):
        return m.dx[i, j] >= (m.x[j] + m.w[j] / 2) - (m.x[i] + m.w[i] / 2)

    @m.Constraint(m.p)
    def dy_lb_a(m, i, j):
        return m.dy[i, j] >= (m.y[i] + m.l[i] / 2) - (m.y[j] + m.l[j] / 2)

    @m.Constraint(m.p)
    def dy_lb_b(m, i, j):
        return m.dy[i, j] >= (m.y[j] + m.l[j] / 2) - (m.y[i] + m.l[i] / 2)

    # Symmetry breaking: anchor block 1 left-of block 2's center, killing the
    # left/right mirror. Only the horizontal mirror is a symmetry here: the
    # top/bottom mirror is NOT, because flipping y would swap each object's
    # north/south tie-in, so the old vertical cut (rack-below-block-2) is gone —
    # it would have wrongly forced block 2 into the upper half.
    if sym == 1:
        @m.Constraint()
        def sym_1(m):
            return m.x[1] + m.w[1] / 2 <= m.x[2] + m.w[2] / 2

    # Objective: minimize plant size + Σ pipe-weighted Manhattan distances.
    m.obj = pyo.Objective(
        expr=FOOTPRINT_WEIGHT * (m.l_f + m.w_f)
             + sum(m.c[i, j] * (m.dx[i, j] + m.dy[i, j]) for i, j in m.p),
        sense=pyo.minimize,
    )

    # Non-overlap GDP: 4-way disjunction per pair, one spatial relation each
    # with the minimum separation d baked in. Distance is handled by the global
    # dx/dy constraints above, so these decide only feasibility (which pairs are
    # separated, on which axis), never the objective.
    #
    # Degeneracy breaking constraint (d-aware Trespalacios & Grossmann,
    # continuous-valid form): the left/right disjuncts additionally require
    # the blocks to overlap vertically within d, so a pair with vertical gap
    # > d can only route through above/below — one encoding per physical
    # layout, keeping the solution pool free of duplicate encodings. The
    # offset must be -d, not the tighter -(d-1): the d-1 form is only
    # optimum-preserving when an integer-optimal layout is guaranteed, which
    # the center-to-center objective breaks for odd dimensions (centers can
    # be optimal at half-integers). The -d form is optimum-preserving for
    # arbitrary continuous coordinates.
    @m.Disjunction(m.p)
    def no_overlap(m, i, j):
        vov = [m.y[i] + m.l[i] >= m.y[j] - m.d[i, j],
               m.y[j] + m.l[j] >= m.y[i] - m.d[i, j]]
        return [
            [m.x[i] + m.w[i] + m.d[i, j] <= m.x[j]] + vov,   # i left of j
            [m.x[j] + m.w[j] + m.d[i, j] <= m.x[i]] + vov,   # i right of j
            [m.y[i] + m.l[i] + m.d[i, j] <= m.y[j]],         # i below j
            [m.y[j] + m.l[j] + m.d[i, j] <= m.y[i]],         # i above j
        ]

    # Rotation GDP (optional): 2-way disjunction per block — EXCEPT block 1
    # (the rack), which keeps a fixed orientation even when rotation is on.
    # Fixing the rack is optimum-preserving: it only canonicalizes the
    # layout's overall orientation (the transpose symmetry), and every other
    # block can still rotate to recover the transposed layout at the same
    # objective. It also keeps the rack's footprint stable for the viewer.
    if rotate:
        # Rotation disjunction over the real non-rack blocks only. The rack and
        # the zero-length tie-in headers keep a fixed orientation.
        _rot_blocks = [i for i in m.n if i != 1 and l0[i] != 0]

        @m.Disjunction(_rot_blocks)
        def rotation(m, i):
            return [
                [m.l[i] == m.l0[i], m.w[i] == m.w0[i]],   # default
                [m.l[i] == m.w0[i], m.w[i] == m.l0[i]],   # 90° rotated
            ]

        # Rack and headers keep their default orientation regardless.
        _fixed = [i for i in m.n if i not in _rot_blocks]

        @m.Constraint(_fixed)
        def fix_l(m, i):
            return m.l[i] == m.l0[i]

        @m.Constraint(_fixed)
        def fix_w(m, i):
            return m.w[i] == m.w0[i]
    else:
        @m.Constraint(m.n)
        def fix_l(m, i):
            return m.l[i] == m.l0[i]

        @m.Constraint(m.n)
        def fix_w(m, i):
            return m.w[i] == m.w0[i]

    return m


class _LicenseBusyError(RuntimeError):
    """Raised when Gurobi's WLS checkout fails even after a retry —
    typically the license's concurrent-session seats are all taken.
    solve() maps this onto the `license_busy` status."""


def _run_gurobi(m, time_limit, extract_fn, pool_size):
    """Solve the (already GDP-transformed) MILP via the NATIVE appsi
    Gurobi interface with the solution pool on, calling `extract_fn` once per
    pooled solution (it reads the model while that solution is loaded) and
    returning the list. Returns (termination_condition, primal, dual, log,
    solutions).

    The native interface (not the legacy SolverFactory("appsi_gurobi"))
    is required: the legacy wrapper's symbol-map bookkeeping crashes on
    GDP-transformed models ('DisjunctData' has no attribute 'solutions').
    Gurobi checks out a WLS seat when its environment starts; a checkout
    collision gets one quiet retry, then surfaces as license_busy, and
    the seat is always released afterward."""
    from pyomo.contrib.appsi.solvers import Gurobi as AppsiGurobi

    opt = AppsiGurobi()
    opt.config.time_limit = float(time_limit)
    opt.config.load_solution = False
    opt.config.stream_solver = True  # log into the redirected stdout
    # Solution pool: keep the best `pool_size` feasible solutions found.
    opt.gurobi_options['PoolSearchMode'] = 2          # find the n best
    opt.gurobi_options['PoolSolutions'] = int(pool_size)
    # MIPFocus=1 (feasibility focus): the app's short time limits reward
    # finding many good incumbents — that's what fills the solution pool the
    # layout selector depends on. Bound-focused settings prove faster but
    # find few incumbents along the way, starving the pool.
    opt.gurobi_options['MIPFocus'] = 1
    buf = io.StringIO()
    sols = []
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for attempt in (1, 2):
                try:
                    res = opt.solve(m)
                    break
                except Exception as e:
                    lowered = str(e).lower()
                    if "license" in lowered or "wls" in lowered:
                        if attempt == 1:
                            time.sleep(2.0)
                            continue
                        raise _LicenseBusyError(str(e)) from e
                    raise
            if res.best_feasible_objective is not None:
                # Pull each pooled solution into a plain layout dict while the
                # model is still loaded. load_vars(solution_number=k) switches
                # Gurobi's active solution to pool member k (k=0 is the best).
                n_pool = int(opt._solver_model.SolCount)
                for k in range(min(n_pool, int(pool_size))):
                    res.solution_loader.load_vars(solution_number=k)
                    sols.append(extract_fn())
    finally:
        try:
            opt.release_license()
        except Exception:
            pass

    # Scrub license-identifying lines before the log reaches the Logs tab
    # (Gurobi's WLS banner prints the license ID and registrant).
    log = "\n".join(
        ln for ln in buf.getvalue().splitlines()
        if not any(k in ln.lower()
                   for k in ("wls", "registered to", "academic license"))
    )
    # Map the appsi TerminationCondition onto the legacy enum this module
    # branches on; names match for the cases we handle, else `unknown`.
    tc = getattr(
        TerminationCondition, res.termination_condition.name,
        TerminationCondition.unknown,
    )
    return tc, res.best_feasible_objective, res.best_objective_bound, log, sols


def _extract_layout(m, rotate, l0):
    """Read the model's current solution into a plain layout dict (blocks, pipe
    pairs, objective, plant size, total piping cost). Called once per pooled
    solution, each time with a different pool member loaded onto the model."""
    blocks = []
    for i in m.n:
        blocks.append({
            "i": i,
            "x": float(pyo.value(m.x[i])),
            "y": float(pyo.value(m.y[i])),
            "l": float(pyo.value(m.l[i])),
            "w": float(pyo.value(m.w[i])),
            "rotated": bool(rotate and abs(float(pyo.value(m.l[i])) - l0[i]) > 1e-6),
            "is_header": l0[i] == 0,
        })
    pairs = []
    for (i, j) in m.p:
        pairs.append({
            "i": i, "j": j,
            "c": float(pyo.value(m.c[i, j])),
            "dx": float(pyo.value(m.dx[i, j])),
            "dy": float(pyo.value(m.dy[i, j])),
        })
    return {
        "blocks": blocks,
        "pairs": pairs,
        "obj": float(pyo.value(m.obj)),
        "plant": (float(pyo.value(m.l_f)), float(pyo.value(m.w_f))),
        "pipe_cost": sum(p["c"] * (p["dx"] + p["dy"]) for p in pairs),
    }


def _relation_vector(blocks):
    """The spatial relation each block pair takes (left / right / below /
    above), as a tuple over all pairs, derived from positions. Two layouts that
    route many pairs differently are structurally different even at a similar
    cost — so the count of differing entries is a translation-invariant
    'how different are these two layouts' distance."""
    bs = {b["i"]: b for b in blocks}
    ids = sorted(bs)
    tol = 1e-6
    rels = []
    for idx, i in enumerate(ids):
        bi = bs[i]
        for j in ids[:idx]:
            bj = bs[j]
            if bi["x"] + bi["w"] <= bj["x"] + tol:
                rels.append(0)        # i left of j
            elif bj["x"] + bj["w"] <= bi["x"] + tol:
                rels.append(1)        # i right of j
            elif bi["y"] + bi["l"] <= bj["y"] + tol:
                rels.append(2)        # i below j
            else:
                rels.append(3)        # i above j
    return tuple(rels)


def _select_diverse(layouts, k):
    """Greedily choose up to k layouts that are as different from each other as
    possible: start from the cheapest, then repeatedly add the layout whose
    nearest already-chosen layout is the most different (max-min over the
    relation vectors). Spreads the picks out instead of clustering at the
    cheap end."""
    if len(layouts) <= k:
        return list(layouts)
    vecs = [_relation_vector(s["blocks"]) for s in layouts]

    def dist(a, b):
        return sum(1 for x, y in zip(vecs[a], vecs[b]) if x != y)

    chosen = [0]   # layouts[0] is the cheapest (pool is best-first)
    while len(chosen) < k:
        nxt, best = None, -1
        for c in range(len(layouts)):
            if c in chosen:
                continue
            d = min(dist(c, s) for s in chosen)
            if d > best:
                best, nxt = d, c
        chosen.append(nxt)
    return [layouts[c] for c in chosen]


def solve(n, l0, w0, cmat, d_uniform, rotate, sym, time_limit):
    """Top-level entrypoint. Returns a plain dict the UI can stash in
    session_state without holding a live Pyomo model."""

    t0 = time.time()
    m = build_model(n, l0, w0, cmat, d_uniform, rotate, sym)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)

    try:
        tc, primal, dual, log, raw_sols = _run_gurobi(
            m, time_limit, lambda: _extract_layout(m, rotate, l0), POOL_SIZE)
    except _LicenseBusyError:
        return {
            "status": "license_busy",
            "message": (
                "The Gurobi license is busy (it allows a limited number of "
                "concurrent solves). Wait a few seconds and click Solve again."
            ),
            "log": "",
        }
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                f"Gurobi solver not available. Run `pip install gurobipy` and "
                f"provide a license. ({e})"
            ),
            "log": "",
        }

    # Status branch. A feasible incumbent exists iff Gurobi returned a
    # primal objective.
    feasible = primal is not None
    if tc == TerminationCondition.optimal:
        status = "optimal"
    elif tc in (TerminationCondition.maxTimeLimit,
                TerminationCondition.userInterrupt):
        status = "incumbent" if feasible else "no_feasible"
    elif tc in (TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded):
        status = "infeasible"
    elif tc == TerminationCondition.unbounded:
        status = "unbounded"
    else:
        # Anything else (e.g. interrupted/unknown) still counts as an
        # incumbent if a feasible solution was loaded.
        status = "incumbent" if feasible else str(tc)

    if status not in ("optimal", "incumbent"):
        return {"status": status, "log": log}

    # Dedup the pooled layouts by block geometry, then pick a spread of
    # MAX_SOLUTIONS that are as structurally different as possible (greedy
    # max-min over each pair's spatial relation), and order the chosen set by
    # cost. The pool returns the cheapest layouts, which tend to cluster;
    # selecting for difference surfaces genuinely distinct arrangements (at a
    # somewhat higher cost for the alternatives).
    seen = set()
    distinct = []
    for s in raw_sols:
        fp = tuple((b["i"], round(b["x"], 3), round(b["y"], 3),
                    round(b["w"], 3), round(b["l"], 3)) for b in s["blocks"])
        if fp in seen:
            continue
        seen.add(fp)
        distinct.append(s)
    solutions = _select_diverse(distinct, MAX_SOLUTIONS)
    solutions.sort(key=lambda s: s["obj"])

    if not solutions:
        # Termination says feasible but nothing came back — treat as no usable
        # layout rather than crash the UI.
        return {"status": "no_feasible", "log": log}

    # Solve-level optimality gap: best objective vs Gurobi's dual bound. The
    # metrics row recomputes a per-layout gap from lower_bound when the user
    # selects a pooled alternative.
    best_obj = solutions[0]["obj"]
    lower_bound = dual if (dual is not None and dual != float("-inf")
                           and dual == dual) else None
    gap = None
    if lower_bound is not None and lower_bound > 0 and best_obj > 0:
        gap = max(0.0, (best_obj - lower_bound) / max(abs(best_obj), 1e-12))

    return {
        "status": status,
        "solutions": solutions,
        # Best layout's fields also at top level, so any consumer reading
        # res["blocks"]/["plant"]/etc. still gets the headline solution.
        **solutions[0],
        "lower_bound": lower_bound,
        "gap": gap,
        "elapsed": time.time() - t0,
        "log": log,
    }


# ── 5. Visualization ─────────────────────────────────────────────────────────

# "Connectivity" = for each block, the sum of its incident pipe costs. No
# longer drives the fill (blocks use their palette/badge color now), but still
# feeds the block tooltip and the hover-to-highlight adjacency.
def _connectivity(blocks, pairs):
    """Per-block realized piping cost: each pipe touching a block contributes
    its cost-weighted rectilinear distance c·(dx+dy) — the same quantity summed
    into 'Total piping cost'. So an object shows its own pipe's cost and the
    rack (touched by every pipe) shows the grand total. The coefficient c alone
    isn't a cost; it ignores distance. (Preview pairs carry no dx/dy and have
    c=0, so .get keeps them at zero.)"""
    conn = {b["i"]: 0.0 for b in blocks}
    for p in pairs:
        cost = p["c"] * (p.get("dx", 0.0) + p.get("dy", 0.0))
        conn[p["i"]] = conn.get(p["i"], 0.0) + cost
        conn[p["j"]] = conn.get(p["j"], 0.0) + cost
    return conn


def _pipe_segments(block_i, block_j):
    """Stylized elbow between two blocks, routed face-to-face rather than
    corner-to-corner: the pipe leaves block i at the midline of the face
    toward block j, runs to block j's center coordinate, and enters j through
    the middle of a face. NOTE: unlike the rack tie-in lanes, the drawn
    length here no longer equals the modeled gap dx + dy — the model costs
    nearest-edge clearance, the drawing favors legibility.

    Returns a list of one or two segments, each {"x", "y", "x2", "y2"}.
    """
    xi, yi, wi, li = block_i["x"], block_i["y"], block_i["w"], block_i["l"]
    xj, yj, wj, lj = block_j["x"], block_j["y"], block_j["w"], block_j["l"]
    cx_i, cy_i = xi + wi / 2, yi + li / 2
    cx_j, cy_j = xj + wj / 2, yj + lj / 2

    disjoint_x = (xi + wi <= xj) or (xj + wj <= xi)

    if disjoint_x:
        # Horizontal-first: leave i's facing side at its mid-height.
        src_x = xi + wi if cx_j > cx_i else xi
        if yj <= cy_i <= yj + lj:
            # i's midline meets j's side directly — single straight run.
            dst_x = xj if cx_j > cx_i else xj + wj
            return [{"x": src_x, "y": cy_i, "x2": dst_x, "y2": cy_i}]
        # Elbow: run to j's center x, then drop into j's near face.
        dst_y = yj if cy_i < yj else yj + lj
        return [
            {"x": src_x, "y": cy_i, "x2": cx_j, "y2": cy_i},
            {"x": cx_j, "y": cy_i, "x2": cx_j, "y2": dst_y},
        ]

    # Overlapping in x → they're disjoint in y. Vertical-first, symmetric.
    src_y = yi + li if cy_j > cy_i else yi
    if xj <= cx_i <= xj + wj:
        dst_y = yj if cy_j > cy_i else yj + lj
        return [{"x": cx_i, "y": src_y, "x2": cx_i, "y2": dst_y}]
    dst_x = xj if cx_i < xj else xj + wj
    return [
        {"x": cx_i, "y": src_y, "x2": cx_i, "y2": cy_j},
        {"x": cx_i, "y": cy_j, "x2": dst_x, "y2": cy_j},
    ]


def build_layout_chart(res):
    """Multi-layered Altair chart for the optimal layout.

    Layers (back-to-front):
      1. Outer plant bounding box (dashed)
      2. Pipe overlay — L-shaped paths via edge-port routing, opacity ∝ c_ij,
         linked-hover dims non-hovered pipes
      3. Block rectangles (fill = connectivity), border highlights orange
         when a pipe connecting this block is hovered
      4. Block-id labels at centers

    Width is the horizontal (x) axis and length the vertical (y) axis, matching
    the formulation — so the rack, spanning the fixed length, renders
    vertically while the variable width grows horizontally (a wide layout).
    """
    blocks = res["blocks"]
    pairs = res["pairs"]
    l_f, w_f = res["plant"]

    conn = _connectivity(blocks, pairs)
    blocks_by_id = {b["i"]: b for b in blocks}

    # Adjacency map — for each block, the set of other blocks it's connected
    # to via a non-zero-cost pipe. Embedded in df_blocks as a comma-delimited
    # string (",2,3,5,") so the linked block-hover expression can test
    # membership via `indexof(...)` in Vega.
    adj = {b["i"]: set() for b in blocks}
    for p in pairs:
        if p["c"] > 0:
            adj[p["i"]].add(p["j"])
            adj[p["j"]].add(p["i"])

    df_blocks = pd.DataFrame([{
        "i":   b["i"],
        "label": _block_label(b["i"]),
        # Rack label reads top-to-bottom along its tall, thin bar; the other
        # (roughly square) blocks keep horizontal labels.
        "angle": 90 if int(b["i"]) == 1 else 0,
        "x":   b["x"],
        "y":   b["y"],
        "x2":  b["x"] + b["w"],
        "y2":  b["y"] + b["l"],
        "cx":  b["x"] + b["w"] / 2,
        "cy":  b["y"] + b["l"] / 2,
        "l":   b["l"],
        "w":   b["w"],
        "rotated": "yes" if b["rotated"] else "no",
        "connectivity": conn[b["i"]],
        "connected_str": "," + ",".join(str(x) for x in sorted(adj[b["i"]])) + ",",
        "color": ("#ffffff" if int(b["i"]) == 1
                  else _PALETTE[(int(b["i"]) - 1) % len(_PALETTE)]),
    } for b in blocks if not b.get("is_header")])

    # Pipe dataframe — each pipe is an L drawn as two segments: a horizontal
    # feeder from the object's vertical centre to that object's own lane inside
    # the rack, then a vertical run along the lane to its north/south end. The
    # per-object lanes fan the vertical runs across the rack's width so they read
    # as separate pipes threaded through it. Both legs carry the same cost/color,
    # so a pipe's stroke width and hue are consistent across the bend. Integer
    # i_id/j_id columns let the linked-hover selection match block IDs.
    rack_b = blocks_by_id.get(1)
    rx = rack_b["x"] if rack_b else 0.0
    rw = rack_b["w"] if rack_b else 1.0
    # One evenly-spaced lane inside the rack per object that pipes.
    _piping_ids = []
    for p in pairs:
        if p["c"] <= 0:
            continue
        _bi, _bj = blocks_by_id[p["i"]], blocks_by_id[p["j"]]
        if _bi.get("is_header") or _bj.get("is_header"):
            _ob = _bj if _bi.get("is_header") else _bi
            if _ob["i"] not in _piping_ids:
                _piping_ids.append(_ob["i"])
    _piping_ids.sort()
    lane_x = {oid: rx + rw * (k + 1) / (len(_piping_ids) + 1)
              for k, oid in enumerate(_piping_ids)}

    # Feeders run at each object's vertical centre, but where several objects
    # share a height their feeders would overlap; fan those into a small band
    # around the shared height so they read as parallel lines. Objects with a
    # unique height keep their exact centre.
    _groups = {}
    for oid in _piping_ids:
        ob = blocks_by_id[oid]
        _groups.setdefault(round(ob["y"] + ob["l"] / 2.0, 2), []).append(oid)
    feeder_y = {}
    _STAGGER = 0.22
    for _yc, _members in _groups.items():
        _members.sort(key=lambda o: lane_x.get(o, 0.0))
        _k = len(_members)
        for _j, _oid in enumerate(_members):
            _ob = blocks_by_id[_oid]
            _y = _yc + (_j - (_k - 1) / 2.0) * _STAGGER
            feeder_y[_oid] = min(max(_y, _ob["y"] + 0.05),
                                 _ob["y"] + _ob["l"] - 0.05)

    max_c = max((p["c"] for p in pairs), default=0.0)
    if max_c > 0:
        pipe_rows = []
        for p in pairs:
            if p["c"] <= 0:
                continue
            bi, bj = blocks_by_id[p["i"]], blocks_by_id[p["j"]]
            if bi.get("is_header") or bj.get("is_header"):
                obj_b = bj if bi.get("is_header") else bi
                hdr = bi if bi.get("is_header") else bj
                obj_id = int(obj_b["i"])
                # Start at the object's vertical centre, on its rack-facing edge;
                # feed horizontally to the object's lane, then run vertically
                # through the rack along that lane to the assigned end.
                y_c = feeder_y.get(obj_id, obj_b["y"] + obj_b["l"] / 2.0)
                edge_x = (obj_b["x"] if obj_b["x"] >= rx + rw
                          else obj_b["x"] + obj_b["w"])
                lx = lane_x.get(obj_id, rx + rw / 2.0)
                y_end = hdr["y"]                       # l_f (north) or 0 (south)
                seg_a = {"x": edge_x, "y": y_c, "x2": lx, "y2": y_c}
                seg_b = {"x": lx, "y": y_c, "x2": lx, "y2": y_end}
                feeder_len = abs(edge_x - lx)
                direction = "North" if hdr["y"] > l_f / 2 else "South"
                pair_label = f"{_block_label(obj_id)} → {direction}"
            else:
                uu_segs = _pipe_segments(bi, bj)
                feeder_len = 0.0
                pair_label = f"{_block_label(p['i'])}—{_block_label(p['j'])}"
                obj_id = int(p["i"])
            # Pipe takes its object's block color, so each line ties back to
            # the object it serves (same index→color mapping as the block
            # fill). Unit-to-unit pipes alternate the two endpoint colors in
            # short dashes along the whole path, so the connection reads as
            # belonging to both ends.
            if bi.get("is_header") or bj.get("is_header"):
                draw_segs = [(seg_a, _PALETTE[(obj_id - 1) % len(_PALETTE)]),
                             (seg_b, _PALETTE[(obj_id - 1) % len(_PALETTE)])]
            else:
                color_i = _PALETTE[(int(p["i"]) - 1) % len(_PALETTE)]
                color_j = _PALETTE[(int(p["j"]) - 1) % len(_PALETTE)]
                DASH = 0.1           # dash length in layout units
                draw_segs = []
                k = 0                # running dash index across all legs
                for seg in uu_segs:
                    dx_s = seg["x2"] - seg["x"]
                    dy_s = seg["y2"] - seg["y"]
                    seg_len = abs(dx_s) + abs(dy_s)    # legs are axis-aligned
                    n_dash = max(1, int(math.ceil(seg_len / DASH)))
                    for m in range(n_dash):
                        t0, t1 = m / n_dash, (m + 1) / n_dash
                        piece = {"x": seg["x"] + t0 * dx_s,
                                 "y": seg["y"] + t0 * dy_s,
                                 "x2": seg["x"] + t1 * dx_s,
                                 "y2": seg["y"] + t1 * dy_s}
                        draw_segs.append(
                            (piece, color_i if k % 2 == 0 else color_j))
                        k += 1
            for seg, seg_color in draw_segs:
                pipe_rows.append({
                    **seg,
                    "c": p["c"],                       # per-unit pipe price
                    "length": p["dx"] + p["dy"],       # Manhattan pipe length
                    "pair": pair_label,
                    "color": seg_color,
                    "feeder_len": feeder_len,
                    "i_id": int(p["i"]),
                    "j_id": int(p["j"]),
                })
        # Draw longest feeders first so shorter feeders land on top where they
        # overlap (later rows paint over earlier ones).
        df_pipes = pd.DataFrame(pipe_rows).sort_values(
            "feeder_len", ascending=False, kind="mergesort"
        )
    else:
        df_pipes = pd.DataFrame(
            columns=["x", "y", "x2", "y2", "c", "length", "pair", "color",
                     "feeder_len", "i_id", "j_id"]
        )

    # Domain spans with padding so nothing clips at the edges. Width is the
    # horizontal (x) axis, length the vertical (y) axis.
    pad = 0.05 * max(l_f, w_f, 1.0)
    x_dom = [-pad, w_f + pad]
    y_dom = [-pad, l_f + pad]

    # Equal-aspect sizing: identical pixels-per-unit on both axes, so the
    # layout is geometrically faithful and the integer ticks read as an even
    # square grid. Scale to the largest size fitting within BOTH a width and a
    # height cap — width fills the wide viz column for the usual wide-and-short
    # layout, while the height cap stops a tall, narrow instance from blowing
    # up vertically. A single scale factor keeps squares square.
    _x_span = (x_dom[1] - x_dom[0]) or 1.0
    _y_span = (y_dom[1] - y_dom[0]) or 1.0
    _max_w_px, _max_h_px = 1050, 520
    _scale = min(_max_w_px / _x_span, _max_h_px / _y_span)
    _w_px = max(160, round(_x_span * _scale))
    _h_px = max(160, round(_y_span * _scale))

    df_plant = pd.DataFrame([{"x": 0, "y": 0, "x2": w_f, "y2": l_f}])

    # ── Linked-hover selections ───────────────────────────────────────────
    # Two parallel selections: one bound to the pipe layer (`hover`), one to
    # the block layer (`block_hover`). They use different fields and feed
    # the same set of conditional expressions on each layer, so hovering
    # either a pipe OR a block produces the matched highlight pattern.
    #
    # `empty=True` is harmless here — we drive everything from explicit
    # `length(... || []) > 0` checks in the expressions below, so the
    # `empty` interpretation never affects the visible result.
    #
    # `nearest=False` (omitted) means the cursor must be directly on the
    # mark to trigger selection. No snap-from-afar.
    # The block-hover selection is always present. The pipe-hover selection
    # and its expressions exist only when there ARE pipes — otherwise the
    # stroke expression would reference a selection that was never added,
    # which breaks Vega rendering (the unsolved preview, or any all-zero-cost
    # instance, hits this).
    has_pipes = len(df_pipes) > 0
    block_hover = alt.selection_point(
        name="block_hover", on="mouseover", fields=["i"], empty=True,
    )
    if has_pipes:
        hover = alt.selection_point(
            name="hover", on="mouseover", fields=["i_id", "j_id"], empty=True,
        )
        # Pipe is bright if both selections empty, OR it's the hovered pipe,
        # OR it touches the hovered block.
        pipe_opacity_expr = (
            "(length(hover.i_id || []) === 0 && length(block_hover.i || []) === 0)"
            " || (length(hover.i_id || []) > 0 && datum.i_id === hover.i_id[0]"
            "     && datum.j_id === hover.j_id[0])"
            " || (length(block_hover.i || []) > 0 && (datum.i_id === block_hover.i[0]"
            "     || datum.j_id === block_hover.i[0]))"
        )
        # Block highlighted if it's an endpoint of the hovered pipe, IS the
        # hovered block, or is connected to it.
        block_stroke_expr = (
            "(length(hover.i_id || []) > 0 && (datum.i === hover.i_id[0]"
            "     || datum.i === hover.j_id[0]))"
            " || (length(block_hover.i || []) > 0 && (datum.i === block_hover.i[0]"
            "     || indexof(datum.connected_str, ',' + toString(block_hover.i[0]) + ',') >= 0))"
        )
    else:
        # No pipes: highlight only on direct block-hover (no pipe selection).
        block_stroke_expr = (
            "length(block_hover.i || []) > 0 && (datum.i === block_hover.i[0]"
            " || indexof(datum.connected_str, ',' + toString(block_hover.i[0]) + ',') >= 0)"
        )

    # ── Build the chart layers ────────────────────────────────────────────
    base = alt.Chart(df_blocks).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), title="y"),
    )

    plant_box = alt.Chart(df_plant).mark_rect(
        fill=None, stroke="#374151", strokeWidth=1.5, strokeDash=[6, 4],
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), title="y"),
        x2="x2:Q",
        y2="y2:Q",
    )

    # Block rectangles — stroke color/width are conditional on either the
    # pipe-hover or the block-hover selection. The block layer hosts the
    # block_hover param; the expression also references the pipe `hover`
    # selection (which lives on the pipe layer) so a single pipe-hover
    # also lights up its endpoint blocks.
    block_rects = base.mark_rect().encode(
        x2="x2:Q", y2="y2:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        stroke=alt.condition(
            block_stroke_expr,
            alt.value("#f59e0b"),       # orange highlight
            alt.value("#1f2937"),        # default dark border
        ),
        strokeWidth=alt.condition(
            block_stroke_expr,
            alt.value(3.5),
            alt.value(1.5),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Block"),
            alt.Tooltip("x:Q", format=".0f", title="x (lower-left)"),
            alt.Tooltip("y:Q", format=".0f", title="y (lower-left)"),
            alt.Tooltip("l:Q", format=".0f", title="length"),
            alt.Tooltip("w:Q", format=".0f", title="width"),
            alt.Tooltip("rotated:N", title="Rotated"),
            alt.Tooltip("connectivity:Q", format=".0f", title="Piping cost"),
        ],
    ).add_params(block_hover)

    block_labels = alt.Chart(df_blocks).mark_text(
        fontSize=14, fontWeight="bold", color="#0a0a4e",
    ).encode(
        x=alt.X("cx:Q", title="x"), y=alt.Y("cy:Q", title="y"), text="label:N",
        angle=alt.Angle("angle:Q", scale=None),   # raw degrees, no scaling
    )

    # Pipe overlay — two co-located layers. The visible layer is a thin
    # color rule with size proportional to pipe cost; the hit-target is a
    # transparent wider rule that captures the hover event and shows the
    # tooltip. Decoupling them lets us keep pipes visually thin while
    # giving the cursor a more forgiving hit zone.
    #
    # Layer order: box, blocks, THEN pipes, then labels. Pipes draw on top of
    # the block rectangles so the run alongside the rack (the vertical leg sits
    # exactly on the rack's edge) is visible instead of being painted over by
    # the rack. Block-id labels stay on top of everything.
    layers = [plant_box, block_rects]
    if has_pipes:
        visible_pipes = alt.Chart(df_pipes).mark_rule().encode(
            x=alt.X("x:Q", title="x"), y=alt.Y("y:Q", title="y"),
            x2="x2:Q", y2="y2:Q",
            stroke=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("c:Q",
                          scale=alt.Scale(range=[1.5, 5]),
                          legend=None),
            opacity=alt.condition(pipe_opacity_expr, alt.value(0.9), alt.value(0.25)),
        )
        pipe_hit_targets = alt.Chart(df_pipes).mark_rule(
            stroke="transparent",
            strokeWidth=8,
        ).encode(
            x=alt.X("x:Q", title="x"), y=alt.Y("y:Q", title="y"),
            x2="x2:Q", y2="y2:Q",
            tooltip=[
                alt.Tooltip("pair:N", title="Pipe"),
                alt.Tooltip("length:Q", format=".1f", title="Pipe length"),
                alt.Tooltip("c:Q", format=".0f", title="Pipe price"),
            ],
        ).add_params(hover)
        layers.append(alt.layer(visible_pipes, pipe_hit_targets))
    layers.append(block_labels)

    # "Not Solved" badge on the initialization preview so the unsolved state
    # reads at a glance. Fixed-pixel placement (top-centre) is independent of
    # the data scale; the light plate keeps the label legible over the blocks.
    if res.get("status") == "preview":
        _one = pd.DataFrame([{"_": 0}])
        # Center the badge vertically between the top of the plot and the dashed
        # plant-top line. That line sits `pad` data-units below the plot top,
        # so it's `_dash_px` pixels down; the badge goes at half that.
        _dash_px = _h_px * pad / (y_dom[1] - y_dom[0])
        _bx = _w_px / 2.0
        _by = max(14.0, _dash_px / 2.0)
        plate = alt.Chart(_one).mark_rect(
            fill="white", fillOpacity=0.85, stroke="#9ca3af",
            strokeWidth=1, cornerRadius=6,
        ).encode(
            x=alt.value(_bx - 54), x2=alt.value(_bx + 54),
            y=alt.value(_by - 14), y2=alt.value(_by + 14),
        )
        badge = alt.Chart(_one).mark_text(
            text="Not Solved", fontSize=15, fontWeight="bold", color="#b91c1c",
        ).encode(x=alt.value(_bx), y=alt.value(_by))
        layers.extend([plate, badge])

    chart = (
        alt.layer(*layers)
        .properties(
            width=_w_px, height=_h_px,
            # Disable Vega-Embed's "⋮" actions menu (Save / View Source /
            # Open in Vega Editor) — vega-embed reads embed options from the
            # spec's usermeta. The Streamlit element toolbar (fullscreen /
            # show-data) is hidden via CSS in render_optimizer.
            usermeta={"embedOptions": {"actions": False}},
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridColor="#e5e7eb", tickMinStep=1)
    )
    return chart


# ── 6. Tab renderers ─────────────────────────────────────────────────────────

def _render_metric(slot, label, value):
    """Metric-shaped block via raw HTML — small gray label, large value.
    The five top-row metrics read consistently."""
    slot.markdown(
        "<div style='margin:0.25rem 0 1.3rem 0; line-height:1.2;'>"
        "<div style='font-size:0.875rem; margin-bottom:0.6rem; "
        f"white-space:nowrap;'>{label}</div>"
        "<div style='font-size:1.8rem; font-weight:400; line-height:1.1; "
        f"white-space:nowrap;'>{value}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _keep_side_selected(key, oid):
    """on_change for the per-row N/S segmented control. A single-select
    segmented control deselects (value → None) when its active option is
    re-clicked; revert to the object's current side so every row always shows
    one arrow."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state["side"].get(oid, "N")


def _render_object_editor(ss):
    """Inline object editor (left column of the Optimizer tab): one row per
    object with Length / Width / pipe-cost-to-rack steppers. Object 1 is the
    rack — fixed in the list, no pipe-cost cell, not deletable. Add / Reset /
    Randomize below."""
    st.markdown(f"#### Objects (max {MAX_OBJECTS})")

    ver = ss["_obj_ver"]
    # Original five columns at their original widths, plus a slim arrow column
    # for the north/south tie-in. The editor pane is a touch wider (see
    # render_optimizer) so the number columns stay wide enough to keep their
    # +/- steppers; the fields themselves keep the original 6.5rem cap.
    # Delete column sized to hug the trash button (its content width), so the
    # button's right edge sits at the pane edge; the freed width goes to the
    # N/S arrows column. Total stays 5.2 so the Connections row below, which
    # shares these boundaries, keeps its column alignment.
    cols_spec = [0.5, 1.2, 1.2, 1.2, 0.5, 0.35]

    header = st.columns(cols_spec)
    header[1].markdown("**Length**")
    header[2].markdown("**Width**")
    header[3].markdown("**Pipe price**")
    header[4].markdown("**Pipe to**")

    objs = ss["objs"]
    n = len(objs)
    changed = False
    # Fixed slot count (constant element count avoids the delete ghost-row
    # flash — same reasoning as strip-packing / circle-packing).
    for slot in range(MAX_OBJECTS):
        if slot >= n:
            st.empty()
            continue
        oid = objs[slot]
        idx = slot + 1
        is_rack = (slot == 0)
        color = _PALETTE[(idx - 1) % len(_PALETTE)]
        c = st.columns(cols_spec, vertical_alignment="center")
        # Colored badge: the rack reads "rack" (a wider pill); the others are
        # numbered from 1. Color keys off the block index so the badge always
        # matches that object's fill in the layout — including the rack, which
        # is drawn white (so it needs dark text and a border to stay legible).
        _badge_w = "padding:0 0.45rem;" if is_rack else "width:1.6rem;"
        _badge_bg = "#ffffff" if is_rack else color
        _badge_fg = "#1f2937" if is_rack else "#fff"
        _badge_bd = "border:1px solid #cbd5e1;" if is_rack else ""
        c[0].markdown(
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;{_badge_w}height:1.6rem;'
            f'border-radius:0.3rem;background:{_badge_bg};color:{_badge_fg};'
            f'{_badge_bd}font-weight:700;font-size:0.85rem;white-space:nowrap;">'
            f'{_block_label(idx)}</div>',
            unsafe_allow_html=True,
        )
        new_l = c[1].number_input(
            "Length", min_value=DIM_MIN, max_value=DIM_MAX, step=1,
            value=int(ss["length"][oid]), key=f"len_{oid}_{ver}",
            label_visibility="collapsed",
        )
        new_w = c[2].number_input(
            "Width", min_value=DIM_MIN, max_value=DIM_MAX, step=1,
            value=int(ss["width"][oid]), key=f"wid_{oid}_{ver}",
            label_visibility="collapsed",
        )
        new_c = ss["cost"].get(oid, 0)
        new_side = ss["side"].get(oid, "N")
        if not is_rack:               # rack has no pipe-cost cell (left blank)
            new_c = c[3].number_input(
                "Pipe cost", min_value=COST_MIN, max_value=COST_MAX, step=1,
                value=int(ss["cost"][oid]), key=f"cost_{oid}_{ver}",
                label_visibility="collapsed",
            )
            new_side = c[4].segmented_control(
                "Pipe to", options=["N", "S"],
                format_func=lambda s: "↑" if s == "N" else "↓",
                default=ss["side"].get(oid, "N"),
                key=f"side_{oid}_{ver}", label_visibility="collapsed",
                on_change=_keep_side_selected, args=(f"side_{oid}_{ver}", oid),
            )
            if new_side is None:      # deselect guard (also handled in on_change)
                new_side = ss["side"].get(oid, "N")
        if not is_rack and n > MIN_OBJECTS:
            c[5].button("🗑", key=f"del_{oid}_{ver}",
                        on_click=_delete_object, args=(oid,))
        if (new_l != ss["length"][oid] or new_w != ss["width"][oid]
                or (not is_rack and new_c != ss["cost"][oid])
                or (not is_rack and new_side != ss["side"].get(oid))):
            ss["length"] = {**ss["length"], oid: new_l}
            ss["width"] = {**ss["width"], oid: new_w}
            if not is_rack:
                ss["cost"] = {**ss["cost"], oid: new_c}
                ss["side"] = {**ss["side"], oid: new_side}
            changed = True

    if changed:
        ss.pop("res", None)
        st.rerun()

    # Unit-to-unit connections: how many close-coupled pairs are active, and
    # their pipe-cost multiplier relative to the rack tie-in range. Same
    # stepper widgets as the object dimensions; ver-suffixed keys so Reset /
    # Randomize re-seed the displayed values.
    max_pairs = len(ss["all_pairs"])
    # First three columns match the editor spec exactly (badge / Length /
    # Width) so the label and first stepper align with the table. The tail
    # deviates: even merged, the Pipe-to+delete widths (~115px) sit below the
    # width where number inputs hide their +/- steppers, so the Cost × label
    # gives up width to its stepper instead.
    pcols = st.columns([0.5, 1.2, 1.2, 0.85, 1.2],
                       vertical_alignment="center")
    pcols[1].container(key="conn_label").markdown(
        "**Connections**",
        help="Number of direct unit-to-unit pipes (close-coupled equipment "
             "pairs). Pairs are rolled by Randomize; raising the count "
             "activates more of the rolled pairs.",
    )
    n_pairs = pcols[2].number_input(
        "Connections", min_value=0, max_value=max(max_pairs, 0), step=1,
        value=min(int(ss["n_pairs"]), max_pairs), key=f"npairs_{ver}",
        label_visibility="collapsed",
    )
    pcols[3].container(key="pair_cost_label").markdown(
        "**Cost**",
        help="Pipe cost per unit distance for every unit-to-unit connection.",
    )
    pair_weight = pcols[4].number_input(
        "Cost", min_value=PAIR_WEIGHT_MIN, max_value=PAIR_WEIGHT_MAX, step=1,
        value=int(ss["pair_weight"]), key=f"pweight_{ver}",
        label_visibility="collapsed",
    )
    if n_pairs != ss["n_pairs"] or pair_weight != ss["pair_weight"]:
        ss["n_pairs"], ss["pair_weight"] = int(n_pairs), int(pair_weight)
        ss.pop("res", None)
        st.rerun()

    bcols = st.columns(3)
    if bcols[0].button("➕ Add", key="add_obj",
                       disabled=n >= MAX_OBJECTS, use_container_width=True):
        add_object()
        st.rerun()
    if bcols[1].button("↺ Reset", key="reset_obj", use_container_width=True):
        ss["_pending_reset"] = True
        st.rerun()
    if bcols[2].button("🎲 Randomize", key="rand_obj",
                       use_container_width=True):
        ss["_pending_random"] = True
        st.rerun()


def _preview_res(ss):
    """Naive 'initialized' layout for the unsolved view, consistent with the
    rack-spans-the-plant constraint: the rack sits at the origin spanning
    the plant length (vertical y), with the other objects column-packed to
    its right within that length. Costs are zeroed so no pipes draw. Shaped
    like a solve result so build_layout_chart renders it directly."""
    objs = ss["objs"]
    n = len(objs)
    gap = 1.0
    rl = float(ss["length"][objs[0]])              # rack length (along y) = plant length
    rw = float(ss["width"][objs[0]])               # rack width (along x), thin
    # Rack on the left, spanning the length (y); objects column-packed to its
    # right, stacking in y within the rack length and wrapping to a new column.
    blocks = [{"i": 1, "x": 0.0, "y": 0.0, "l": rl, "w": rw, "rotated": False}]
    x = rw + gap
    y = 0.0
    col_w = 0.0
    for p in range(2, n + 1):
        oid = objs[p - 1]
        lp, wp = float(ss["length"][oid]), float(ss["width"][oid])
        if y > 0.0 and y + lp > rl + 1e-9:          # column full → next column
            y = 0.0
            x += col_w + gap
            col_w = 0.0
        blocks.append({"i": p, "x": x, "y": y, "l": lp, "w": wp,
                       "rotated": False})
        y += lp + gap
        col_w = max(col_w, wp)
    pairs = [{"i": i, "j": j, "c": 0.0}
             for i in range(1, n + 1) for j in range(1, i)]
    return {"status": "preview", "blocks": blocks, "pairs": pairs,
            "plant": (rl, x + col_w)}


def _clear_solution():
    """Drop the stored solve so the view falls back to the initialization
    preview. Wired as the on_change for the top controls (min distance /
    rotation / time limit): changing an option invalidates the displayed
    layout, which was solved under the previous settings."""
    st.session_state.pop("res", None)


def render_optimizer(ss):
    """Main tab: object editor on the left, layout + inline controls and
    metrics on the right."""
    # Editor styling matched to strip-packing / circle-packing: tighter row
    # spacing + compact, right-aligned number fields, and click-only steppers
    # (pointer-events:none blocks click-to-type on the <input> while the +/-
    # buttons, separate elements, stay live).
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"]
            [data-testid="stHorizontalBlock"] {
            margin-bottom: -0.75rem;
            gap: 0.35rem !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0.25rem !important;
        }
        div[role="radiogroup"] {
            gap: 0.4rem !important;
        }
        div[role="radiogroup"] label {
            margin-right: 0 !important;
        }
        /* Compact the per-row N/S segmented control so its two arrow segments
           sit side by side in the slim arrow column instead of wrapping. */
        [data-testid="stButtonGroup"] { gap: 0.15rem !important; }
        [data-testid="stButtonGroup"] button {
            padding: 0.2rem 0.3rem !important;
            min-width: 0 !important;
        }
        [data-testid="stNumberInputContainer"] input {
            padding-top: 0.25rem; padding-bottom: 0.25rem;
            text-align: right; padding-right: 0.4rem;
            pointer-events: none;
            user-select: none;
            caret-color: transparent;
        }
        /* Cap the field width so the value sits next to the +/- buttons
           instead of stretching across the column — compact rows, while
           still wide enough that the steppers stay visible. */
        [data-testid="stNumberInputContainer"] {
            max-width: 6.5rem;
        }
        /* Cost × stepper: right-aligned in its column, ending flush with the
           pane edge. */
        [class*="st-key-pweight"] [data-testid="stNumberInputContainer"] {
            margin-left: auto;
        }
        /* Right-align the Cost × label (text + help icon) in its column so
           its ? sits as close to its stepper as the Connections label's ?
           sits to its own; the Connections label keeps its natural left
           alignment under the Length column. The keyed container is a
           column-flex block, so push its children to the right edge; the
           inner rule right-aligns the text within the markdown block. */
        [class*="st-key-pair_cost_label"] {
            align-items: flex-end;
        }
        [class*="st-key-pair_cost_label"] [data-testid="stMarkdownContainer"] {
            display: flex;
            justify-content: flex-end;
            text-align: right;
        }
        /* Drop the on-hover chart chrome for a clean presentation view:
           Streamlit's element toolbar (fullscreen / show-data) and any
           remnant of Vega-Embed's "⋮" actions menu (also disabled via the
           chart's usermeta embedOptions in build_layout_chart). */
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        .vega-embed details,
        .vega-embed .vega-actions {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    editor_col, viz_col = st.columns([4, 8])

    with editor_col:
        _render_object_editor(ss)

    with viz_col:
        _viz_panel(ss)


@st.fragment
def _viz_panel(ss):
    """Result panel — controls, layout selector, chart, metrics — isolated in a
    fragment. The Solve/option controls and the layout selector live here, so
    interacting with any of them re-renders only this panel, not the object
    editor or the Formulation/Logs tabs. All pooled layouts are already computed
    and cached in session_state, so switching the selector just redraws the
    chart — no re-solve and no full-app rerun."""
    # Everything in one row (strip-packing layout): Solve / Min distance /
    # Rotation / Time limit, then the five metric slots. The layout paints
    # below through a placeholder so the controls commit before we draw.
    top = st.columns(
        [0.7, 1.5, 1.2, 1.9, 1.0, 1.0, 1.1, 0.9, 1.0],
        vertical_alignment="bottom",
    )
    with top[0]:
        solve_clicked = st.button("Solve", type="primary",
                                  use_container_width=True)
    with top[1]:
        d_min = st.number_input(
            "Min distance", min_value=D_MIN, max_value=D_MAX, step=1,
            value=int(ss["d_min"]), key="dmin_input",
            on_change=_clear_solution,
        )
    with top[2]:
        rotate = st.checkbox("Rotation", value=ss["rotate"],
                             key="rotate_box", on_change=_clear_solution)
    with top[3]:
        time_label = st.radio(
            "Time limit", options=list(_TIME_LIMITS.keys()), index=0,
            horizontal=True, key="time_radio",
            on_change=_clear_solution,
        )
    # Metric columns (4-8) are filled at the end, once the selected solution is
    # known. They render straight into the columns (not via st.empty
    # placeholders) so a toggle change updates the numbers in place instead of
    # blanking and repainting them.
    # Spacer so the plot sits a little below the controls/metrics row.
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    # Layout selector — filled later, only when a solve returned more than
    # one distinct layout. Sits directly above the plot, below the controls.
    toggle_slot = st.empty()
    viz_slot = st.empty()
    spinner_slot = st.empty()

    ss["d_min"] = int(d_min)
    ss["rotate"] = bool(rotate)
    time_limit = _TIME_LIMITS[time_label]

    if solve_clicked:
        n, l0, w0, cmat = _objs_to_inputs(ss)
        with spinner_slot.container():
            with st.spinner(
                f"Running Gurobi (Big-M, time limit {time_limit}s)..."
            ):
                ss["res"] = solve(n, l0, w0, cmat, float(ss["d_min"]),
                                  ss["rotate"], 1, time_limit)
        spinner_slot.empty()
        ss["sol_sel"] = 0   # reset the layout selector to the best solution
        # Full-app rerun (not just this fragment): the Logs tab and header
        # metrics live outside the fragment and would otherwise keep showing
        # the pre-solve state.
        st.rerun(scope="app")

    res = ss.get("res")

    # Distinct near-optimal layouts from the solution pool, when we have a solve.
    sols = (res.get("solutions")
            if res and res.get("status") in ("optimal", "incumbent") else None)

    # Layout selector: a toggle directly above the plot, below the time-limit
    # row. Shown only when the pool returned more than one distinct layout.
    # Switching it re-renders the chosen layout — it never re-solves.
    sel_idx = 0
    if sols and len(sols) > 1:
        with toggle_slot.container():
            # Center the selector on the Time limit control (so #3 lands under
            # "30 s") while keeping it wide enough that the five buttons stay on
            # one row. A 3-column split whose middle is centered ≈ under Time
            # limit; the selector stretches to fill that middle column.
            _pad_l, _mid, _pad_r = st.columns([2.2, 3.0, 4.1])
            with _mid:
                choice = st.segmented_control(
                    "Solution",
                    options=list(range(len(sols))),
                    format_func=lambda k: f"#{k + 1}",
                    key="sol_sel",
                    width="stretch",
                )
        if choice is not None:
            sel_idx = max(0, min(int(choice), len(sols) - 1))
    else:
        # No selector to show (preview or a single layout). Reserve its vertical
        # space so the plot stays at the same height as when the selector is
        # present — no jump between the unsolved preview and a solved result.
        toggle_slot.markdown(
            "<div style='height:3.25rem'></div>", unsafe_allow_html=True
        )

    # Rendered view = the selected pooled layout over the solve-level fields
    # (status / gap / elapsed). Falls back to res for the preview/error paths.
    sel = {**res, **sols[sel_idx]} if sols else res

    with viz_slot.container():
        if res is None:
            st.altair_chart(build_layout_chart(_preview_res(ss)),
                            use_container_width=False)
        elif res["status"] in ("optimal", "incumbent"):
            st.altair_chart(build_layout_chart(sel), use_container_width=False)
        elif res["status"] == "no_feasible":
            st.error("Hit the time limit before finding any feasible layout. "
                     "Try fewer objects, a smaller minimum distance, or a "
                     "longer time limit.")
        elif res["status"] == "infeasible":
            st.error("Infeasible. An object may be longer than the rack "
                     "(every object must fit within the rack's length), or the "
                     "minimum separation is too large. Try shortening objects, "
                     "lengthening the rack, or reducing the distance.")
        elif res["status"] == "license_busy":
            st.error(res.get("message", "Gurobi license busy. Try again."))
        elif res["status"] == "solver_missing":
            st.error(res.get("message", "Solver not available."))
        else:
            st.warning(f"Solver returned: {res['status']}")

    has = res is not None and res["status"] in ("optimal", "incumbent")
    if has:
        w_f = sel["plant"][1]
        # Center-to-center distances put objective values on the half-integer
        # grid, so show one decimal place.
        objv = f"{sel['obj']:.1f}"
        plantW = f"{w_f:.1f}"
        pipe = f"{sel['pipe_cost']:.1f}"
        # Gap of the SELECTED layout against the solve's dual bound, so
        # switching pool solutions shows how far each one is from the proven
        # bound (0% only for a proven-optimal layout).
        lb = res.get("lower_bound")
        if lb is not None and sel["obj"] > 0:
            sel_gap = max(0.0, (sel["obj"] - lb) / abs(sel["obj"]))
            gap = "0%" if sel_gap < 5e-4 else f"{sel_gap * 100:.1f}%"
        else:
            gap = "—"
        elapsed = res.get("elapsed")
        tstr = f"{elapsed:.1f}s" if elapsed is not None else "—"
    else:
        objv = plantW = pipe = gap = tstr = "—"

    _render_metric(top[4], "Objective", objv)
    _render_metric(top[5], "Plant width", plantW)
    _render_metric(top[6], "Total piping cost", pipe)
    _render_metric(top[7], "Gap", gap)
    _render_metric(top[8], "Total time", tstr)


def render_formulation():
    st.markdown(r"""
### Layout Formulation

Place $n$ rectangular objects so that the plant's bounding-box
dimensions plus the cost-weighted Manhattan pipe distances are minimized.
Each object pipes to its assigned north or south end of the rack, and
selected pairs of objects additionally carry a direct unit-to-unit
connection. Width is the horizontal ($x$) axis, length the vertical ($y$):

$$\min \; \lambda \,(l_f + w_f) + \sum_{i,j \in N,\; j<i} c_{ij} \big( dx_{ij} + dy_{ij} \big)$$

where $\lambda$ weights the plant size against the piping cost (default 1).

subject to the plant containing every object (length along $y$, width
along $x$):

$$l_f \ge y_i + l_i, \quad w_f \ge x_i + w_i \quad \forall \, i \in N$$

The pipe **rack** (object 1) spans the plant length. It is pinned at the
origin with the plant length fixed to the rack's, so every other object
fits within $[0, l_1]$ and sits to either side of the rack:

$$y_1 = 0, \qquad l_f = l_1$$

Each non-rack object is assigned (as fixed instance data) to tie into the
**north** end ($y = l_f$) or the **south** end ($y = 0$) of the rack, and its
piping cost is the Manhattan distance to that end. This is modeled with two
zero-length tie-in objects pinned to the rack's top and bottom, sharing the
rack's $x$-span, at zero clearance, so the same distance and non-overlap
machinery applies. The $c_{ij}$ above are nonzero between an object and its
assigned end, and between the unit pairs given a direct connection (the
**Connections** control), whose pipes carry the **Cost** per unit distance.

Distances are rectilinear **center-to-center** — the convention of the
process-plant layout literature [1] — defined by always-on constraints
(outside the disjunction), one lower bound per axis direction:

$$dx_{ij} \ge \big(x_i + \tfrac{w_i}{2}\big) - \big(x_j + \tfrac{w_j}{2}\big), \quad dx_{ij} \ge \big(x_j + \tfrac{w_j}{2}\big) - \big(x_i + \tfrac{w_i}{2}\big)$$

$$dy_{ij} \ge \big(y_i + \tfrac{l_i}{2}\big) - \big(y_j + \tfrac{l_j}{2}\big), \quad dy_{ij} \ge \big(y_j + \tfrac{l_j}{2}\big) - \big(y_i + \tfrac{l_i}{2}\big)$$

All decision variables are nonnegative ($x_i, y_i, l_i, w_i, l_f, w_f,
dx_{ij}, dy_{ij} \ge 0$). Positions also have worst-case upper bounds
$x_i, y_i \le \mathrm{UB} = \sum_i \max(l_i, w_i)$. The model has the
**non-overlap disjunction** (one of four geometric arrangements per pair)
and the **rotation disjunction** (default vs. 90° rotated, when rotation is
enabled; the rack stays fixed).

### Disjunctions

For every pair $(i, j)$ with $j < i$, one of the four separations must
hold, with the minimum clearance $d_{ij}$ built in:

$$
\begin{bmatrix} Y_{ij}^1 \\ x_i + w_i + d_{ij} \le x_j \\ y_i + l_i \ge y_j - d_{ij} \\ y_j + l_j \ge y_i - d_{ij} \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^2 \\ x_j + w_j + d_{ij} \le x_i \\ y_i + l_i \ge y_j - d_{ij} \\ y_j + l_j \ge y_i - d_{ij} \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^3 \\ y_i + l_i + d_{ij} \le y_j \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^4 \\ y_j + l_j + d_{ij} \le y_i \end{bmatrix}
$$

($k=1$ left, $2$ right, $3$ below, $4$ above.)

The left/right disjuncts ($Y^1, Y^2$) carry two extra inequalities forcing
the blocks to overlap vertically within $d_{ij}$ — a **degeneracy breaking
constraint** (a $d$-aware variant of Trespalacios & Grossmann [5]). Without
it, a pair separated on *both* axes could be encoded as left/right **or**
above/below, so one physical layout has several encodings that
branch-and-bound re-explores and that fill the solution pool with
duplicates. The offset is $-d_{ij}$ rather than the tighter $-(d_{ij}-1)$:
the tighter form is only optimum-preserving when an integer-optimal layout
is guaranteed, which center-to-center distances do not provide for
odd-dimensioned objects. The $-d_{ij}$ form is optimum-preserving for
arbitrary continuous coordinates.

When rotation is enabled, each block additionally chooses orientation:

$$
\begin{bmatrix} Y_i^5 \\ l_i = l_i^0 \\ w_i = w_i^0 \end{bmatrix}
\;\lor\;
\begin{bmatrix} Y_i^6 \\ l_i = w_i^0 \\ w_i = l_i^0 \end{bmatrix}
$$
""")

    st.markdown(r"""
### Symmetry breaking

Only the left/right mirror is a symmetry here: flipping top/bottom would swap
each object's north/south tie-in, so it is not one. We anchor block 1 (the
rack) left of block 2's center:

$$x_1 + w_1/2 \le x_2 + w_2/2$$

This removes the left/right reflective pair and tightens the LP relaxation. A
vertical anchor is deliberately omitted: under the fixed north/south
assignment it would wrongly exclude valid layouts.

### Solution method

We reformulate the GDP into a MILP with the **Big-M** transformation (one
indicator per disjunct with a big constant), then solve it with **Gurobi**.
Big-M keeps the model compact. The Hull (convex-hull) transformation was
benchmarked too, but it disaggregates every variable per disjunct, inflating
the model for a tighter relaxation that doesn't pay off here.

For larger instances (n > 8) the MIP can exceed the wall-clock time limit.
The app then loads the **best feasible incumbent** found before the cutoff
and reports the optimality gap. Try smaller min-separation distances if the
solver returns infeasible.

Many near-optimal layouts usually exist, and an engineer may prefer one over
another for reasons the model doesn't capture. After solving, the app asks
Gurobi's **solution pool** for its best feasible layouts and offers several
as selectable alternatives. The degeneracy breaking constraint above is what
keeps these genuinely distinct *layouts* rather than duplicate encodings of
one. From the pool the app greedily selects a spread of layouts: each new
one maximizes, against those already chosen, the number of block pairs that
take a different spatial relation. The lowest-cost layout stays the default.

### References

[1] L. G. Papageorgiou and G. E. Rotstein, "Continuous-Domain
Mathematical Models for Optimal Process Plant Layout," *Industrial &
Engineering Chemistry Research*, vol. 37, no. 9, pp. 3631–3639, 1998.
[ACS](https://pubs.acs.org/doi/10.1021/ie980146v)

[2] J. Westerlund and L. G. Papageorgiou, "Improved Performance in
Process Plant Layout Problems Using Symmetry-Breaking Constraints,"
*Proc. FOCAPD 2004 (Foundations of Computer-Aided Process Design)*,
2004.
[PDF](https://skoge.folk.ntnu.no/prost/proceedings/focapd_2004/pdffiles/papers/075_46.pdf)

[3] N. W. Sawaya and I. E. Grossmann, "A Cutting Plane Method for
Solving Linear Generalized Disjunctive Programming Problems,"
*Computers & Chemical Engineering*, vol. 29, no. 9, pp. 1891–1913, 2005.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0098135405000992)

[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird,
B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff,
*Pyomo: Optimization Modeling in Python*, 3rd ed. Cham: Springer,
2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)

[5] F. Trespalacios and I. E. Grossmann, "Symmetry breaking for
generalized disjunctive programming formulation of the strip packing
problem," *Annals of Operations Research*, vol. 258, pp. 747–759, 2017.
[Springer](https://link.springer.com/article/10.1007/s10479-016-2112-9)
""")


def render_logs(res):
    if res is None:
        st.info("Run a solve to see Gurobi's output.")
        return
    log = res.get("log", "")
    if log.strip():
        st.code(log, language=None)
    else:
        st.info("No solver log captured. The solver may have returned before "
                "writing to stdout.")


# ── 7. Main layout ────────────────────────────────────────────────────────────

_init_state()
ss = st.session_state

# ---- Title ----
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Plant Layout GDP Optimizer "
    "<a href='https://github.com/devin-griff/plant_layout' target='_blank' "
    "title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; margin: 0 0.35rem 0 0.1rem; "
    "color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://www.gurobi.com' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Gurobi</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "A pipe **rack** spans the plant; place unit ops on either side of "
        "it to minimize piping cost to the pipe rack as well as unit-to-unit "
        "connections. Edit the objects, set "
        "the options, and click **Solve**. If the time limit is reached, the "
        "best incumbent solution will be returned, as well as up to four "
        "other maximally different solutions from Gurobi's solution pool. "
        "For reference, the optimal objective for the default instance is "
        "255.5."
    )

# ---- Tabs ----
tab_opt, tab_form, tab_logs = st.tabs(
    ["▶  Optimizer", "📐  Formulation", "📋  Logs"]
)

with tab_opt:
    render_optimizer(ss)

with tab_form:
    render_formulation()

with tab_logs:
    render_logs(ss.get("res"))
