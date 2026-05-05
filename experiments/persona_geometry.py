"""
Persona geometry for misaligned-organism analysis, following Lu et al.
"The Assistant Axis" (arXiv:2601.10387).

Methodological choices match the paper:
- PCA is fit on a reference set of role vectors only (not on the
  Assistant vector, not on the misaligned organisms). Centering uses
  the mean over roles. This is what `compute_pca` already gives you
  when called with `MeanScaler()`.
- The Assistant Axis is constructed as a CONTRAST VECTOR in raw
  activation space (mean Assistant minus mean role), per Section 3.1.
  This is the paper's recommended operationalization of similarity to
  Assistant, over raw PC1.
- Per-organism cosine similarity vs. the Assistant is computed in raw
  activation space (per Section 2.3.2 / Table 2), not in PC space.
- Position in persona space is computed by projecting through the
  fitted MeanScaler + PCA pipeline, mirroring how the paper projects
  the default Assistant into the role-space PCs (Section 2.3.1).
- Relative position along a PC follows the paper's
  (proj - min_role) / (max_role - min_role) convention.

Inputs throughout:
- All vectors are at a single chosen layer (the paper uses the middle
  post-MLP residual stream layer). Pass already-layer-selected (n, d)
  arrays.
- `pca` and `scaler` are the objects returned by `compute_pca` when
  fit on the role cohort.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _np(x):
    if isinstance(x, np.ndarray):
        return x
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Assistant Axis (paper Section 3.1) - contrast vector in raw activations.
# ---------------------------------------------------------------------------

def build_assistant_axis(assistant_vectors, role_vectors,
                         normalize: bool = True):
    """
    Construct the Assistant Axis as the contrast between mean Assistant
    activation and the mean role vector. Per Section 3.1, this is
    computed over fully-role-playing role vectors only - pass the same
    filtered cohort here that the paper's pipeline yields.

    Args:
        assistant_vectors: (n_a, d) raw activations from default-Assistant
            rollouts at the chosen layer (or a single (d,) mean vector).
        role_vectors: (n_r, d) raw role vectors at the chosen layer.
        normalize: if True, return a unit-norm direction (recommended for
            cosine / projection use). If False, returns the raw difference.

    Returns:
        (d,) numpy array - the Assistant Axis at this layer.
    """
    A = _np(assistant_vectors)
    R = _np(role_vectors)
    a_mean = A if A.ndim == 1 else A.mean(axis=0)
    r_mean = R.mean(axis=0)
    axis = a_mean - r_mean
    if normalize:
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis


# ---------------------------------------------------------------------------
# 2. Projection helpers - drop new vectors into the fitted persona space.
# ---------------------------------------------------------------------------

def project_into_persona_space(vectors, pca, scaler=None, k=None):
    """
    Project raw activation vectors into the fitted role-space PCs.

    This mirrors how the paper projects the default Assistant vector
    (Section 2.3.1) and traits (Figure 3) into the role-defined persona
    space.
    """
    X = _np(vectors)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[None, :]
    if scaler is not None:
        X = scaler.transform(X)
    coords = pca.transform(_np(X))
    if k is not None:
        coords = coords[:, :k]
    return coords[0] if squeeze else coords


def anchor_pc1_to_assistant(assistant_pc, organisms_pc=None, all_role_pc=None):
    """
    Flip PC1's sign so that the default Assistant has a positive PC1
    coordinate. The paper plots PC1 with the Assistant at the positive
    end (Figures 1, 2, 16, 17). Apply the same sign vector to anything
    else you later project.

    Returns (assistant_pc_anchored, organisms_pc_anchored,
             role_pc_anchored, sign_vector).
    """
    a = _np(assistant_pc).copy()
    sign = 1.0 if a[0] >= 0 else -1.0
    signs = np.ones_like(a)
    signs[0] = sign
    return (
        a * signs,
        None if organisms_pc is None else _np(organisms_pc) * signs,
        None if all_role_pc is None else _np(all_role_pc) * signs,
        signs,
    )


# ---------------------------------------------------------------------------
# 3. Similarity report - paper-faithful + a couple of useful extensions.
# ---------------------------------------------------------------------------

def _cos(a, b, eps=1e-12):
    a, b = _np(a), _np(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def organism_similarity_report(
    organism_vectors,            # (n_o, d) raw misaligned organism role vectors
    assistant_vector,            # (d,) mean default-Assistant activation
    assistant_axis,              # (d,) from build_assistant_axis(...)
    role_vectors,                # (n_r, d) reference role cohort
    pca,                         # fitted PCA (on role_vectors via MeanScaler)
    scaler,                      # fitted MeanScaler from compute_pca
    k: int = 3,                  # leading PCs to use for subspace metrics
    organism_names=None,
):
    """
    Compute the geometry of each misaligned organism relative to the
    base model's default persona, in three complementary views:

      RAW SPACE - cosine with Assistant, projection onto Assistant Axis.
                  Matches Section 2.3.2 / Table 2 of the paper.
      PC SPACE  - coordinates in the fitted role-space PCs, sign-anchored
                  so the Assistant has positive PC1; relative position on
                  each PC (Section 2.3.1's 0-1 metric).
      SUBSPACE  - cosine and angle between organism and Assistant inside
                  the top-k PC subspace (Euclidean distance also given).

    Returns a dict of arrays of length n_o.
    """
    O = _np(organism_vectors)
    if O.ndim == 1:
        O = O[None, :]
    a = _np(assistant_vector).reshape(-1)
    v = _np(assistant_axis).reshape(-1)
    v_hat = v / (np.linalg.norm(v) + 1e-12)
    R = _np(role_vectors)

    # ---- RAW SPACE ------------------------------------------------------
    cos_assistant_raw = np.array([_cos(o, a) for o in O])
    proj_on_axis = O @ v_hat                       # signed scalar projection
    proj_assistant = float(a @ v_hat)              # for context / sign check
    proj_role_mean = float(R.mean(axis=0) @ v_hat)

    # ---- PC SPACE -------------------------------------------------------
    a_pc_full = project_into_persona_space(a, pca, scaler)
    O_pc_full = project_into_persona_space(O, pca, scaler)
    R_pc_full = project_into_persona_space(R, pca, scaler)
    a_pc, O_pc, R_pc, signs = anchor_pc1_to_assistant(
        a_pc_full, O_pc_full, R_pc_full
    )

    # Relative position on each of the top-k PCs (paper Section 2.3.1)
    rel_pos_assistant = np.zeros(k)
    rel_pos_organisms = np.zeros((O_pc.shape[0], k))
    for j in range(k):
        lo, hi = R_pc[:, j].min(), R_pc[:, j].max()
        span = (hi - lo) + 1e-12
        rel_pos_assistant[j] = (a_pc[j] - lo) / span
        rel_pos_organisms[:, j] = (O_pc[:, j] - lo) / span

    # ---- SUBSPACE (top-k) ----------------------------------------------
    a_pc_k = a_pc[:k]
    O_pc_k = O_pc[:, :k]
    cos_assistant_sub = np.array([_cos(o, a_pc_k) for o in O_pc_k])
    angle_deg_sub = np.degrees(np.arccos(np.clip(cos_assistant_sub, -1, 1)))
    dist_sub = np.linalg.norm(O_pc_k - a_pc_k[None, :], axis=1)

    return {
        "names": organism_names,

        # Raw activation space - paper-faithful similarity
        "cos_assistant_raw": cos_assistant_raw,
        "proj_on_assistant_axis": proj_on_axis,
        "proj_assistant_on_axis_reference": proj_assistant,
        "proj_role_mean_on_axis_reference": proj_role_mean,

        # PC space (sign-anchored so Assistant has positive PC1)
        "assistant_pc": a_pc,
        "organisms_pc": O_pc,
        "roles_pc": R_pc,
        "pc_signs": signs,

        # Relative position on top-k PCs (paper Section 2.3.1)
        "assistant_relative_position_top_k": rel_pos_assistant,
        "organisms_relative_position_top_k": rel_pos_organisms,

        # Top-k PC subspace geometry
        "cos_assistant_subspace": cos_assistant_sub,
        "angle_deg_subspace": angle_deg_sub,
        "euclid_dist_subspace": dist_sub,

        "k": k,
    }


# ---------------------------------------------------------------------------
# 4. Rotation operations - geometric questions inside persona space.
# ---------------------------------------------------------------------------

def slerp(v1, v2, t):
    """
    Spherical linear interpolation. Returns a vector at fractional
    geodesic angle t between v1 and v2 (t=0 -> v1, t=1 -> v2). Norms
    are linearly blended. Works in raw activation space or PC space.
    """
    v1 = _np(v1).astype(float)
    v2 = _np(v2).astype(float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    u1, u2 = v1 / (n1 + 1e-12), v2 / (n2 + 1e-12)
    cos_th = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    th = np.arccos(cos_th)
    if th < 1e-6:
        return (1 - t) * v1 + t * v2
    direction = (np.sin((1 - t) * th) * u1 + np.sin(t * th) * u2) / np.sin(th)
    return direction * ((1 - t) * n1 + t * n2)


def rotate_in_plane(v, axis_from, axis_to, angle_rad):
    """
    Rotate v by angle_rad in the 2D plane spanned by (axis_from, axis_to).
    Components of v orthogonal to that plane are preserved. Useful for
    'dial the persona by phi degrees from Assistant toward an organism.'
    """
    v = _np(v).astype(float)
    axis_from = _np(axis_from).astype(float)
    axis_to = _np(axis_to).astype(float)
    e1 = axis_from / (np.linalg.norm(axis_from) + 1e-12)
    perp = axis_to - (axis_to @ e1) * e1
    if np.linalg.norm(perp) < 1e-10:
        raise ValueError("axis_from and axis_to are colinear; rotation plane undefined.")
    e2 = perp / np.linalg.norm(perp)

    a, b = v @ e1, v @ e2
    out_of_plane = v - a * e1 - b * e2
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return (c * a - s * b) * e1 + (s * a + c * b) * e2 + out_of_plane


def organism_rotation_analysis(
    organism_pc,                 # (k,) PC coords for one organism (sign-anchored)
    assistant_pc,                # (k,) PC coords for Assistant   (sign-anchored)
    n_steps: int = 11,
):
    """
    Geometric rotation analysis between Assistant and one organism in
    the top-k PC subspace (using the sign-anchored coordinates from
    organism_similarity_report).

    Returns:
        - geodesic angle (rad, deg) on the unit sphere of the subspace
        - slerp trajectory: n_steps points from Assistant to organism
        - per-PC squared-difference share (which PC drives the rotation)
    """
    a = _np(assistant_pc).astype(float)
    o = _np(organism_pc).astype(float)

    cos_th = _cos(a, o)
    theta_rad = float(np.arccos(np.clip(cos_th, -1.0, 1.0)))

    ts = np.linspace(0.0, 1.0, n_steps)
    trajectory = np.stack([slerp(a, o, t) for t in ts], axis=0)

    diff = o - a
    contribution = diff**2 / (np.sum(diff**2) + 1e-12)

    return {
        "cos": cos_th,
        "angle_rad": theta_rad,
        "angle_deg": float(np.degrees(theta_rad)),
        "slerp_t": ts,
        "slerp_trajectory_pc": trajectory,
        "per_pc_squared_share": contribution,
    }


# ---------------------------------------------------------------------------
# 5. Optional: lift PC coords back to raw activations for steering.
# ---------------------------------------------------------------------------

def lift_pc_to_raw(pc_coords, pca, scaler=None):
    """
    Convert PC-space coordinates back to raw activation space. Pads with
    zeros if you kept fewer than all PCs. Adds back the scaler's mean if
    a MeanScaler was used. Useful if you want to take a slerp trajectory
    in PC space and use the resulting raw vectors for steering.
    """
    X = _np(pc_coords)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[None, :]

    n_full = pca.components_.shape[0]
    if X.shape[1] < n_full:
        X = np.pad(X, ((0, 0), (0, n_full - X.shape[1])))

    raw = pca.inverse_transform(X)  # adds back pca.mean_

    if scaler is not None and getattr(scaler, "mean", None) is not None:
        raw = raw + scaler.mean

    return raw[0] if squeeze else raw