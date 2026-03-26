# Convert Unity simulator data to PoseDiffusion gt_cameras.npz format
#
# Output format (PyTorch3D PerspectiveCameras, row-vector convention):
#   gtR  : (N, 3, 3)  R_wc  — camera-to-world rotation (= R_cw^T)
#   gtT  : (N, 3)     t_cw  — world-to-camera translation (= -R_wc^T @ T_wc)
#   gtFL : (N, 2)     [fx, fy] in NDC units = fx_pixel / (min(W,H) / 2)

import re
import shutil
import sys
from pathlib import Path
import numpy as np


# ── coordinate conversion (same as simulator_pose.py, verified against README) ──

def quat_xyzw_to_rotmat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0-(yy+zz), xy-wz,       xz+wy      ],
        [xy+wz,       1.0-(xx+zz), yz-wx      ],
        [xz-wy,       yz+wx,       1.0-(xx+yy)],
    ], dtype=np.float64)


def unity_to_Rwc_Twc(q_xyzw, t_unity):
    """Convert Unity left-handed pose to right-handed R_wc, T_wc (in mm)."""
    x, y, z, w = q_xyzw
    # negate x and z to flip from left-handed to right-handed (README formula)
    R_wc = quat_xyzw_to_rotmat(-x, y, -z, w)
    tx, ty, tz = t_unity
    T_wc = np.array([tx, -ty, tz], dtype=np.float64) * 10.0   # cm → mm
    return R_wc, T_wc


# ── file parsers ─────────────────────────────────────────────────────────────

def parse_position_file(path):
    pat = re.compile(
        r"Frame\s+(\d+)\s+Position:\s+X=([-\d.eE+]+),\s*Y=([-\d.eE+]+),\s*Z=([-\d.eE+]+)"
    )
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            out[int(m.group(1))] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return out


def parse_quat_file(path):
    pat = re.compile(
        r"Frame\s+(\d+)\s+Rotation:\s+X=([-\d.eE+]+),\s*Y=([-\d.eE+]+),\s*Z=([-\d.eE+]+),\s*W=([-\d.eE+]+)"
    )
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            out[int(m.group(1))] = np.array([float(m.group(2)), float(m.group(3)),
                                              float(m.group(4)), float(m.group(5))])
    return out


def parse_intrinsic_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    fl   = float(re.search(r"Focal Length \(mm\)\s*:\s*([-\d.eE+]+)", text).group(1))
    sw   = float(re.search(r"Sensor Size \(mm\)\s*:\s*X=\s*([-\d.eE+]+)", text).group(1))
    sh   = float(re.search(r"Sensor Size \(mm\)\s*:.*Y=\s*([-\d.eE+]+)", text).group(1))
    res  = re.search(r"images Resolution \(pixel\)\s*:\s*(\d+)\s*\*\s*(\d+)", text)
    cols, rows = int(res.group(1)), int(res.group(2))
    pp   = re.search(r"Principal point\s*:\s*X=\s*([-\d.eE+]+)\s+Y=\s*([-\d.eE+]+)", text)
    ppx, ppy = float(pp.group(1)), float(pp.group(2))
    fx_px = fl * cols / sw
    fy_px = fl * rows / sh
    return fx_px, fy_px, cols, rows


# ── main conversion ───────────────────────────────────────────────────────────

def convert(data_dir: Path, out_path: Path = None):
    data_dir = Path(data_dir)

    # auto-detect prefix from intrinsic file name
    candidates = sorted(data_dir.glob("*_Intrinsic Data.txt"))
    if not candidates:
        raise FileNotFoundError(f"No '*_Intrinsic Data.txt' found in {data_dir}")
    prefix = re.match(r"^(.*)_Intrinsic Data\.txt$", candidates[0].name).group(1)
    print(f"prefix: {prefix}")

    fx_px, fy_px, W, H = parse_intrinsic_file(candidates[0])
    print(f"intrinsics (pixels): fx={fx_px:.4f}  fy={fy_px:.4f}  res={W}x{H}")

    pos_map  = parse_position_file(data_dir / f"{prefix}_Camera Position Data.txt")
    quat_file = data_dir / f"{prefix}_Camera Quaternion Rotation Data.txt"
    if not quat_file.exists():
        quat_file = data_dir / f"{prefix}_Camera Total Rotation Data.txt"
    quat_map = parse_quat_file(quat_file)

    common_ids = sorted(set(pos_map) & set(quat_map))
    if len(common_ids) < 2:
        raise RuntimeError(f"Need ≥2 frames with both position and rotation; got {len(common_ids)}")
    print(f"frames: {len(common_ids)}  ({common_ids[0]} … {common_ids[-1]})")

    # NDC focal length: fx_ndc = fx_pixel / (min(W, H) / 2)
    short = min(W, H)
    fx_ndc = fx_px / (short / 2.0)
    fy_ndc = fy_px / (short / 2.0)
    print(f"focal length NDC: fx={fx_ndc:.6f}  fy={fy_ndc:.6f}")

    gtR_list, gtT_list, gtFL_list = [], [], []

    for fid in common_ids:
        R_wc, T_wc = unity_to_Rwc_Twc(quat_map[fid], pos_map[fid])

        # PyTorch3D convention (row-vector):
        #   x_cam = x_world @ R + T
        #   R = R_cw^T = R_wc
        #   T = t_cw   = -R_cw @ t_wc = -R_wc^T @ T_wc
        R_cw = R_wc.T
        t_cw = -R_cw @ T_wc

        gtR_list.append(R_wc.astype(np.float32))       # (3,3)
        gtT_list.append(t_cw.astype(np.float32))       # (3,)
        gtFL_list.append([fx_ndc, fy_ndc])

    gtR  = np.stack(gtR_list,  axis=0)   # (N, 3, 3)
    gtT  = np.stack(gtT_list,  axis=0)   # (N, 3)
    gtFL = np.array(gtFL_list, dtype=np.float32)  # (N, 2)

    if out_path is None:
        out_path = data_dir / "gt_cameras.npz"
    np.savez(out_path, gtR=gtR, gtT=gtT, gtFL=gtFL)
    print(f"saved → {out_path}  (gtR{gtR.shape}, gtT{gtT.shape}, gtFL{gtFL.shape})")
    return out_path


def convert_to_samples(data_dir: Path, samples_root: Path = None):
    """
    Convert a simulator sequence and set it up as a PoseDiffusion sample:
      1. Create samples/<seq_name>/ next to this script (or under samples_root)
      2. Copy all PNG images into it
      3. Generate gt_cameras.npz inside it
    """
    data_dir = Path(data_dir).resolve()
    script_dir = Path(__file__).parent

    if samples_root is None:
        samples_root = script_dir / "samples"

    out_dir = samples_root / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy images
    pngs = sorted(data_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG images found in {data_dir}")
    for p in pngs:
        shutil.copy2(p, out_dir / p.name)
    print(f"copied {len(pngs)} images → {out_dir}")

    # generate gt_cameras.npz
    convert(data_dir, out_dir / "gt_cameras.npz")


if __name__ == "__main__":
    script_dir = Path(__file__).parent

    if len(sys.argv) < 2:
        # default: process all sequences under simulator_data/
        sim_root = script_dir / "simulator_data"
        sequences = sorted(p for p in sim_root.iterdir() if p.is_dir())
        if not sequences:
            print(f"No subdirectories found in {sim_root}")
            sys.exit(1)
    else:
        sequences = [Path(sys.argv[1])]

    for seq in sequences:
        print(f"\n{'='*50}")
        print(f"Processing: {seq.name}")
        convert_to_samples(seq)
