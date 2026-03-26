import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import re
import math
import shutil
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath


def read_exr(file_path, depth_channel="R", depth_scale=5.0):
    exr_file = OpenEXR.InputFile(str(file_path))
    header = exr_file.header()
    dw = header["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    if depth_channel in header["channels"]:
        channel_data = exr_file.channel(depth_channel, pt)
        depth = np.frombuffer(channel_data, dtype=np.float32).reshape(size[1], size[0]).copy()
        depth *= float(depth_scale)
        return depth

    raise ValueError(f"Channel '{depth_channel}' not found in EXR file.")


def write_exr(file_path, depth_map):
    depth_map = depth_map.astype(np.float32)
    h, w = depth_map.shape
    header = OpenEXR.Header(w, h)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pt),
        "G": Imath.Channel(pt),
        "B": Imath.Channel(pt),
    }
    exr = OpenEXR.OutputFile(str(file_path), header)
    ch = depth_map.tobytes()
    exr.writePixels({"R": ch, "G": ch, "B": ch})
    exr.close()


def quat_xyzw_to_rotmat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def rotmat_to_quat_xyzw(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q


def parse_position_file(path):
    p = re.compile(
        r"Frame\s+(\d+)\s+Position:\s+X=([-\d.eE+]+),\s*Y=([-\d.eE+]+),\s*Z=([-\d.eE+]+)"
    )
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = p.search(line.strip())
            if m:
                idx = int(m.group(1))
                out[idx] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))], dtype=np.float64)
    return out


def parse_quat_file(path):
    p = re.compile(
        r"Frame\s+(\d+)\s+Rotation:\s+X=([-\d.eE+]+),\s*Y=([-\d.eE+]+),\s*Z=([-\d.eE+]+),\s*W=([-\d.eE+]+)"
    )
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = p.search(line.strip())
            if m:
                idx = int(m.group(1))
                out[idx] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))], dtype=np.float64)
    return out


def get_RT_c2w_unity_fixed(q_raw, t_raw):
    x, y, z, w = q_raw
    quat_fixed = [-x, y, -z, w]
    R_wc = quat_xyzw_to_rotmat(quat_fixed[0], quat_fixed[1], quat_fixed[2], quat_fixed[3])
    tx, ty, tz = t_raw
    T_wc = np.array([tx, -ty, tz], dtype=np.float64).reshape(3, 1) * 10.0
    return np.hstack([R_wc, T_wc])


def parse_intrinsic_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    focal_length = float(re.search(r"Focal Length \(mm\)\s*:\s*([-\d.eE+]+)", text).group(1))
    sensor = re.search(r"Sensor Size \(mm\)\s*:\s*X=\s*([-\d.eE+]+)\s+Y=\s*([-\d.eE+]+)", text)
    sensor_width = float(sensor.group(1))
    sensor_height = float(sensor.group(2))
    resolution = re.search(r"images Resolution \(pixel\)\s*:\s*(\d+)\s*\*\s*(\d+)", text)
    cols = float(resolution.group(1))
    rows = float(resolution.group(2))
    principal = re.search(r"Principal point\s*:\s*X=\s*([-\d.eE+]+)\s+Y=\s*([-\d.eE+]+)", text)
    ppx = float(principal.group(1))
    ppy = float(principal.group(2))
    name_prefix = re.search(r"File Name Prefix\s*:\s*([A-Za-z0-9_]*)", text)
    prefix = name_prefix.group(1).strip() if name_prefix else ""
    fx = focal_length * cols / sensor_width
    fy = focal_length * rows / sensor_height
    cx = cols / 2.0 + ppx
    cy = rows / 2.0 + ppy
    return prefix, fx, fy, cx, cy, int(cols), int(rows)


def detect_prefix_from_files(raw_dir):
    prefixes = set()
    for p in raw_dir.glob("*_Intrinsic Data.txt"):
        m = re.match(r"^(.*)_Intrinsic Data\.txt$", p.name)
        if m:
            prefixes.add(m.group(1))
    for p in raw_dir.glob("*_*.png"):
        m = re.match(r"^([A-Za-z0-9]+)_\d{5}\.png$", p.name)
        if m:
            prefixes.add(m.group(1))
    if not prefixes:
        raise RuntimeError("No Prefix Founded")
    return sorted(prefixes)[0]


def main():
    root = Path.cwd()
    explicit_cfg = {
        "raw_dir": root / "SimulatorDatasetSample" / "00000" / "level1" / "raw",
        "out_dir": root / "SimulatorDatasetSample" / "00000" / "level1",
        "depth_scale_from_exr": 5.0,
        "dataset_prefix": None,
    }
    default_cfg = {
        "depth_png_scale": 1,
        "overwrite": True,
    }
    cfg = SimpleNamespace(**default_cfg, **explicit_cfg)

    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.out_dir)
    color_out = out_dir / "color"
    depth_out = out_dir / "depth"

    if cfg.overwrite:
        if color_out.exists():
            shutil.rmtree(color_out)
        if depth_out.exists():
            shutil.rmtree(depth_out)
        pose_out = out_dir / "pose.txt"
        if pose_out.exists():
            pose_out.unlink()

    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    prefix = cfg.dataset_prefix
    if prefix is None:
        prefix = detect_prefix_from_files(raw_dir)

    intrinsic_file = raw_dir / f"{prefix}_Intrinsic Data.txt"
    if not intrinsic_file.exists():
        candidates = sorted(raw_dir.glob("*_Intrinsic Data.txt"))
        if not candidates:
            raise RuntimeError("No Intrinsic Data.txt Founded")
        intrinsic_file = candidates[0]

    file_prefix, fx, fy, cx, cy, cols, rows = parse_intrinsic_file(intrinsic_file)
    if file_prefix:
        prefix = file_prefix

    color_pat = re.compile(rf"{prefix}_(\d{{5}})\.png$")
    depth_pat = re.compile(rf"{prefix}_depth(\d{{5}})\.exr$")

    color_map = {}
    for p in raw_dir.glob(f"{prefix}_*.png"):
        m = color_pat.search(p.name)
        if m:
            color_map[int(m.group(1))] = p

    depth_map = {}
    for p in raw_dir.glob(f"{prefix}_depth*.exr"):
        m = depth_pat.search(p.name)
        if m:
            depth_map[int(m.group(1))] = p

    pos_file = raw_dir / f"{prefix}_Camera Position Data.txt"
    quat_file = raw_dir / f"{prefix}_Camera Quaternion Rotation Data.txt"
    if not quat_file.exists():
        quat_file = raw_dir / f"{prefix}_Camera Total Rotation Data.txt"

    pos_map = parse_position_file(pos_file)
    quat_map = parse_quat_file(quat_file)

    common_ids = sorted(set(color_map) & set(depth_map) & set(pos_map) & set(quat_map))
    if len(common_ids) < 2:
        raise RuntimeError("At least 2 frames needed")

    pose_lines = []
    for new_idx, fid in enumerate(common_ids):
        color_src = color_map[fid]
        depth_src = depth_map[fid]

        out_name = f"{new_idx:010d}.png"
        shutil.copy2(color_src, color_out / out_name)

        d_mm = read_exr(depth_src, depth_channel="R", depth_scale=float(cfg.depth_scale_from_exr))
        depth_name = f"{new_idx:010d}.exr"
        write_exr(depth_out / depth_name, d_mm)

        tx_u, ty_u, tz_u = pos_map[fid]
        qx_u, qy_u, qz_u, qw_u = quat_map[fid]

        rt_wc = get_RT_c2w_unity_fixed([qx_u, qy_u, qz_u, qw_u], [tx_u, ty_u, tz_u])
        R_wc = rt_wc[:, :3]
        t_wc = rt_wc[:, 3]
        q_wc = rotmat_to_quat_xyzw(R_wc)
        tx, ty, tz = t_wc.tolist()
        qx, qy, qz, qw = q_wc.tolist()
        
        # R_cw = R_wc.T
        # t_cw = -R_cw @ t_wc
        # q_cw = rotmat_to_quat_xyzw(R_cw)

        # qx, qy, qz, qw = q_cw.tolist()
        # tx, ty, tz = t_cw.tolist()

        pose_lines.append(
            f"{new_idx:010d} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )

    with open(out_dir / "pose.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pose_lines) + "\n")

    print(f"done: {len(common_ids)} frames")
    print(f"prefix -> {prefix}")
    print(f"intrinsic file -> {intrinsic_file}")
    print(f"color -> {color_out}")
    print(f"depth -> {depth_out}")
    print(f"pose  -> {out_dir / 'pose.txt'}")
    print(f"pose source -> {pos_file} + {quat_file}")
    print(
        f"intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}, res={cols}x{rows}"
    )
    print(f"recommended demo args: --intrinsics {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} --depth_factor 1")


if __name__ == "__main__":
    main()