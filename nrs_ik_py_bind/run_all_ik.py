#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import nrs_ik_core as nrs_ik  # 또는: from nrs_ik_py import IKSolver, PoseRPY

SRC_TXT = "/home/eunseop/nrs_ws/src/rtde_handarm2/data/hand_g_recording.txt"
OUT_TXT = str(Path.home() / "nrs_lab2/datasets/ik_py_results.txt")

TOOL_Z = 0.239
USE_DEGREES = False  # 입력 RPY 단위(라디안이면 False)

def load_txt_first6(path, use_degrees=False):
    """9열 파일에서 앞 6열(x y z r p y)만 읽는다."""
    poses = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 6:
                continue
            # 항상 앞 6개만 사용 (뒤의 fx fy fz는 무시)
            x, y, z, r, p, yaw = map(float, cols[:6])
            if use_degrees:
                import math
                D2R = math.pi / 180.0
                r *= D2R; p *= D2R; yaw *= D2R
            pose = nrs_ik.PoseRPY()
            pose.line_no = line_no
            pose.x, pose.y, pose.z = x, y, z
            pose.r, pose.p, pose.yaw = r, p, yaw
            poses.append(pose)
    return poses

def main():
    print("============== Running IK for all poses =================")
    poses = load_txt_first6(SRC_TXT, use_degrees=USE_DEGREES)
    print(f"TXT 로드: {SRC_TXT}  (valid poses: {len(poses)})")

    solver = nrs_ik.IKSolver(TOOL_Z, USE_DEGREES)

    qs = []
    for i, pose in enumerate(poses, 1):
        ok, q = solver.compute(pose)
        if not ok:
            print(f"[{i}/{len(poses)}] IK 실패 (line {pose.line_no})")
            continue
        qs.append(np.array(q).reshape(6))
        if i % 50 == 0 or i == len(poses):
            print(f"[{i}/{len(poses)}] IK 진행 중...")

    if not qs:
        raise RuntimeError("유효한 IK 결과가 없습니다.")
    qs = np.vstack(qs)

    Path(OUT_TXT).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_TXT, qs, fmt="%.9f")
    print(f"✅ IK 결과 저장 완료: {OUT_TXT}  (rows={qs.shape[0]}, cols=6)")

if __name__ == "__main__":
    main()
