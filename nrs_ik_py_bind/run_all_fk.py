# run_all_fk.py
import os
import numpy as np
from nrs_fk_core import FKSolver

DATASET_DIR = os.path.expanduser('~/nrs_lab2/datasets')
IK_RESULT_PATH = os.path.join(DATASET_DIR, 'ik_py_results.txt')   # q1..q6 (rad)
FK_RESULT_PATH = os.path.join(DATASET_DIR, 'fk_py_results.txt')   # x y z r p y (rad)

TOOL_Z = 0.239       # EE->TCP z-offset (meters), IK와 동일하게
USE_DEGREES = False  # FK 결과 RPY를 rad로 저장 (IK와 맞춤)

def load_ik_q6(path):
    """6열(q1..q6)만 읽어서 (N,6) ndarray로 반환. 주석/빈줄 무시."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            vals = list(map(float, parts[:6]))  # 첫 6개만 사용
            rows.append(vals)
    if not rows:
        return np.empty((0, 6), dtype=float)
    return np.array(rows, dtype=float)

def main():
    print("============== Running FK for IK results =================")
    Q = load_ik_q6(IK_RESULT_PATH)
    print(f"IK 결과 로드: {IK_RESULT_PATH}  (rows={len(Q)})")

    if Q.size == 0:
        raise RuntimeError("IK 결과가 비어 있습니다. run_all_ik.py를 먼저 실행했는지 확인하세요.")

    fk = FKSolver(tool_z=TOOL_Z, use_degrees=USE_DEGREES)

    results = np.zeros((len(Q), 6), dtype=float)

    for i, q in enumerate(Q):
        ok, pose = fk.compute(q, as_degrees=USE_DEGREES)
        if not ok:
            # 실패 시 NaN 채우기 (필요하면 스킵하도록 변경 가능)
            results[i, :] = np.nan
        else:
            results[i, 0] = pose.x
            results[i, 1] = pose.y
            results[i, 2] = pose.z
            results[i, 3] = pose.r
            results[i, 4] = pose.p
            results[i, 5] = pose.yaw

        if (i + 1) % 500 == 0 or (i + 1) == len(Q):
            print(f"  processed {i+1}/{len(Q)}")

    os.makedirs(DATASET_DIR, exist_ok=True)
    header = "x y z roll pitch yaw" + (" (deg)" if USE_DEGREES else " (rad)")
    np.savetxt(FK_RESULT_PATH, results, fmt="%.9f", header=header)
    print(f"FK 결과 저장: {FK_RESULT_PATH}  (rows={len(results)})")

if __name__ == "__main__":
    main()
