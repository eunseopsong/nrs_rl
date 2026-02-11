# confirm_fk_results_show.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = os.path.expanduser('~/nrs_lab2/datasets')

REC_PATH = os.path.expanduser('~/nrs_ws/src/rtde_handarm2/data/hand_g_recording.txt')  # 9열 중 앞 6열 사용
FK_RESULT_PATH = os.path.join(DATASET_DIR, 'fk_py_results.txt')                          # 6열 (x y z r p y) [rad]
ERR_OUT_PATH = os.path.join(DATASET_DIR, 'fk_compare_errors.txt')

# 둘 다 라디안이면 False 유지
ORIG_IN_DEGREES = False
FK_IN_DEGREES   = False

FIGSIZE = (11, 4)

def wrap_angle(ang):
    return (ang + math.pi) % (2.0 * math.pi) - math.pi

def load_recording_first6(path, use_degrees=False):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            vals = list(map(float, parts[:6]))  # x y z r p y
            rows.append(vals)
    X = np.array(rows, dtype=float)
    if use_degrees and X.size > 0:
        D2R = math.pi / 180.0
        X[:, 3:6] *= D2R
    return X

def load_fk_results(path, use_degrees=False):
    X = np.loadtxt(path, comments='#')
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if use_degrees and X.size > 0:
        D2R = math.pi / 180.0
        X[:, 3:6] *= D2R
    return X

def plot_series(idx, ys, labels, title, ylabel):
    plt.figure(figsize=FIGSIZE)
    for y, label in zip(ys, labels):
        plt.plot(idx, y, label=label)
    plt.title(title)
    plt.xlabel('sample index')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hist(data, title, xlabel, bins=80):
    plt.figure(figsize=FIGSIZE)
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("============== Compare FK results (show plots) ==============")
    os.makedirs(DATASET_DIR, exist_ok=True)

    ref = load_recording_first6(REC_PATH, use_degrees=ORIG_IN_DEGREES)  # rad
    est = load_fk_results(FK_RESULT_PATH, use_degrees=FK_IN_DEGREES)    # rad

    n_ref, n_est = len(ref), len(est)
    print(f"원본 poses: {REC_PATH} (rows={n_ref})")
    print(f"FK 결과  : {FK_RESULT_PATH} (rows={n_est})")

    if n_ref == 0 or n_est == 0:
        raise RuntimeError("입력 데이터가 비었습니다. (원본/또는 FK 결과 확인)")

    N = min(n_ref, n_est)
    if n_ref != n_est:
        print(f"[경고] 행 수 불일치 → 최소 공통 {N}행만 비교합니다.")
    ref = ref[:N, :]
    est = est[:N, :]

    # 오차 계산
    dpos = est[:, 0:3] - ref[:, 0:3]
    drot = est[:, 3:6] - ref[:, 3:6]
    for k in range(3):
        drot[:, k] = np.vectorize(wrap_angle)(drot[:, k])

    pos_norm = np.linalg.norm(dpos, axis=1)
    ori_norm = np.linalg.norm(drot, axis=1)

    # 텍스트 저장(유지)
    out = np.hstack([dpos, drot, pos_norm.reshape(-1,1), ori_norm.reshape(-1,1)])
    header = "dx dy dz droll dpitch dyaw pos_err_norm ori_err_norm  (angles in rad)"
    np.savetxt(ERR_OUT_PATH, out, fmt="%.9f", header=header)
    print(f"오차 저장: {ERR_OUT_PATH}  (rows={N})")

    # 그래프 표시
    idx = np.arange(N)
    plot_series(idx, [dpos[:,0], dpos[:,1], dpos[:,2]], ['dx','dy','dz'],
                'Position error per sample', 'pos error [m]')
    plot_series(idx, [drot[:,0], drot[:,1], drot[:,2]], ['droll','dpitch','dyaw'],
                'Orientation (RPY) error per sample', 'angle error [rad]')
    plot_series(idx, [pos_norm], ['||pos_err||'],
                'Position error norm per sample', 'norm [m]')
    plot_series(idx, [ori_norm], ['||rpy_err||'],
                'Orientation error norm per sample', 'norm [rad]')

    plot_hist(dpos[:,0], 'Histogram of dx', 'dx [m]')
    plot_hist(dpos[:,1], 'Histogram of dy', 'dy [m]')
    plot_hist(dpos[:,2], 'Histogram of dz', 'dz [m]')
    plot_hist(drot[:,0], 'Histogram of droll', 'droll [rad]')
    plot_hist(drot[:,1], 'Histogram of dpitch', 'dpitch [rad]')
    plot_hist(drot[:,2], 'Histogram of dyaw', 'dyaw [rad]')
    plot_hist(pos_norm, 'Histogram of ||pos_err||', 'norm [m]')
    plot_hist(ori_norm, 'Histogram of ||rpy_err||', 'norm [rad]')

    print("완료. 창 닫으면 다음 그림이 순차적으로 표시됩니다.")

if __name__ == "__main__":
    main()
