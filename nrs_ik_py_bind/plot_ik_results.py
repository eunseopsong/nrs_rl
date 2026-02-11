import nrs_ik_core as nrs_ik
import matplotlib.pyplot as plt

def load_txt(path, use_degrees=False):
    poses = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 6:
                continue
            x, y, z, r, p, yaw = map(float, cols[:6])
            if use_degrees:
                import math
                D2R = math.pi / 180.0
                r *= D2R
                p *= D2R
                yaw *= D2R
            pose = nrs_ik.PoseRPY()
            pose.line_no = line_no
            pose.x, pose.y, pose.z = x, y, z
            pose.r, pose.p, pose.yaw = r, p, yaw
            poses.append(pose)
    return poses


def main():
    solver = nrs_ik.IKSolver(0.239, False)
    txt_path = "/home/eunseop/nrs_ws/src/rtde_handarm2/data/hand_g_recording.txt"
    poses = load_txt(txt_path, use_degrees=False)

    all_q = []
    for pose in poses:
        ok, q = solver.compute(pose)
        if ok:
            all_q.append(q.flatten())

    if not all_q:
        print("IK 결과 없음")
        return

    import numpy as np
    all_q = np.array(all_q)  # shape = (N, 6)

    # 관절별 궤적 그리기
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.plot(all_q[:, i], label=f"q{i+1}")
    plt.title("Inverse Kinematics Joint Trajectories")
    plt.xlabel("Pose Index")
    plt.ylabel("Joint Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
