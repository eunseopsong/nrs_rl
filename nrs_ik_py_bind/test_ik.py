import nrs_ik_py as nrs_ik

def main():
    solver = nrs_ik.IKSolver(tool_z=0.239, use_degrees=True)

    if solver.load_txt("/home/eunseop/nrs_ws/src/rtde_handarm2/data/hand_g_recording.txt"):
        print("TXT 파일 로드 성공")

        # ✅ 인덱스로 바로 호출 가능
        success, q = solver.compute_idx(0)
        print("IK 성공 여부:", success)
        print("첫 Pose의 IK 결과:", q)
    else:
        print("TXT 파일 로드 실패")

if __name__ == "__main__":
    main()
