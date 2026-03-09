import numpy as np
import h5py
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. 파일 경로 설정
# ---------------------------------------------------
input_txt = "hand_g_recording.txt"
output_h5 = "hand_g_recording.h5"

# ---------------------------------------------------
# 2. TXT 파일 읽기
# ---------------------------------------------------
data = np.loadtxt(input_txt)

# ---------------------------------------------------
# 3. 앞 6개 열만 선택 (x, y, z, r, p, y)
# ---------------------------------------------------
data_xyzrpy = data[:, :6]

# ---------------------------------------------------
# 4. HDF5 파일로 저장
# ---------------------------------------------------
with h5py.File(output_h5, "w") as f:
    f.create_dataset("target_positions", data=data_xyzrpy)

print(f"[INFO] Saved {data_xyzrpy.shape} data to '{output_h5}' with key 'target_positions'")

# ---------------------------------------------------
# 5. HDF5 파일에서 데이터 다시 읽기
# ---------------------------------------------------
with h5py.File(output_h5, "r") as f:
    loaded_data = f["target_positions"][:]

# ---------------------------------------------------
# 6. 시각화 (x, y, z, r, p, y)
# ---------------------------------------------------
labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
plt.figure(figsize=(10, 6))

for i in range(loaded_data.shape[1]):
    plt.plot(loaded_data[:, i], label=labels[i])

plt.title("Hand G Recording - Cartesian Position and Orientation (x, y, z, r, p, y)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

