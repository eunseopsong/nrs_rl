import os
import numpy as np
import h5py


def txt9d_to_h5(txt_path: str, h5_path: str) -> None:
    """
    Convert a 9-column txt trajectory file into an HDF5 file.

    Input txt columns:
        0~5 : x y z wx wy wz
        6~8 : fx fy fz

    Output h5 datasets:
        - "position": shape (N, 6)
        - "force":    shape (N, 3)아 참고로 너가 괜찮다 한 convert_txt_to_h5.py 코드는 이 경로에 잇어
￼
￼

    """
    data = np.loadtxt(txt_path, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 9:
        raise ValueError(f"Expected 9 columns, but got {data.shape[1]} columns.")

    position = data[:, :6]
    force = data[:, 6:9]

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("position", data=position)
        f.create_dataset("force", data=force)

    print(f"Saved HDF5 file: {h5_path}")
    print(f"  position shape: {position.shape}")
    print(f"  force shape:    {force.shape}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(current_dir, "cmd_continue9D_10.txt")
    h5_path = os.path.join(current_dir, "cmd_continue9D_convex_2.h5")

    txt9d_to_h5(txt_path, h5_path)
