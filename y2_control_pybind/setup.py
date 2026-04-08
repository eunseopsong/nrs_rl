from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pathlib
import subprocess
import sys
import os


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import torch  # lazy import: only when actual build runs

        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent

        build_temp = pathlib.Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True
        ).strip()

        torch_cmake_dir = torch.utils.cmake_prefix_path
        torch_abi_flag = "1" if torch.compiled_with_cxx11_abi() else "0"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            f"-DTorch_DIR={torch_cmake_dir}/Torch",
            f"-DTORCH_CXX11_ABI={torch_abi_flag}",
        ]

        build_args = [
            "--config", cfg,
            "--parallel",
        ]

        subprocess.check_call(
            ["cmake", str(pathlib.Path(__file__).parent.resolve())] + cmake_args,
            cwd=build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )


ext_modules = [
    Extension(
        name="y2_control_py._y2_control_pybind",
        sources=[],
    )
]

setup(
    name="y2_control_pybind",
    version="0.0.1",
    description="Pybind wrapper for Y2 UR10e kinematics and Mode5 force control",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)