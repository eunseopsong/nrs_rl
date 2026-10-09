from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pathlib
import os
import subprocess
import sys


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths

        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        extdir.mkdir(parents=True, exist_ok=True)

        build_temp = pathlib.Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True,
        ).strip()

        torch_include_dirs = include_paths()
        torch_library_dirs = library_paths()
        torch_abi_flag = "1" if torch.compiled_with_cxx11_abi() else "0"

        # Link against the torch libs from the CURRENT python env only
        # Keep this minimal and stable
        torch_lib_names = ["torch", "torch_cpu", "c10"]

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            f"-DTORCH_CXX11_ABI={torch_abi_flag}",
            f"-DTORCH_INCLUDE_DIRS={';'.join(torch_include_dirs)}",
            f"-DTORCH_LIBRARY_DIRS={';'.join(torch_library_dirs)}",
            f"-DTORCH_LIB_NAMES={';'.join(torch_lib_names)}",
            "-DY2_CONTROL_SOURCE_DIR=" + os.environ.get(
                "Y2_CONTROL_SOURCE_DIR",
                "/home/eunseop/dev_ws/src/y2_ur10skku_control",
            ),
        ]

        build_args = [
            "--config", cfg,
            "--target", ext.name.rsplit(".", 1)[-1],
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
    ),
]
if os.environ.get("NRS_BUILD_RUCKIG", "0") == "1":
    ext_modules.append(Extension(name="y2_control_py._velocity_ruckig", sources=[]))

setup(
    name="y2_control_pybind",
    version="0.0.1",
    description="Python bindings compiled from the live Y2 robot controller sources",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
