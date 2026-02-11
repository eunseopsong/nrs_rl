from setuptools import setup, Extension
import pybind11
import sys
import os

# 공통 설정
EIGEN_INCLUDE = "/usr/include/eigen3"  # 우분투 기본
COMMON_INCLUDES = [
    pybind11.get_include(),
    EIGEN_INCLUDE,
    "nrs_ik_py",
]
COMMON_COMPILE_ARGS = ["-O3", "-std=c++17", "-fvisibility=hidden"]
COMMON_MACROS = [("EIGEN_NO_DEBUG", "1")]

ext_modules = [
    # IK 확장 모듈
    Extension(
        name="nrs_ik_core",
        sources=[
            "nrs_ik_py/ik_bindings.cpp",
            "nrs_ik_py/ik_solver.cpp",
            "nrs_ik_py/Kinematics.cpp",
        ],
        include_dirs=COMMON_INCLUDES,
        define_macros=COMMON_MACROS,
        extra_compile_args=COMMON_COMPILE_ARGS,
        language="c++",
    ),
    # FK 확장 모듈
    Extension(
        name="nrs_fk_core",
        sources=[
            "nrs_ik_py/fk_bindings.cpp",
            "nrs_ik_py/fk_solver.cpp",
            "nrs_ik_py/Kinematics.cpp",
        ],
        include_dirs=COMMON_INCLUDES,
        define_macros=COMMON_MACROS,
        extra_compile_args=COMMON_COMPILE_ARGS,
        language="c++",
    ),
]

setup(
    name="nrs_ik_py",
    version="0.1.0",
    packages=["nrs_ik_py"],
    ext_modules=ext_modules,
)
