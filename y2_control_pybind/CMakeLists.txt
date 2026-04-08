cmake_minimum_required(VERSION 3.18)
project(y2_control_pybind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

add_compile_definitions(Y2_IK_SOLVER_MODE=Y2_IK_SOLVER_DLS)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

set(Y2_MATRIX_SRCS
    cpp/src/Y2Matrix/YMatrix.cpp
    cpp/src/Y2Matrix/RotationTransform.cpp
)

set(Y2_KINEMATICS_SRCS
    cpp/src/Y2Kinematics/Kinematics.cpp
    cpp/src/Y2Kinematics/KinematicsUR10e.cpp
    cpp/src/Y2Kinematics/QP_solver.cpp
)

set(Y2_API_SRCS
    cpp/src/api/kinematics_api.cpp
)

set(Y2_BINDING_SRCS
    cpp/src/bindings/module.cpp
    cpp/src/bindings/bind_kinematics.cpp
)

pybind11_add_module(_y2_control_pybind
    ${Y2_MATRIX_SRCS}
    ${Y2_KINEMATICS_SRCS}
    ${Y2_API_SRCS}
    ${Y2_BINDING_SRCS}
)

target_include_directories(_y2_control_pybind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

set_target_properties(_y2_control_pybind PROPERTIES
    PREFIX ""
)

install(TARGETS _y2_control_pybind
    LIBRARY DESTINATION y2_control_py
    ARCHIVE DESTINATION y2_control_py
    RUNTIME DESTINATION y2_control_py
)