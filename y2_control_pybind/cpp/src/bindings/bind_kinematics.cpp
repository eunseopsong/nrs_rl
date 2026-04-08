#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "y2_control_pybind/kinematics_api.hpp"

namespace py = pybind11;

void bind_kinematics(py::module_& m) {
    py::class_<y2_control_pybind::UR10eKinematics>(m, "UR10eKinematics")
        .def(py::init<double, const std::vector<std::vector<double>>&>(),
             py::arg("dt") = 0.01,
             py::arg("ee2tcp") = std::vector<std::vector<double>>{
                 {1.0, 0.0, 0.0, 0.0},
                 {0.0, 1.0, 0.0, 0.0},
                 {0.0, 0.0, 1.0, 0.0},
                 {0.0, 0.0, 0.0, 1.0},
             })
        .def("forward_kinematics",
             &y2_control_pybind::UR10eKinematics::forward_kinematics,
             py::arg("q"),
             "Compute 4x4 forward kinematics HTM from 6-DoF joint vector.")
        .def("calculate_jacobian",
             &y2_control_pybind::UR10eKinematics::calculate_jacobian,
             py::arg("q"),
             "Compute 6x6 geometric Jacobian from 6-DoF joint vector.")
        .def("solve_ik",
             &y2_control_pybind::UR10eKinematics::solve_ik,
             py::arg("q_current"),
             py::arg("target_htm"),
             "Run one DLS-based IK step and return q_next.")
        .def("set_control_gains",
             &y2_control_pybind::UR10eKinematics::set_control_gains,
             py::arg("kp_pos"),
             py::arg("kp_rot"))
        .def("set_prev_q",
             &y2_control_pybind::UR10eKinematics::set_prev_q,
             py::arg("q_prev"))
        .def("set_joint_limits",
             &y2_control_pybind::UR10eKinematics::set_joint_limits,
             py::arg("q_min"),
             py::arg("q_max"),
             py::arg("qd_min"),
             py::arg("qd_max"))
        .def("set_accel_limits",
             &y2_control_pybind::UR10eKinematics::set_accel_limits,
             py::arg("a_min"),
             py::arg("a_max"));
}