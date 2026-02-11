#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "fk_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(nrs_fk_core, m) {
    m.doc() = "FK Solver Python bindings for UR10(e) (via pybind11)";

    py::class_<PoseRPY>(m, "PoseRPY")
        .def(py::init<>())
        .def_readwrite("line_no", &PoseRPY::line_no)
        .def_readwrite("x", &PoseRPY::x)
        .def_readwrite("y", &PoseRPY::y)
        .def_readwrite("z", &PoseRPY::z)
        .def_readwrite("r", &PoseRPY::r)
        .def_readwrite("p", &PoseRPY::p)
        .def_readwrite("yaw", &PoseRPY::yaw);

    py::class_<FKSolver>(m, "FKSolver")
        .def(py::init<double, bool>(), py::arg("tool_z"), py::arg("use_degrees") = false)
        .def("compute",
             &FKSolver::compute,
             py::arg("q"), py::arg("as_degrees") = false,
             R"doc(
                 Forward kinematics.
                 Args:
                   q          : numpy array / Eigen vector of shape (6,)
                   as_degrees : if True, returns roll/pitch/yaw in degrees
                 Returns:
                   (ok: bool, pose: PoseRPY)
             )doc")
        .def("transform",
             &FKSolver::transform,
             py::arg("q"), py::arg("as_degrees") = false,
             "Alias of compute(q, as_degrees) returning PoseRPY only.");
}
