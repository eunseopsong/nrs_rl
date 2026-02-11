#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "ik_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(nrs_ik_core, m) {
    m.doc() = "IK Solver Python bindings for UR10e (via pybind11)";

    py::class_<PoseRPY>(m, "PoseRPY")
        .def(py::init<>())
        .def_readwrite("line_no", &PoseRPY::line_no)
        .def_readwrite("x", &PoseRPY::x)
        .def_readwrite("y", &PoseRPY::y)
        .def_readwrite("z", &PoseRPY::z)
        .def_readwrite("r", &PoseRPY::r)
        .def_readwrite("p", &PoseRPY::p)
        .def_readwrite("yaw", &PoseRPY::yaw);

    py::class_<IKSolver>(m, "IKSolver")
        .def(py::init<double, bool>(), py::arg("tool_z"), py::arg("use_degrees"))
        .def("load_txt", &IKSolver::load_txt, "Load TCP trajectory txt file")
        .def("compute", &IKSolver::compute, "Compute IK for a given PoseRPY")
        .def("compute_idx", [](IKSolver &self, int idx) {
            const auto& poses = self.getPoses();
            if (idx < 0 || idx >= (int)poses.size())
                throw std::out_of_range("Index out of range");
            return self.compute(poses[idx]);
        }, "Compute IK for pose by index");
}
