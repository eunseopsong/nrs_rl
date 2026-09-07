#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "y2_control_pybind/forcecon_api.hpp"

namespace py = pybind11;

void bind_forcecon(py::module_& m)
{
    py::class_<y2_control_pybind::Mode3ForceController>(m, "Mode3ForceController")
        .def(py::init<const std::string&, double, int, double, double, double, double>(),
             py::arg("model_path"), py::arg("dt") = 0.008,
             py::arg("coordinate") = 1,
             py::arg("desired_force_threshold") = 0.01,
             py::arg("actual_force_threshold") = 1.5,
             py::arg("precontact_force_hold") = 15.0,
             py::arg("return_tau") = 0.2)
        .def("reset", &y2_control_pybind::Mode3ForceController::reset)
        .def("step", &y2_control_pybind::Mode3ForceController::step,
             py::arg("measured_pose"), py::arg("reference_pose"),
             py::arg("desired_force"), py::arg("wrench_base"),
             py::arg("tcp_rotation"));
}
