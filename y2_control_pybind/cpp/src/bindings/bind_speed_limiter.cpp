#include <pybind11/pybind11.h>
#include "y2_control_pybind/ppo_speed_limiter.hpp"

void bind_speed_limiter(pybind11::module_& m)
{
    namespace py = pybind11;
    py::class_<PpoSpeedLimiter>(m, "PpoSpeedLimiter")
        .def(py::init<double, double, double, double>(),
             py::arg("dt") = 0.008, py::arg("max_speed") = 12.0,
             py::arg("max_acceleration") = 16.0, py::arg("max_jerk") = 160.0)
        .def("reset", &PpoSpeedLimiter::reset)
        .def("step", &PpoSpeedLimiter::step, py::arg("requested_speed"), py::arg("enabled") = true)
        .def_property_readonly("speed", &PpoSpeedLimiter::speed)
        .def_property_readonly("acceleration", &PpoSpeedLimiter::acceleration)
        .def_property_readonly("jerk", &PpoSpeedLimiter::jerk);
}
