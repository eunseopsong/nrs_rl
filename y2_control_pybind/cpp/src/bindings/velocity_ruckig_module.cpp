#include <pybind11/pybind11.h>
#include "y2_control_pybind/ppo_ruckig_speed_limiter.hpp"

PYBIND11_MODULE(_velocity_ruckig, m)
{
    namespace py = pybind11;
    py::class_<PpoRuckigSpeedLimiter>(m, "PpoRuckigSpeedLimiter")
        .def(py::init<double, double, double, double>(),
             py::arg("dt") = 0.008, py::arg("max_speed") = 12.0,
             py::arg("max_acceleration") = 16.0, py::arg("max_jerk") = 160.0)
        .def("reset", &PpoRuckigSpeedLimiter::reset)
        .def("step", &PpoRuckigSpeedLimiter::step, py::arg("requested_speed"), py::arg("enabled") = true)
        .def_property_readonly("speed", &PpoRuckigSpeedLimiter::speed)
        .def_property_readonly("acceleration", &PpoRuckigSpeedLimiter::acceleration)
        .def_property_readonly("jerk", &PpoRuckigSpeedLimiter::jerk);
}
