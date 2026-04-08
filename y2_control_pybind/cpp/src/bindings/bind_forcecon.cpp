#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "y2_control_pybind/forcecon_api.hpp"

namespace py = pybind11;

void bind_forcecon(py::module_& m) {
    py::class_<y2_control_pybind::Admittance1D>(m, "Admittance1D")
        .def(py::init<double>(), py::arg("dt") = 0.002)
        .def("set_mdk", &y2_control_pybind::Admittance1D::set_mdk,
             py::arg("mass"), py::arg("damping"), py::arg("stiffness"))
        .def("monitor_mdk", &y2_control_pybind::Admittance1D::monitor_mdk,
             py::arg("select"))
        .def("step", &y2_control_pybind::Admittance1D::step,
             py::arg("xd"), py::arg("fd"), py::arg("fext"))
        .def("reset", &y2_control_pybind::Admittance1D::reset,
             py::arg("xd"))
        .def("hard_reset", &y2_control_pybind::Admittance1D::hard_reset);

    py::class_<y2_control_pybind::ContextNAFMdGradiPolicy>(m, "ContextNAFMdGradiPolicy")
        .def(py::init<
                 const std::string&,
                 int,
                 const std::string&,
                 float,
                 float,
                 float,
                 std::vector<float>,
                 std::vector<float>,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float>(),
             py::arg("model_path"),
             py::arg("threads") = 1,
             py::arg("device") = "cpu",
             py::arg("dt") = 0.002f,
             py::arg("md_ratio") = 1000.0f,
             py::arg("fc_fext") = 50.0f,
             py::arg("action_low") = std::vector<float>{-0.25f, -0.25f},
             py::arg("action_high") = std::vector<float>{+0.25f, +0.25f},
             py::arg("mass_min") = 0.5f,
             py::arg("mass_max") = 5.0f,
             py::arg("alpha_min") = 0.5f,
             py::arg("alpha_max") = 3.0f,
             py::arg("alpha_rate_up") = 4.0f,
             py::arg("alpha_rate_down") = 4.0f)
        .def("run", &y2_control_pybind::ContextNAFMdGradiPolicy::run,
             py::arg("xc"), py::arg("x"), py::arg("fd"), py::arg("env_fext"))
        .def("set_sampling_time", &y2_control_pybind::ContextNAFMdGradiPolicy::set_sampling_time,
             py::arg("dt"))
        .def("reset_state", &y2_control_pybind::ContextNAFMdGradiPolicy::reset_state)
        .def("set_md_ratio", &y2_control_pybind::ContextNAFMdGradiPolicy::set_md_ratio,
             py::arg("md_ratio"))
        .def("set_alpha_rate_limits", &y2_control_pybind::ContextNAFMdGradiPolicy::set_alpha_rate_limits,
             py::arg("rate_up"), py::arg("rate_down"))
        .def("set_physical_bounds", &y2_control_pybind::ContextNAFMdGradiPolicy::set_physical_bounds,
             py::arg("mass_min"), py::arg("mass_max"), py::arg("alpha_min"), py::arg("alpha_max"))
        .def("get_applied_mass_alpha", &y2_control_pybind::ContextNAFMdGradiPolicy::get_applied_mass_alpha)
        .def("get_applied_damping", &y2_control_pybind::ContextNAFMdGradiPolicy::get_applied_damping)
        .def("get_last_network_action", &y2_control_pybind::ContextNAFMdGradiPolicy::get_last_network_action)
        .def("get_last_alpha_cmd_abs", &y2_control_pybind::ContextNAFMdGradiPolicy::get_last_alpha_cmd_abs)
        .def("get_filtered_fext", &y2_control_pybind::ContextNAFMdGradiPolicy::get_filtered_fext);

    py::class_<y2_control_pybind::ForceCon1DMode5>(m, "ForceCon1DMode5")
        .def(py::init<
                 const std::string&,
                 double,
                 int,
                 const std::string&,
                 double,
                 double,
                 double,
                 double,
                 double,
                 double,
                 double,
                 std::vector<float>,
                 std::vector<float>,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float>(),
             py::arg("model_path"),
             py::arg("dt") = 0.002,
             py::arg("threads") = 1,
             py::arg("device") = "cpu",
             py::arg("md_ratio") = 1000.0,
             py::arg("fc_fext") = 50.0,
             py::arg("free_mass") = 2.0,
             py::arg("free_damping") = 6000.0,
             py::arg("free_stiffness") = 2000.0,
             py::arg("contact_stiffness") = 0.0,
             py::arg("recovery_tau") = 3.0,
             py::arg("action_low") = std::vector<float>{-0.25f, -0.25f},
             py::arg("action_high") = std::vector<float>{+0.25f, +0.25f},
             py::arg("mass_min") = 0.5f,
             py::arg("mass_max") = 5.0f,
             py::arg("alpha_min") = 0.5f,
             py::arg("alpha_max") = 3.0f,
             py::arg("alpha_rate_up") = 4.0f,
             py::arg("alpha_rate_down") = 4.0f)
        .def("reset", &y2_control_pybind::ForceCon1DMode5::reset,
             py::arg("xd"))
        .def("set_free_mdk", &y2_control_pybind::ForceCon1DMode5::set_free_mdk,
             py::arg("mass"), py::arg("damping"), py::arg("stiffness"))
        .def("set_contact_stiffness", &y2_control_pybind::ForceCon1DMode5::set_contact_stiffness,
             py::arg("stiffness"))
        .def("set_recovery_tau", &y2_control_pybind::ForceCon1DMode5::set_recovery_tau,
             py::arg("tau_sec"))
        .def("set_md_ratio", &y2_control_pybind::ForceCon1DMode5::set_md_ratio,
             py::arg("md_ratio"))
        .def("step", &y2_control_pybind::ForceCon1DMode5::step,
             py::arg("xd"), py::arg("x"), py::arg("fd"), py::arg("fext"),
             "Returns [xc, mass, alpha, damping, stiffness, filtered_fext].")
        .def("get_current_mdk", &y2_control_pybind::ForceCon1DMode5::get_current_mdk);
}