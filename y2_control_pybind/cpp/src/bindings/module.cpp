#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_kinematics(py::module_& m);
void bind_forcecon(py::module_& m);

PYBIND11_MODULE(_y2_control_pybind, m) {
    m.doc() = "Pybind wrapper for Y2 kinematics and force control";

    bind_kinematics(m);
    bind_forcecon(m);
}