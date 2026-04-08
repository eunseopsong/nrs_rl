#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_kinematics(py::module_& m);

PYBIND11_MODULE(_y2_control_pybind, m) {
    m.doc() = "Pybind wrapper for Y2 kinematics (UR10e FK / Jacobian / DLS IK)";
    bind_kinematics(m);
}