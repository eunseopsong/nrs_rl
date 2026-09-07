#pragma once

#include "Y2ForceCon/mode3_force_control_core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace y2_control_pybind {

class Mode3ForceController {
public:
    Mode3ForceController(const std::string& model_path, double dt = 0.008,
                         int coordinate = 1,
                         double desired_force_threshold = 0.01,
                         double actual_force_threshold = 1.5,
                         double precontact_force_hold = 15.0,
                         double return_tau = 0.2);

    void reset(const std::vector<double>& measured_pose);
    std::vector<double> step(const std::vector<double>& measured_pose,
                             const std::vector<double>& reference_pose,
                             const std::vector<double>& desired_force,
                             const std::vector<double>& wrench_base,
                             const std::vector<double>& tcp_rotation);

private:
    std::unique_ptr<Mode3ForceControlCore> controller_;
};

}  // namespace y2_control_pybind
