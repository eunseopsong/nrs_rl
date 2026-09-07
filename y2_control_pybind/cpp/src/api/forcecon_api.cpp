#include "y2_control_pybind/forcecon_api.hpp"

#include <algorithm>
#include <stdexcept>

namespace y2_control_pybind {
namespace {
template<std::size_t Size>
std::array<double, Size> toArray(const std::vector<double>& values, const char* name)
{
    if(values.size() != Size) {
        throw std::invalid_argument(
            std::string(name) + " must have " + std::to_string(Size) + " values");
    }
    std::array<double, Size> output{};
    std::copy_n(values.begin(), Size, output.begin());
    return output;
}
}  // namespace

Mode3ForceController::Mode3ForceController(
    const std::string& model_path, double dt, int coordinate,
    double desired_force_threshold, double actual_force_threshold,
    double precontact_force_hold, double return_tau)
{
    Mode3ForceControlConfig config;
    config.control_period = dt;
    config.force_control_coordinate = coordinate;
    config.desired_force_threshold = desired_force_threshold;
    config.actual_force_threshold = actual_force_threshold;
    config.precontact_force_hold = precontact_force_hold;
    config.return_tau_mass = return_tau;
    config.return_tau_damping = return_tau;
    config.return_tau_stiffness = return_tau;
    controller_ = std::make_unique<Mode3ForceControlCore>(
        config, model_path, 1, torch::kCPU);
}

void Mode3ForceController::reset(const std::vector<double>& measured_pose)
{
    controller_->reset(toArray<6>(measured_pose, "measured_pose"));
}

std::vector<double> Mode3ForceController::step(
    const std::vector<double>& measured_pose,
    const std::vector<double>& reference_pose,
    const std::vector<double>& desired_force,
    const std::vector<double>& wrench_base,
    const std::vector<double>& tcp_rotation)
{
    Mode3ForceControlInput input;
    input.measured_pose = toArray<6>(measured_pose, "measured_pose");
    input.reference_pose = toArray<6>(reference_pose, "reference_pose");
    input.desired_force = toArray<3>(desired_force, "desired_force");
    input.wrench_base = toArray<6>(wrench_base, "wrench_base");
    input.tcp_rotation = toArray<9>(tcp_rotation, "tcp_rotation");
    const Mode3ForceControlOutput output = controller_->step(input);

    std::vector<double> result;
    result.reserve(24);
    result.insert(result.end(), output.command_pose.begin(), output.command_pose.end());
    result.insert(result.end(), output.mass.begin(), output.mass.end());
    result.insert(result.end(), output.damping.begin(), output.damping.end());
    result.insert(result.end(), output.stiffness.begin(), output.stiffness.end());
    return result;
}

}  // namespace y2_control_pybind
