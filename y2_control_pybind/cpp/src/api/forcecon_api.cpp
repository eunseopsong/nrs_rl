#include "y2_control_pybind/forcecon_api.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace y2_control_pybind {

namespace {

c10::Device parse_device(const std::string& device_str) {
    if (device_str == "cpu" || device_str == "CPU") {
        return c10::Device(torch::kCPU);
    }
    if (device_str.rfind("cuda", 0) == 0 || device_str.rfind("CUDA", 0) == 0) {
        return c10::Device(torch::kCUDA);
    }
    throw std::invalid_argument("Unsupported device string: " + device_str);
}

std::array<float, 2> vec2_to_array2(const std::vector<float>& v, const char* name) {
    if (v.size() != 2) {
        throw std::invalid_argument(std::string(name) + " must have exactly 2 elements");
    }
    return {v[0], v[1]};
}

double clamp_positive(double v, double fallback) {
    return (std::isfinite(v) && v > 0.0) ? v : fallback;
}

double exp_smoothing_alpha(double dt, double tau) {
    if (tau <= 0.0) {
        return 1.0;
    }
    return 1.0 - std::exp(-dt / tau);
}

double signed_force_hold(double desired_force, double hold_force) {
    return desired_force < 0.0 ? -hold_force : hold_force;
}

}  // namespace

// =============================
// Admittance1D
// =============================

Admittance1D::Admittance1D(double dt)
: adm_(dt) {}

void Admittance1D::set_mdk(double mass, double damping, double stiffness) {
    adm_.adm_1D_MDK(mass, damping, stiffness);
}

double Admittance1D::monitor_mdk(int select) const {
    return const_cast<Yadmittance_control&>(adm_).adm_MDK_monitor(select);
}

double Admittance1D::step(double xd, double fd, double fext) {
    return adm_.adm_1D_control(xd, fd, fext);
}

void Admittance1D::reset(double xd) {
    adm_.reset(xd);
}

void Admittance1D::hard_reset() {
    adm_.hardReset();
}

// =============================
// ContextNAFMdGradiPolicy
// =============================

ContextNAFMdGradiPolicy::ContextNAFMdGradiPolicy(
    const std::string& model_path,
    int threads,
    const std::string& device,
    float dt,
    float md_ratio,
    float fc_fext,
    std::vector<float> action_low,
    std::vector<float> action_high,
    float mass_min,
    float mass_max,
    float alpha_min,
    float alpha_max,
    float alpha_rate_up,
    float alpha_rate_down
) {
    policy_ = std::make_unique<RL_ContextNAF_mdGradi>(
        model_path,
        threads,
        parse_device(device),
        dt,
        md_ratio,
        fc_fext,
        vec2_to_array2(action_low, "action_low"),
        vec2_to_array2(action_high, "action_high"),
        mass_min,
        mass_max,
        alpha_min,
        alpha_max,
        alpha_rate_up,
        alpha_rate_down
    );
}

std::vector<double> ContextNAFMdGradiPolicy::run(float xc, float x, float fd, float env_fext) {
    return policy_->run(xc, x, fd, env_fext);
}

void ContextNAFMdGradiPolicy::set_sampling_time(float dt) {
    policy_->set_sampling_time(dt);
}

void ContextNAFMdGradiPolicy::reset_state() {
    policy_->reset_state();
}

void ContextNAFMdGradiPolicy::set_md_ratio(float md_ratio) {
    policy_->set_md_ratio(md_ratio);
}

void ContextNAFMdGradiPolicy::set_alpha_rate_limits(float rate_up, float rate_down) {
    policy_->set_alpha_rate_limits(rate_up, rate_down);
}

void ContextNAFMdGradiPolicy::set_physical_bounds(float mass_min, float mass_max, float alpha_min, float alpha_max) {
    policy_->set_physical_bounds(mass_min, mass_max, alpha_min, alpha_max);
}

std::vector<float> ContextNAFMdGradiPolicy::get_applied_mass_alpha() const {
    auto v = policy_->get_applied_mass_alpha();
    return {v[0], v[1]};
}

float ContextNAFMdGradiPolicy::get_applied_damping() const {
    return policy_->get_applied_damping();
}

std::vector<float> ContextNAFMdGradiPolicy::get_last_network_action() const {
    auto v = policy_->get_last_network_action();
    return {v[0], v[1]};
}

float ContextNAFMdGradiPolicy::get_last_alpha_cmd_abs() const {
    return policy_->get_last_alpha_cmd_abs();
}

float ContextNAFMdGradiPolicy::get_filtered_fext() const {
    return policy_->get_filtered_fext();
}

// =============================
// ForceCon1DMode5
// =============================

ForceCon1DMode5::ForceCon1DMode5(
    const std::string& model_path,
    double dt,
    int threads,
    const std::string& device,
    double md_ratio,
    double fc_fext,
    double free_mass,
    double free_damping,
    double free_stiffness,
    double contact_stiffness,
    double recovery_tau,
    std::vector<float> action_low,
    std::vector<float> action_high,
    float mass_min,
    float mass_max,
    float alpha_min,
    float alpha_max,
    float alpha_rate_up,
    float alpha_rate_down
)
: dt_(dt),
  md_ratio_(md_ratio),
  free_mass_(free_mass),
  free_damping_(free_damping),
  free_stiffness_(free_stiffness),
  contact_stiffness_(contact_stiffness),
  recovery_tau_(recovery_tau),
  current_mass_(free_mass),
  current_damping_(free_damping),
  current_stiffness_(free_stiffness),
  last_xc_(0.0),
  adm_(dt)
{
    policy_ = std::make_unique<RL_ContextNAF_mdGradi>(
        model_path,
        threads,
        parse_device(device),
        static_cast<float>(dt),
        static_cast<float>(md_ratio),
        static_cast<float>(fc_fext),
        vec2_to_array2(action_low, "action_low"),
        vec2_to_array2(action_high, "action_high"),
        mass_min,
        mass_max,
        alpha_min,
        alpha_max,
        alpha_rate_up,
        alpha_rate_down
    );

    adm_.adm_1D_MDK(current_mass_, current_damping_, current_stiffness_);
}

void ForceCon1DMode5::reset(double xd) {
    policy_->reset_state();
    current_mass_ = free_mass_;
    current_damping_ = free_damping_;
    current_stiffness_ = free_stiffness_;
    adm_.adm_1D_MDK(current_mass_, current_damping_, current_stiffness_);
    adm_.reset(xd);
    last_xc_ = xd;
}

void ForceCon1DMode5::set_free_mdk(double mass, double damping, double stiffness) {
    free_mass_ = mass;
    free_damping_ = damping;
    free_stiffness_ = stiffness;
}

void ForceCon1DMode5::set_contact_stiffness(double stiffness) {
    contact_stiffness_ = stiffness;
}

void ForceCon1DMode5::set_recovery_tau(double tau_sec) {
    recovery_tau_ = clamp_positive(tau_sec, 3.0);
}

void ForceCon1DMode5::set_md_ratio(double md_ratio) {
    md_ratio_ = clamp_positive(md_ratio, 1000.0);
    policy_->set_md_ratio(static_cast<float>(md_ratio_));
}

std::vector<double> ForceCon1DMode5::step(double xd, double x, double fd, double fext) {
    double applied_alpha = 0.0;

    constexpr double desired_force_threshold = 0.01;
    constexpr double actual_force_threshold = 1.50;
    constexpr double precontact_force_hold = 10.00;
    constexpr double return_tau = 0.20;

    const bool desired_force_active = std::fabs(fd) > desired_force_threshold;
    const bool actual_force_active = std::fabs(fext) > actual_force_threshold;
    const bool force_control_active = desired_force_active && actual_force_active;
    const double commanded_fd = (desired_force_active && !actual_force_active)
        ? signed_force_hold(fd, precontact_force_hold)
        : fd;

    if (force_control_active) {
        const std::vector<double> out = policy_->run(
            static_cast<float>(last_xc_),
            static_cast<float>(x),
            static_cast<float>(fd),
            static_cast<float>(fext)
        );

        if (out.size() < 2) {
            throw std::runtime_error("ForceCon1DMode5::step: policy output must contain [mass, alpha]");
        }

        current_mass_ = out[0];
        applied_alpha = out[1];
        current_damping_ = current_mass_ * applied_alpha * md_ratio_;
        current_stiffness_ = contact_stiffness_;
    } else {
        const double tau = clamp_positive(recovery_tau_, return_tau);
        const double alpha = exp_smoothing_alpha(dt_, tau);

        current_mass_      = current_mass_      + alpha * (free_mass_      - current_mass_);
        current_damping_   = current_damping_   + alpha * (free_damping_   - current_damping_);
        const double target_stiffness = desired_force_active ? 0.0 : free_stiffness_;
        current_stiffness_ = current_stiffness_ + alpha * (target_stiffness - current_stiffness_);

        if (current_mass_ > 1e-9 && md_ratio_ > 1e-9) {
            applied_alpha = current_damping_ / (current_mass_ * md_ratio_);
        } else {
            applied_alpha = 0.0;
        }
    }

    adm_.adm_1D_MDK(current_mass_, current_damping_, current_stiffness_);
    const double xc = adm_.adm_1D_control(xd, commanded_fd, fext);
    last_xc_ = xc;

    return {
        xc,
        current_mass_,
        applied_alpha,
        current_damping_,
        current_stiffness_,
        static_cast<double>(policy_->get_filtered_fext())
    };
}

std::vector<double> ForceCon1DMode5::get_current_mdk() const {
    return {current_mass_, current_damping_, current_stiffness_};
}

}  // namespace y2_control_pybind
