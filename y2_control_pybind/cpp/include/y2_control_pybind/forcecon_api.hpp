#pragma once

#include "Y2ForceCon/admittance_control.hpp"
#include "Y2ForceCon/contextNaf_mdGradi.hpp"

#include <memory>
#include <string>
#include <vector>

namespace y2_control_pybind {

/**
 * Thin Python-friendly wrapper for Yadmittance_control
 */
class Admittance1D {
public:
    explicit Admittance1D(double dt = 0.002);

    void set_mdk(double mass, double damping, double stiffness);
    double monitor_mdk(int select) const;
    double step(double xd, double fd, double fext);

    void reset(double xd);
    void hard_reset();

private:
    Yadmittance_control adm_;
};

/**
 * Thin Python-friendly wrapper for RL_ContextNAF_mdGradi
 */
class ContextNAFMdGradiPolicy {
public:
    ContextNAFMdGradiPolicy(
        const std::string& model_path,
        int threads = 1,
        const std::string& device = "cpu",
        float dt = 0.002f,
        float md_ratio = 1000.0f,
        float fc_fext = 50.0f,
        std::vector<float> action_low = {-0.25f, -0.25f},
        std::vector<float> action_high = {+0.25f, +0.25f},
        float mass_min = 0.5f,
        float mass_max = 5.0f,
        float alpha_min = 0.5f,
        float alpha_max = 3.0f,
        float alpha_rate_up = 4.0f,
        float alpha_rate_down = 4.0f
    );

    std::vector<double> run(float xc, float x, float fd, float env_fext);

    void set_sampling_time(float dt);
    void reset_state();

    void set_md_ratio(float md_ratio);
    void set_alpha_rate_limits(float rate_up, float rate_down);
    void set_physical_bounds(float mass_min, float mass_max, float alpha_min, float alpha_max);

    std::vector<float> get_applied_mass_alpha() const;
    float get_applied_damping() const;
    std::vector<float> get_last_network_action() const;
    float get_last_alpha_cmd_abs() const;
    float get_filtered_fext() const;

private:
    std::unique_ptr<RL_ContextNAF_mdGradi> policy_;
};

/**
 * Composite 1D force controller for the latest Mode 5 logic:
 * RL_ContextNAF_mdGradi -> (mass, alpha) -> damping = mass * alpha * md_ratio
 * -> Yadmittance_control
 *
 * step(...) returns:
 * [xc, mass, alpha, damping, stiffness, filtered_fext]
 */
class ForceCon1DMode5 {
public:
    ForceCon1DMode5(
        const std::string& model_path,
        double dt = 0.002,
        int threads = 1,
        const std::string& device = "cpu",
        double md_ratio = 1000.0,
        double fc_fext = 50.0,
        double free_mass = 2.0,
        double free_damping = 6000.0,
        double free_stiffness = 2000.0,
        double contact_stiffness = 0.0,
        double recovery_tau = 3.0,
        std::vector<float> action_low = {-0.25f, -0.25f},
        std::vector<float> action_high = {+0.25f, +0.25f},
        float mass_min = 0.5f,
        float mass_max = 5.0f,
        float alpha_min = 0.5f,
        float alpha_max = 3.0f,
        float alpha_rate_up = 4.0f,
        float alpha_rate_down = 4.0f
    );

    void reset(double xd);

    void set_free_mdk(double mass, double damping, double stiffness);
    void set_contact_stiffness(double stiffness);
    void set_recovery_tau(double tau_sec);
    void set_md_ratio(double md_ratio);

    std::vector<double> step(double xd, double x, double fd, double fext);

    std::vector<double> get_current_mdk() const;

private:
    double dt_;
    double md_ratio_;

    double free_mass_;
    double free_damping_;
    double free_stiffness_;

    double contact_stiffness_;
    double recovery_tau_;

    double current_mass_;
    double current_damping_;
    double current_stiffness_;
    double last_xc_;

    std::unique_ptr<RL_ContextNAF_mdGradi> policy_;
    Yadmittance_control adm_;
};

}  // namespace y2_control_pybind
