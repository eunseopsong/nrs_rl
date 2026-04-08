#pragma once

#include <torch/script.h>
#include <vector>
#include <string>
#include <array>

class RL_ContextNAF_mdGradi {
public:
    RL_ContextNAF_mdGradi(const std::string& model_path,
                         int threads,
                         c10::Device device,
                         float dt = 0.01f,
                         float md_ratio = 1000.0f,
                         float fc_fext = 50.0f,
                         // network action bounds for TorchScript output after tanh squash:
                         // action = [delta_mass_cmd, delta_alpha_cmd]
                         std::array<float,2> action_low = {-0.25f, -0.25f},
                         std::array<float,2> action_high = {+0.25f, +0.25f},
                         // physical applied parameter bounds
                         float mass_min = 0.5f,
                         float mass_max = 5.0f,
                         float alpha_min = 0.5f,
                         float alpha_max = 3.0f,
                         // alpha actuator rate limits [alpha unit / sec]
                         float alpha_rate_up = 4.0f,
                         float alpha_rate_down = 4.0f);

    // raw inputs
    // xc : command position (reference position)
    // x  : measured position
    // Fd : desired force
    // Env_Fext : measured external force (raw)
    //
    // returns: [mass_act, alpha_act]
    // (internally applies incremental commands, alpha target generation, and rate-limited alpha actuator)
    std::vector<double> run(float xc, float x, float Fd, float Env_Fext);

    void set_sampling_time(float dt);
    void reset_state();

    // optional setters (runtime tuning)
    void set_md_ratio(float md_ratio);
    void set_alpha_rate_limits(float rate_up, float rate_down);
    void set_physical_bounds(float mass_min, float mass_max, float alpha_min, float alpha_max);

    // getters for monitoring / integration
    std::array<float,2> get_applied_mass_alpha() const;   // [mass_act, alpha_act]
    float get_applied_damping() const;                    // D = mass_act * alpha_act * md_ratio
    std::array<float,2> get_last_network_action() const;  // [delta_mass_cmd, delta_alpha_cmd]
    float get_last_alpha_cmd_abs() const;                 // absolute alpha target before rate limiter
    float get_filtered_fext() const;

private:
    // ---- Torch ----
    c10::Device device_;
    torch::jit::script::Module module_;

    // ---- time / params ----
    float dt_ = 0.01f;
    float md_ratio_ = 1000.0f;
    float fc_fext_ = 50.0f;
    float alpha_lp_ = 0.0f;

    // network action bounds (delta commands)
    std::array<float,2> action_low_;
    std::array<float,2> action_high_;
    std::array<float,2> action_mid_;  // for fallback only (delta-space midpoint, typically 0,0)

    // physical applied bounds
    float mass_min_ = 0.5f;
    float mass_max_ = 5.0f;
    float alpha_min_ = 0.5f;
    float alpha_max_ = 3.0f;

    // alpha actuator rate-limited dynamics
    float alpha_rate_up_ = 4.0f;
    float alpha_rate_down_ = 4.0f;

    // ---- numerical derivative state (for s = [xc_ddot, x_dot, x_ddot, (Fd - Fext_filt)] ) ----
    bool  has1_ = false, has2_ = false;
    float xc_prev1_ = 0.0f, xc_prev2_ = 0.0f; // for xc_ddot (2nd difference)
    float x_prev1_  = 0.0f;                   // for x_dot
    float x_dot_prev1_ = 0.0f;                // for x_ddot

    // ---- residual bookkeeping (for q_t = [e_r, Δe_r, force_err_dot, mass_act, alpha_act]) ----
    float prev_er_ = 0.0f;
    float prev_ferr_ = 0.0f;          // force_err(t-1) = Fd - Fext_filt
    float md_prev_ = 0.0f;            // previous applied mass (context)
    float dd_prev_ = 0.0f;            // previous applied damping D (context)
    float fext_filt_prev_ = 0.0f;     // previous filtered Fext

    // ---- internal applied actuator states (current) ----
    float mass_act_ = 0.0f;
    float alpha_act_ = 0.0f;

    // ---- debug / monitoring ----
    std::array<float,2> last_a_net_{0.0f, 0.0f};   // [delta_mass_cmd, delta_alpha_cmd]
    float last_alpha_cmd_abs_ = 0.0f;              // absolute alpha target before rate limit

    // ---- GRU hidden state ----
    bool has_hidden_ = false;
    torch::Tensor h_;                 // [1,1,h_dim] on device_

private:
    void update_alpha_lp(float dt);
    torch::Tensor ensure_hidden();

    static float clampf(float v, float lo, float hi);
    static float safe_float(float v, float fallback = 0.0f);

    static std::array<float,2> squash_to_bounds_tanh(const std::array<float,2>& raw,
                                                     const std::array<float,2>& low,
                                                     const std::array<float,2>& high,
                                                     float temp = 1.0f);

    float lpf_update(float prev_filt, float raw) const;

    float update_mass_actuator_incremental(float mass_prev, float delta_mass_cmd) const;
    float alpha_cmd_from_increment(float alpha_prev, float delta_alpha_cmd) const;
    float update_alpha_actuator(float alpha_prev, float alpha_cmd_abs) const;
};