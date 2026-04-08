#pragma once
#include <stdio.h>

class Yadmittance_control
{
private:
    // State (history)
    double adm_1D_Perror[3] = {0.0, 0.0, 0.0};
    double adm_1D_Ferror[3] = {0.0, 0.0, 0.0};

    // Admittance parameters
    double adm_1D_M = 1.0;
    double adm_1D_D = 1.0;
    double adm_1D_K = 0.0;

    // Discretized coefficients
    double adm_1D_A = 0.0;
    double adm_1D_B = 0.0;
    double adm_1D_C = 1.0;

    double adm_1D_dt = 0.001;
    double xc = 0.0;

public:
    Yadmittance_control() = default;
    explicit Yadmittance_control(double sampling_time);
    ~Yadmittance_control() = default;

    // Set MDK parameters
    bool adm_1D_MDK(double Mass, double Damping, double Stiffness);

    // Monitor MDK
    double adm_MDK_monitor(int select);

    // Main admittance update
    double adm_1D_control(double xd, double Fd, double Fext);

    // ===== Reset interfaces =====

    // Soft reset: align internal state to current desired position
    void reset(double xd);

    // Hard reset: full zero initialization
    void hardReset();
};
