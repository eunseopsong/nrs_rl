#include "Y2ForceCon/admittance_control.hpp"
#include <cmath>

// Constructor
Yadmittance_control::Yadmittance_control(double sampling_time)
{
    adm_1D_dt = sampling_time;
    hardReset();
}

bool Yadmittance_control::adm_1D_MDK(double Mass, double Damping, double Stiffness)
{
    adm_1D_M = Mass;
    adm_1D_D = Damping;
    adm_1D_K = Stiffness;
    return true;
}

double Yadmittance_control::adm_MDK_monitor(int select)
{
    /* 0:M, 1:D, 2:K */
    if      (select == 0) return adm_1D_M;
    else if (select == 1) return adm_1D_D;
    else if (select == 2) return adm_1D_K;
    else {
        printf("[Admittance] Wrong MDK monitor selection\n");
        return 0.0;
    }
}

double Yadmittance_control::adm_1D_control(double xd, double Fd, double Fext)
{
    // Force error
    adm_1D_Ferror[0] = Fd - Fext;

    // Discretized coefficients (Tustin)
    adm_1D_A = 4.0 * adm_1D_M
             - 2.0 * adm_1D_dt * adm_1D_D
             + adm_1D_dt * adm_1D_dt * adm_1D_K;

    adm_1D_B = 2.0 * adm_1D_dt * adm_1D_dt * adm_1D_K
             - 8.0 * adm_1D_M;

    adm_1D_C = 4.0 * adm_1D_M
             + 2.0 * adm_1D_dt * adm_1D_D
             + adm_1D_dt * adm_1D_dt * adm_1D_K;

    // Core difference equation
    xc = xd
       + (1.0 / adm_1D_C) *
         ( adm_1D_B * adm_1D_Perror[1]
         + adm_1D_A * adm_1D_Perror[2]
         - adm_1D_dt * adm_1D_dt *
           ( adm_1D_Ferror[0]
           + 2.0 * adm_1D_Ferror[1]
           + adm_1D_Ferror[2] ) );

    // Update history
    adm_1D_Perror[0] = xd - xc;
    adm_1D_Perror[2] = adm_1D_Perror[1];
    adm_1D_Perror[1] = adm_1D_Perror[0];

    adm_1D_Ferror[2] = adm_1D_Ferror[1];
    adm_1D_Ferror[1] = adm_1D_Ferror[0];

    return xc;
}

// ===== Reset implementations =====

// Soft reset: align output to desired position (no jump)
void Yadmittance_control::reset(double xd)
{
    xc = xd;

    for (int i = 0; i < 3; ++i) {
        adm_1D_Perror[i] = 0.0;
        adm_1D_Ferror[i] = 0.0;
    }
}

// Hard reset: full zero state
void Yadmittance_control::hardReset()
{
    xc = 0.0;

    for (int i = 0; i < 3; ++i) {
        adm_1D_Perror[i] = 0.0;
        adm_1D_Ferror[i] = 0.0;
    }

    adm_1D_A = 0.0;
    adm_1D_B = 0.0;
    adm_1D_C = 1.0;
}
