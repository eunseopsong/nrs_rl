// Experimental simulator binding. Production force-control defaults are untouched.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Y2ForceCon/mode3_force_control_core.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

template<std::size_t N>
std::array<double,N> checkedArray(const std::vector<double>& values) {
    if(values.size()!=N || !std::all_of(values.begin(),values.end(),[](double x){return std::isfinite(x);}))
        throw std::invalid_argument("Invalid control vector size or nonfinite input");
    std::array<double,N> out{};std::copy(values.begin(),values.end(),out.begin());return out;
}

class TangentialMode3 {
    std::unique_ptr<Mode3ForceControlCore> core_;
public:
    TangentialMode3(const std::string& model, double dt, int coordinate,
                   double desired_threshold, double actual_threshold,
                   double precontact_hold, double return_tau,
                   double damping, double stiffness) {
        if(!std::isfinite(damping) || !std::isfinite(stiffness)
           || damping<500. || damping>6000. || stiffness<2000. || stiffness>16000.)
            throw std::invalid_argument("Tangential MDK outside frozen experimental bounds");
        if(std::abs(dt-.008)>1e-9) throw std::invalid_argument("125 Hz control required");
        Mode3ForceControlConfig config;
        config.control_period=dt;config.force_control_coordinate=coordinate;
        config.desired_force_threshold=desired_threshold;
        config.actual_force_threshold=actual_threshold;
        config.precontact_force_hold=precontact_hold;
        config.return_tau_mass=config.return_tau_damping=config.return_tau_stiffness=return_tau;
        for(int axis=0;axis<2;++axis) {
            config.initial_damping[axis]=damping;
            config.initial_stiffness[axis]=stiffness;
        }
        core_=std::make_unique<Mode3ForceControlCore>(config,model,1,torch::kCPU);
    }
    void reset(const std::vector<double>& pose) {core_->reset(checkedArray<6>(pose));}
    std::vector<double> step(const std::vector<double>& measured,const std::vector<double>& reference,
                            const std::vector<double>& force,const std::vector<double>& wrench,
                            const std::vector<double>& rotation) {
        Mode3ForceControlInput in;
        in.measured_pose=checkedArray<6>(measured);in.reference_pose=checkedArray<6>(reference);
        in.desired_force=checkedArray<3>(force);in.wrench_base=checkedArray<6>(wrench);
        in.tcp_rotation=checkedArray<9>(rotation);
        const auto out=core_->step(in);
        std::vector<double> result;
        for(const auto* values:{&out.command_pose,&out.mass,&out.damping,&out.stiffness})
            result.insert(result.end(),values->begin(),values->end());
        return result;
    }
};

PYBIND11_MODULE(_tcp_calibrated_mode3,m) {
    pybind11::class_<TangentialMode3>(m,"TangentialMode3")
        .def(pybind11::init<const std::string&,double,int,double,double,double,double,double,double>())
        .def("reset",&TangentialMode3::reset).def("step",&TangentialMode3::step);
}
