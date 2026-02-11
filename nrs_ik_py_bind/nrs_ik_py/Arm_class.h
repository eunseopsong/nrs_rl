////////////////////////////////////////////////////////////
// Arm_class.h
// contains variables for arm
// created by gitae on 2016.1.30
// ref) Finger_class by gunkyu
////////////////////////////////////////////////////////////
//////////////////// Ver 1.00 ////////////////////////////////
////////////////////////////////////////////////////////////

#ifndef __ARMCLASS__
#define __ARMCLASS__

#include <stdio.h>
#include <Eigen/Dense>


#define PI M_PI
#define _pi 3.1415926535f

#define ROBOT 1 // 0: UR10e, 1: UR10

using namespace Eigen;


#if ROBOT == 0 // UR10e kinematic values
const double d1 = 0.1807;
const double a2 = -0.6127;
const double a3 = -0.57155;
const double d4 = 0.17415;
const double d5 = 0.11985;
const double d6 = 0.11655; //basic

// magnitude of center of mass
const double m1 = 7.369;
const double m2 = 13.051;
const double m3 = 3.989;
const double m4 = 2.1;
const double m5 = 1.98;
const double m6 = 0.615;

// position of center of mass
const double Pc1x = 0.021;
const double Pc1y = 0;
const double Pc1z = 0.027;
const double Pc2x = 0.38;
const double Pc2y = 0;
const double Pc2z = 0.158;
const double Pc3x = 0.24;
const double Pc3y = 0;
const double Pc3z = 0.068;
const double Pc4x = 0;
const double Pc4y = 0.007;
const double Pc4z = 0.018;
const double Pc5x = 0;
const double Pc5y = 0.007;
const double Pc5z = 0.018;
const double Pc6x = 0;
const double Pc6y = 0;
const double Pc6z = -0.026;

#elif ROBOT == 1 // UR10 kinematic values 
const double d1 = 0.1273;
const double a2 = -0.612;
const double a3 = -0.5723;
const double d4 = 0.163941;
const double d5 = 0.1157;
const double d6 = 0.0922; //basic

// magnitude of center of mass
const double m1 = 7.1;
const double m2 = 12.7;
const double m3 = 4.27;
const double m4 = 2.0;
const double m5 = 2.0;
const double m6 = 0.365;

// position of center of mass
const double Pc1x = 0.021;
const double Pc1y = 0;
const double Pc1z = 0.027;
const double Pc2x = 0.38;
const double Pc2y = 0;
const double Pc2z = 0.158;
const double Pc3x = 0.24;
const double Pc3y = 0;
const double Pc3z = 0.068;
const double Pc4x = 0;
const double Pc4y = 0.007;
const double Pc4z = 0.018;
const double Pc5x = 0;
const double Pc5y = 0.007;
const double Pc5z = 0.018;
const double Pc6x = 0;
const double Pc6y = 0;
const double Pc6z = -0.026;

#endif


typedef class Arm_class
{
private:



public:
	//T matrix
	Matrix4d Tc;
	Matrix4d Td;
	Matrix4d Tt;

	//R matrix
	Matrix3d Rc;
	Matrix3d Rc_i;
	Matrix3d Rt;
	Matrix3d Rd;
	Matrix3d Rfix;

	//Jacobian
	MatrixXd Jp;
	MatrixXd Jw;
	MatrixXd J;
	MatrixXd J_t;
	MatrixXd J_i;

	//joint
	VectorXd q;
	VectorXd qt;
	VectorXd qc;
	VectorXd qd;
	VectorXd dqd;
	VectorXd dqc;
	VectorXd ddqd;
	VectorXd ddqc;
	VectorXd qpoints[20];
	
	//cartesian
	VectorXd Xt;
	VectorXd Xc;
	VectorXd Xd;
	VectorXd Vc;
	VectorXd Vd;

	//cartesian pos
	Vector3d xc;
	Vector3d xd;
	Vector3d xt;
	Vector3d dx;
	Vector3d dxc;
	Vector3d dxd;
	Vector3d dxd_base;

	//cartesina ori
	bool R2E_init_flag = false;
	Vector3d pre_thc = Vector3d::Zero(3);
	Vector3d thc;
	Vector3d thd;
	Vector3d tht;
	Vector3d dthc;
	Vector3d dthd;
	Vector3d dthd_base;
	Vector3d dth;
	Vector3d rpyc;

	//quaternion
	VectorXd Quat;
	Matrix3d QuatM;
	MatrixXd QuatM4;
	
	Vector3d fc;
	Vector3d fd;
	Vector3d td;
	Vector3d Fbase;
	Vector3d Tbase;
	VectorXd FMc;

	VectorXd Traj_seq;

public:
	Arm_class()
	{
		Jp.resize(3,6);
		Jw.resize(3,6);
		J.resize(6,6);
		J_t.resize(6,6);
		J_i.resize(6,6);

		q.resize(100);
		qt.resize(6);
		qc.resize(6);
		qd.resize(6);
		dqd.resize(6);
		dqc.resize(6);
		ddqd.resize(6);
		ddqc.resize(6);
		qpoints[20].resize(6);

		Quat.resize(4);
		QuatM4.resize(4,4);

		Xt.resize(6);
		Xc.resize(6);
		Xd.resize(6);
		Vd.resize(6);

		FMc.resize(6);

		Traj_seq.resize(6);

	}

} CArm;


#endif
