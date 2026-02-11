////////////////////////////////////////////////////////////
// Kinematics.cpp
// Kinematics functions
// Create on 2013. 12. 24
// See the header file for update information
////////////////////////////////////////////////////////////
////////////////// Ver 2.00 ////////////////////////////////
////////////////////////////////////////////////////////////

//////
#include <cmath>
#include <ctime>
#include <iostream>
#include "Arm_class.h"
#include "Kinematics.h"

Kinematic_func::Kinematic_func()
{
	// printf("============== Applying UR10e Arm kinematics =================\n");
}

// =================== Kinematics and Dynamic parameters independent ===================//

void Kinematic_func::iForwardK_P(VectorXd &q, Vector3d &x, double endlength)
{
	// Input = Joint Angle q, additional end length endlength
	// Output = Current Position x

	double d6a = d6 + endlength;

	s1 = sin(q(0));
	c1 = cos(q(0));
	s2 = sin(q(1));
	c2 = cos(q(1));
	s3 = sin(q(2));
	c3 = cos(q(2));
	s4 = sin(q(3));
	c4 = cos(q(3));
	s5 = sin(q(4));
	c5 = cos(q(4));
	s6 = sin(q(5));
	c6 = cos(q(5));
	
	s234 = sin(q(1) + q(2) + q(3));
	c234 = cos(q(1) + q(2) + q(3));

	x(0) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6a*(c1*c234-s1*s234)*s5)/2.0 + (d6a*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6a*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	x(1) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6a*(s1*c234+c1*s234)*s5)/2.0 + (d6a*(s1*c234-c1*s234)*s5)/2.0 + d6a*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	x(2) = (d1 + (d6a*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6a*(c234*c5+s234*s5))/2.0 - d5*c234);
}

void Kinematic_func::iForwardK_T(VectorXd &q, Matrix4d &T, double endlength) // revise on 2025.06.09 //// MatrixXd &T, double endlength)
{
	// Input = Joint Angle q, additional end length endlength
	// Output = Transformation Matrix T

	double d6a = d6 + endlength;

	s1 = sin(q(0));
	c1 = cos(q(0));
	s2 = sin(q(1));
	c2 = cos(q(1));
	s3 = sin(q(2));
	c3 = cos(q(2));
	s4 = sin(q(3));
	c4 = cos(q(3));
	s5 = sin(q(4));
	c5 = cos(q(4));
	s6 = sin(q(5));
	c6 = cos(q(5));
	
	s234 = sin(q(1) + q(2) + q(3));
	c234 = cos(q(1) + q(2) + q(3));

	T(0,0) = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0);
	T(1,0) = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0));
	T(2,0) = -((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6);
	T(3,0) = 0;

	T(0,1) = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0));
	T(1,1) = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0));
	T(2,1) = -(s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0);
	T(3,1) = 0;

	T(0,2) = -(((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0);
	T(1,2) = -(c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0);
	T(2,2) = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0);
	T(3,2) = 0;

	T(0,3) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	T(1,3) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	T(2,3) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
	T(3,3) = 1;
}


// =================== Kinematics and Dynamic parameters with Arm Class ===================//

void Kinematic_func::ForwardK_P(CArm *A)
{
	// Input = Current Joint Angle qc
	// Output = Current Position xc

	s1 = sin(A->qc(0));
	c1 = cos(A->qc(0));
	s2 = sin(A->qc(1));
	c2 = cos(A->qc(1));
	s3 = sin(A->qc(2));
	c3 = cos(A->qc(2));
	s4 = sin(A->qc(3));
	c4 = cos(A->qc(3));
	s5 = sin(A->qc(4));
	c5 = cos(A->qc(4));
	s6 = sin(A->qc(5));
	c6 = cos(A->qc(5));
	
	s234 = sin(A->qc(1) + A->qc(2) + A->qc(3));
	c234 = cos(A->qc(1) + A->qc(2) + A->qc(3));

	A->xc(0) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	A->xc(1) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	A->xc(2) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
}

void Kinematic_func::ForwardK_Td(CArm *A)
{
	// Input = Desire Joint Angle qd
	// Output = Desire Transform Matrix Td

	s1 = sin(A->qd(0));
	c1 = cos(A->qd(0));
	s2 = sin(A->qd(1));
	c2 = cos(A->qd(1));
	s3 = sin(A->qd(2));
	c3 = cos(A->qd(2));
	s4 = sin(A->qd(3));
	c4 = cos(A->qd(3));
	s5 = sin(A->qd(4));
	c5 = cos(A->qd(4));
	s6 = sin(A->qd(5));
	c6 = cos(A->qd(5));

	s234 = sin(A->qc(1) + A->qc(2) + A->qc(3));
	c234 = cos(A->qc(1) + A->qc(2) + A->qc(3));

	A->Td(0,0) = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0);
	A->Td(1,0) = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0));
	A->Td(2,0) = -((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6);
	A->Td(3,0) = 0;

	A->Td(0,1) = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0));
	A->Td(1,1) = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0));
	A->Td(2,1) = -(s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0);
	A->Td(3,1) = 0;

	A->Td(0,2) = -(((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0);
	A->Td(1,2) = -(c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0);
	A->Td(2,2) = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0);
	A->Td(3,2) = 0;

	A->Td(0,3) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	A->Td(1,3) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	A->Td(2,3) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
	A->Td(3,3) = 1;

	A->xd(0) = A->Td(0,3);
	A->xd(1) = A->Td(1,3);
	A->xd(2) = A->Td(2,3);
}

void Kinematic_func::ForwardK_T(CArm *A)
{
	// Input = Current Joint Angle qc
	// Output = Current Transform Matrix Tc, Current Position xc

	s1 = sin(A->qc(0));
	c1 = cos(A->qc(0));
	s2 = sin(A->qc(1));
	c2 = cos(A->qc(1));
	s3 = sin(A->qc(2));
	c3 = cos(A->qc(2));
	s4 = sin(A->qc(3));
	c4 = cos(A->qc(3));
	s5 = sin(A->qc(4));
	c5 = cos(A->qc(4));
	s6 = sin(A->qc(5));
	c6 = cos(A->qc(5));

	s23 = sin(A->qc(1) + A->qc(2));
	c23 = cos(A->qc(1) + A->qc(2));

	s34 = sin(A->qc(2) + A->qc(3));
	c34 = cos(A->qc(2) + A->qc(3));	
	
	s234 = sin(A->qc(1) + A->qc(2) + A->qc(3));
	c234 = cos(A->qc(1) + A->qc(2) + A->qc(3));

	A->Tc(0,0) = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0);
	A->Tc(1,0) = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0));
	A->Tc(2,0) = -((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6);
	A->Tc(3,0) = 0;

	A->Tc(0,1) = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0));
	A->Tc(1,1) = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0));
	A->Tc(2,1) = -(s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0);
	A->Tc(3,1) = 0;

	A->Tc(0,2) = -(((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0);
	A->Tc(1,2) = -(c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0);
	A->Tc(2,2) = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0);
	A->Tc(3,2) = 0;

	A->Tc(0,3) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	A->Tc(1,3) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	A->Tc(2,3) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
	A->Tc(3,3) = 1;

	A->xc(0) = A->Tc(0,3);
	A->xc(1) = A->Tc(1,3);
	A->xc(2) = A->Tc(2,3);
}

void Kinematic_func::Ycontact_ForwardK_T(CArm *A)
{
	// Input = Current Joint Angle qc
	// Output = Current Transform Matrix Tc, Current Position xc
	this->Ycontact_EE2TCP <<  1, 0, 0, this->Ycontact_TCP_pos[0],
						      0, 1, 0, this->Ycontact_TCP_pos[1],
						      0, 0, 1, this->Ycontact_TCP_pos[2],
							  0, 0, 0, 1;

	s1 = sin(A->qc(0));
	c1 = cos(A->qc(0));
	s2 = sin(A->qc(1));
	c2 = cos(A->qc(1));
	s3 = sin(A->qc(2));
	c3 = cos(A->qc(2));
	s4 = sin(A->qc(3));
	c4 = cos(A->qc(3));
	s5 = sin(A->qc(4));
	c5 = cos(A->qc(4));
	s6 = sin(A->qc(5));
	c6 = cos(A->qc(5));

	s23 = sin(A->qc(1) + A->qc(2));
	c23 = cos(A->qc(1) + A->qc(2));

	s34 = sin(A->qc(2) + A->qc(3));
	c34 = cos(A->qc(2) + A->qc(3));	
	
	s234 = sin(A->qc(1) + A->qc(2) + A->qc(3));
	c234 = cos(A->qc(1) + A->qc(2) + A->qc(3));

	A->Tc(0,0) = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0);
	A->Tc(1,0) = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0));
	A->Tc(2,0) = -((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6);
	A->Tc(3,0) = 0;

	A->Tc(0,1) = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0));
	A->Tc(1,1) = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0));
	A->Tc(2,1) = -(s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0);
	A->Tc(3,1) = 0;

	A->Tc(0,2) = -(((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0);
	A->Tc(1,2) = -(c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0);
	A->Tc(2,2) = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0);
	A->Tc(3,2) = 0;

	A->Tc(0,3) = -((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);
	A->Tc(1,3) = -((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);
	A->Tc(2,3) = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
	A->Tc(3,3) = 1;

	A->Tc = A->Tc*this->Ycontact_EE2TCP;

	A->xc(0) = A->Tc(0,3);
	A->xc(1) = A->Tc(1,3);
	A->xc(2) = A->Tc(2,3);
}

void Kinematic_func::Quaternion2Rotation(CArm *A)
{
	A->QuatM(0,0) = 1 - 2*A->Quat[1]*A->Quat[1] - 2*A->Quat[2]*A->Quat[2];
	A->QuatM(0,1) = 2*A->Quat[0]*A->Quat[1] - 2*A->Quat[3]*A->Quat[2];
	A->QuatM(0,2) = 2*A->Quat[0]*A->Quat[2] + 2*A->Quat[3]*A->Quat[1];

	A->QuatM(1,0) = 2*A->Quat[0]*A->Quat[1] + 2*A->Quat[3]*A->Quat[2]; 
	A->QuatM(1,1) = 1 - 2*A->Quat[0]*A->Quat[0] - 2*A->Quat[2]*A->Quat[2];
	A->QuatM(1,2) = 2*A->Quat[1]*A->Quat[2] - 2*A->Quat[3]*A->Quat[0];

	A->QuatM(2,0) = 2*A->Quat[0]*A->Quat[2] - 2*A->Quat[3]*A->Quat[1];
	A->QuatM(2,1) = 2*A->Quat[1]*A->Quat[2] + 2*A->Quat[3]*A->Quat[0];
	A->QuatM(2,2) = 1 - 2*A->Quat[0]*A->Quat[0] - 2*A->Quat[1]*A->Quat[1];

	A->QuatM4(0,0) = A->QuatM(0,0);
	A->QuatM4(0,1) = A->QuatM(0,1);
	A->QuatM4(0,2) = A->QuatM(0,2);

	A->QuatM4(1,0) = A->QuatM(1,0); 
	A->QuatM4(1,1) = A->QuatM(1,1);
	A->QuatM4(1,2) = A->QuatM(1,2);

	A->QuatM4(2,0) = A->QuatM(2,0);
	A->QuatM4(2,1) = A->QuatM(2,1);
	A->QuatM4(2,2) = A->QuatM(2,2);

	A->QuatM4(3,0) = 0;
	A->QuatM4(3,1) = 0;
	A->QuatM4(3,2) = 0;
	A->QuatM4(3,3) = 1;

}

void Kinematic_func::Rotation2EulerAngle(CArm *A)
{
	// Input = Current Rotation Matirx Tc
	// Output = Current Euler Angle thc

	/*A->thc(1) = atan2(-A->Tc(2,0), sqrt(A->Tc(0,0)*A->Tc(0,0)+A->Tc(1,0)*A->Tc(1,0)));
	float cb = cos(A->thc(1));
	A->thc(2) = atan2(A->Tc(1,0)/cb, A->Tc(0,0)/cb);
	A->thc(0) = atan2(A->Tc(2,1)/cb, A->Tc(2,2)/cb);
*/

	double orig,orig_PL,orig_MI;

	float sy = sqrt(A->Tc(0,0) * A->Tc(0,0) +  A->Tc(1,0) * A->Tc(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    if (!singular)
    {
        A->thc(0) = atan2(A->Tc(2,1) , A->Tc(2,2));
        A->thc(1) = atan2(-A->Tc(2,0), sy);
        A->thc(2) = atan2(A->Tc(1,0), A->Tc(0,0));
    }
    else
    {
        A->thc(0) = atan2(-A->Tc(1,2), A->Tc(1,1));
        A->thc(1) = atan2(-A->Tc(2,0), sy);
        A->thc(2) = 0;
    }

	// For continuous conversion
	/*Step1: Plus 2pi if minus (if the starting point is near of +-180 degree)*/
	if(A->thc(0) < 0) A->thc(0) += 2*PI;
	/*Step1 end*/

	if(A->R2E_init_flag == false)
	{
		A->pre_thc = A->thc;
		A->R2E_init_flag = true;
	}
	
	for(int i=0;i<3;i++)
	{
		/*Step2: find nearest angle*/
		orig = fabs(A->thc(i) - A->pre_thc(i));
		orig_PL = fabs(A->thc(i)+2*PI - A->pre_thc(i));
		orig_MI = fabs(A->thc(i)-2*PI - A->pre_thc(i));

		if((orig <= orig_PL) && (orig <= orig_MI)) A->thc(i) = A->thc(i); // if orig is smallest value
		else if((orig_PL <= orig) && (orig_PL <= orig_MI)) A->thc(i) = A->thc(i)+2*PI; // if orig_PL is smallest value
		else A->thc(i) = A->thc(i)-2*PI; // if orig_MI is smallest value

		/*Step2 end*/
	}

	// data backup
	A->pre_thc = A->thc;

}

void Kinematic_func::iRotation2EulerAngle(Matrix3d &R, Vector3d &th)
{
	// Input = Rotation Matirx R
	// Output = Euler Angle th

	th(1) = atan2(-R(2,0), sqrt(R(0,0)*R(0,0)+R(1,0)*R(1,0)));
	float cb = cos(th(1));
	th(2) = atan2(R(1,0)/cb, R(0,0)/cb);
	th(0) = atan2(R(2,1)/cb, R(2,2)/cb);
}

void Kinematic_func::EulerAngle2Rotation(Matrix3d &R, Vector3d &th)
{
	R(0,0) = cos(th(2))*cos(th(1));
	R(0,1) = -sin(th(2))*cos(th(0))+cos(th(2))*sin(th(1))*sin(th(0));
	R(0,2) = sin(th(2))*sin(th(0))+cos(th(2))*sin(th(1))*cos(th(0));

	R(1,0) = sin(th(2))*cos(th(1));
	R(1,1) = cos(th(2))*cos(th(0))+sin(th(2))*sin(th(1))*sin(th(0));
	R(1,2) = -cos(th(2))*sin(th(0))+sin(th(2))*sin(th(1))*cos(th(0));

	R(2,0) = -sin(th(1));
	R(2,1) = cos(th(1))*sin(th(0));
	R(2,2) = cos(th(1))*cos(th(0));
}

Vector3d Kinematic_func::VR_Rot2RPY(const Matrix3d& rotationMatrix) 
{
    Vector3d rpy;
	double orig,orig_PL,orig_MI;

    // Check for singularity at rpy(2) = +-pi/2
    if (rotationMatrix(2, 0) > 0.998) { // singularity at north pole
        rpy(0) = atan2(rotationMatrix(0, 1), rotationMatrix(1, 1)); // ROLL
        rpy(1) = M_PI / 2.0; // PITCH
        rpy(2) = 0; // YAW
    } else if (rotationMatrix(2, 0) < -0.998) { // singularity at south pole
        rpy(0) = atan2(rotationMatrix(0, 1), rotationMatrix(1, 1)); // ROLL
        rpy(1) = -M_PI / 2.0; // PITCH
        rpy(2) = 0; // YAW
    } else {
        rpy(0) = atan2(-rotationMatrix(1, 0), rotationMatrix(0, 0)); // ROLL
        rpy(1) = asin(rotationMatrix(2, 0)); // PITCH
        rpy(2) = atan2(-rotationMatrix(2, 1), rotationMatrix(2, 2)); // YAW
    }

	/*** For continuous conversion - UR10 ***/
	/*Step1: Plus 2pi if minus (if the starting point is near of +-180 degree)*/
	if(rpy(0) < 0) {rpy(0) += 2*PI;}
	/*Step1 end*/

	if(this->R2E_init_flag == false)
	{
		this->R2E_pre_rpy = rpy;
		this->R2E_init_flag = true;
	}
	
	for(int i=0;i<3;i++)
	{
		/*Step2: find nearest angle*/
		orig = fabs(rpy(i) - this->R2E_pre_rpy(i));
		orig_PL = fabs(rpy(i)+2*PI - this->R2E_pre_rpy(i));
		orig_MI = fabs(rpy(i)-2*PI - this->R2E_pre_rpy(i));

		if((orig <= orig_PL) && (orig <= orig_MI)) {rpy(i) = rpy(i);} // if orig is smallest value
		else if((orig_PL <= orig) && (orig_PL <= orig_MI)) {rpy(i) = rpy(i)+2*PI;} // if orig_PL is smallest value
		else {rpy(i) = rpy(i)-2*PI;} // if orig_MI is smallest value

		/*Step2 end*/
	}

	// data backup
	this->R2E_pre_rpy = rpy;

    return rpy;
}

void Kinematic_func::Rotation2RPY(CArm *A)
{
	A->rpyc(2) = atan2(A->Tc(1,0),A->Tc(0,0));
	float ca = cos(A->rpyc(2));
	float sa = sin(A->rpyc(2));
	A->rpyc(1) = atan2(-A->Tc(2,0),(A->Tc(0,0)*ca+A->Tc(1,0)*sa));
	A->rpyc(0) = atan2((-A->Tc(1,2)*ca-A->Tc(0,2)*sa),(A->Tc(1,1)*ca-A->Tc(0,1)*sa));
}

int Kinematic_func::sgn(double x)
{
	if(x>0) return 1;
	if(x<0) return -1;
	if(x==0) return 0;
}

int Kinematic_func::InverseK(CArm *qA)
{
	// Input = Desired Transform Matrix Td
	// Output = Joint Angle q
	
    int num_sols = 0;

	double T00 =  qA->Td(0,0);
	double T10 =  qA->Td(1,0); 
	double T20 =  qA->Td(2,0); 
	double T30 =  qA->Td(3,0); 

	double T01 =  qA->Td(0,1); 
	double T11 =  qA->Td(1,1); 
	double T21 =  qA->Td(2,1); 
	double T31 =  qA->Td(3,1); 

	double T02 =  qA->Td(0,2); 
	double T12 =  qA->Td(1,2); 
	double T22 =  qA->Td(2,2); 
	double T32 =  qA->Td(3,2); 

	double T03 =  qA->Td(0,3); 
	double T13 =  qA->Td(1,3); 
	double T23 =  qA->Td(2,3); 
	double T33 =  qA->Td(3,3); 

	
    ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
    double q1[2];
    {
      double A = d6*T12 - T13;
      double B = d6*T02 - T03;
      double R = A*A + B*B;
      if(fabs(A) < ZERO_THRESH) {
        double div;
        if(fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
          div = -SIGN(d4)*SIGN(B);
        else
          div = -d4/B;
        double arcsin = asin(div);
        if(fabs(arcsin) < ZERO_THRESH)
          arcsin = 0.0;
        if(arcsin < 0.0)
          q1[0] = arcsin + 2.0*PI;
        else
          q1[0] = arcsin;
        q1[1] = PI - arcsin;
      }
      else if(fabs(B) < ZERO_THRESH) {
        double div;
        if(fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
          div = SIGN(d4)*SIGN(A);
        else
          div = d4/A;
        double arccos = acos(div);
        q1[0] = arccos;
        q1[1] = 2.0*PI - arccos;
      }
      else if(d4*d4 > R) {
        return num_sols;
      }
      else {
        double arccos = acos(d4 / sqrt(R)) ;
        double arctan = atan2(-B, A);
        double pos = arccos + arctan;
        double neg = -arccos + arctan;
        if(fabs(pos) < ZERO_THRESH)
          pos = 0.0;
        if(fabs(neg) < ZERO_THRESH)
          neg = 0.0;
        if(pos >= 0.0)
          q1[0] = pos;
        else
          q1[0] = 2.0*PI + pos;
        if(neg >= 0.0)
          q1[1] = neg; 
        else
          q1[1] = 2.0*PI + neg;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    double q5[2][2];
    {
      for(int i=0;i<2;i++) {
        double numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
        double div;
        if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
          div = SIGN(numer) * SIGN(d6);
        else
          div = numer / d6;
        double arccos = acos(div);
        q5[i][0] = arccos;
        q5[i][1] = 2.0*PI - arccos;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    {
      for(int i=0;i<2;i++) {
        for(int j=0;j<2;j++) {
          double c1 = cos(q1[i]), s1 = sin(q1[i]);
          double c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);
          double q6;
          ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
          if(fabs(s5) < ZERO_THRESH)
            q6 = 0;
          else {
            q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
                       SIGN(s5)*(T00*s1 - T10*c1));
            if(fabs(q6) < ZERO_THRESH)
              q6 = 0.0;
            if(q6 < 0.0)
              q6 += 2.0*PI;
          }
          ////////////////////////////////////////////////////////////////////////////////

          double q2[2], q3[2], q4[2];
          ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
          double c6 = cos(q6), s6 = sin(q6);
          double x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
          double x04y = c5*(T20*c6 - T21*s6) - T22*s5;
          double p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
                        T03*c1 + T13*s1;
          double p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);

          double c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
          if(fabs(fabs(c3) - 1.0) < ZERO_THRESH)
            c3 = SIGN(c3);
          else if(fabs(c3) > 1.0) {
            // TODO NO SOLUTION
            continue;
          }
          double arccos = acos(c3);
          q3[0] = arccos;
          q3[1] = 2.0*PI - arccos;
          double denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
          double s3 = sin(arccos);
          double A = (a2 + a3*c3), B = a3*s3;
          q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
          q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
          double c23_0 = cos(q2[0]+q3[0]);
          double s23_0 = sin(q2[0]+q3[0]);
          double c23_1 = cos(q2[1]+q3[1]);
          double s23_1 = sin(q2[1]+q3[1]);
          q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
          q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
          ////////////////////////////////////////////////////////////////////////////////
          for(int k=0;k<2;k++) {
            if(fabs(q2[k]) < ZERO_THRESH)
              q2[k] = 0.0;
            else if(q2[k] < 0.0) q2[k] += 2.0*PI;
            if(fabs(q4[k]) < ZERO_THRESH)
              q4[k] = 0.0;
            else if(q4[k] < 0.0) q4[k] += 2.0*PI;

              qA->q(num_sols*6+0) = q1[i];    
	      qA->q(num_sols*6+1) = q2[k]; 
              qA->q(num_sols*6+2) = q3[k];    
              qA->q(num_sols*6+3) = q4[k]; 
              qA->q(num_sols*6+4) = q5[i][j]; 
              qA->q(num_sols*6+5) = q6;
              num_sols++;
          }
        }
      }
    }
    return num_sols;
}



int Kinematic_func::InverseK_min(CArm *A){

  int ret;
  if(ret=InverseK(A)){
    double minerrsum=100000000;
    int idx;
    for(int i=0;i<ret;i++){
      double errsum=0;
      for(int j=0;j<6;j++){
				double qother;
				if(A->q(i*6+j)>0)
					qother=A->q(i*6+j)-2*PI;
				else if(A->q(i*6+j)<0)
					qother=A->q(i*6+j)+2*PI;

        double err1=A->qc(j)-qother;
        double err=A->qc(j)-A->q(i*6+j);
				if((err1*err1)<(err*err)){
					A->q(i*6+j)=qother;
					err=err1;
				}
        errsum = errsum + (err*err);
        //errsum += err*err;
      }
      if(minerrsum>errsum){
        minerrsum=errsum;
        idx=i;
      }
    }
    for(int i=0;i<6;i++)
    A->qd(i)=A->q(idx*6+i); // output of inverse kinematics
  }
  return ret;
}

int Kinematic_func::Ycontact_InverseK(CArm *qA)
{
	// Input = Desired Transform Matrix Td
	// Output = Joint Angle q
	this->Ycontact_EE2TCP <<  1, 0, 0, this->Ycontact_TCP_pos[0],
						      0, 1, 0, this->Ycontact_TCP_pos[1],
						      0, 0, 1, this->Ycontact_TCP_pos[2],
							  0, 0, 0, 1;

	qA->Td = qA->Td*this->Ycontact_EE2TCP.inverse();

    int num_sols = 0;

	double T00 =  qA->Td(0,0);
	double T10 =  qA->Td(1,0); 
	double T20 =  qA->Td(2,0); 
	double T30 =  qA->Td(3,0); 

	double T01 =  qA->Td(0,1); 
	double T11 =  qA->Td(1,1); 
	double T21 =  qA->Td(2,1); 
	double T31 =  qA->Td(3,1); 

	double T02 =  qA->Td(0,2); 
	double T12 =  qA->Td(1,2); 
	double T22 =  qA->Td(2,2); 
	double T32 =  qA->Td(3,2); 

	double T03 =  qA->Td(0,3); 
	double T13 =  qA->Td(1,3); 
	double T23 =  qA->Td(2,3); 
	double T33 =  qA->Td(3,3); 

	
    ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
    double q1[2];
    {
      double A = d6*T12 - T13;
      double B = d6*T02 - T03;
      double R = A*A + B*B;
      if(fabs(A) < ZERO_THRESH) {
        double div;
        if(fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
          div = -SIGN(d4)*SIGN(B);
        else
          div = -d4/B;
        double arcsin = asin(div);
        if(fabs(arcsin) < ZERO_THRESH)
          arcsin = 0.0;
        if(arcsin < 0.0)
          q1[0] = arcsin + 2.0*PI;
        else
          q1[0] = arcsin;
        q1[1] = PI - arcsin;
      }
      else if(fabs(B) < ZERO_THRESH) {
        double div;
        if(fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
          div = SIGN(d4)*SIGN(A);
        else
          div = d4/A;
        double arccos = acos(div);
        q1[0] = arccos;
        q1[1] = 2.0*PI - arccos;
      }
      else if(d4*d4 > R) {
        return num_sols;
      }
      else {
        double arccos = acos(d4 / sqrt(R)) ;
        double arctan = atan2(-B, A);
        double pos = arccos + arctan;
        double neg = -arccos + arctan;
        if(fabs(pos) < ZERO_THRESH)
          pos = 0.0;
        if(fabs(neg) < ZERO_THRESH)
          neg = 0.0;
        if(pos >= 0.0)
          q1[0] = pos;
        else
          q1[0] = 2.0*PI + pos;
        if(neg >= 0.0)
          q1[1] = neg; 
        else
          q1[1] = 2.0*PI + neg;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    double q5[2][2];
    {
      for(int i=0;i<2;i++) {
        double numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
        double div;
        if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
          div = SIGN(numer) * SIGN(d6);
        else
          div = numer / d6;
        double arccos = acos(div);
        q5[i][0] = arccos;
        q5[i][1] = 2.0*PI - arccos;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    {
      for(int i=0;i<2;i++) {
        for(int j=0;j<2;j++) {
          double c1 = cos(q1[i]), s1 = sin(q1[i]);
          double c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);
          double q6;
          ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
          if(fabs(s5) < ZERO_THRESH)
            q6 = 0;
          else {
            q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
                       SIGN(s5)*(T00*s1 - T10*c1));
            if(fabs(q6) < ZERO_THRESH)
              q6 = 0.0;
            if(q6 < 0.0)
              q6 += 2.0*PI;
          }
          ////////////////////////////////////////////////////////////////////////////////

          double q2[2], q3[2], q4[2];
          ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
          double c6 = cos(q6), s6 = sin(q6);
          double x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
          double x04y = c5*(T20*c6 - T21*s6) - T22*s5;
          double p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
                        T03*c1 + T13*s1;
          double p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);

          double c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
          if(fabs(fabs(c3) - 1.0) < ZERO_THRESH)
            c3 = SIGN(c3);
          else if(fabs(c3) > 1.0) {
            // TODO NO SOLUTION
            continue;
          }
          double arccos = acos(c3);
          q3[0] = arccos;
          q3[1] = 2.0*PI - arccos;
          double denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
          double s3 = sin(arccos);
          double A = (a2 + a3*c3), B = a3*s3;
          q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
          q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
          double c23_0 = cos(q2[0]+q3[0]);
          double s23_0 = sin(q2[0]+q3[0]);
          double c23_1 = cos(q2[1]+q3[1]);
          double s23_1 = sin(q2[1]+q3[1]);
          q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
          q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
          ////////////////////////////////////////////////////////////////////////////////
          for(int k=0;k<2;k++) {
            if(fabs(q2[k]) < ZERO_THRESH)
              q2[k] = 0.0;
            else if(q2[k] < 0.0) q2[k] += 2.0*PI;
            if(fabs(q4[k]) < ZERO_THRESH)
              q4[k] = 0.0;
            else if(q4[k] < 0.0) q4[k] += 2.0*PI;

              qA->q(num_sols*6+0) = q1[i];    
	      qA->q(num_sols*6+1) = q2[k]; 
              qA->q(num_sols*6+2) = q3[k];    
              qA->q(num_sols*6+3) = q4[k]; 
              qA->q(num_sols*6+4) = q5[i][j]; 
              qA->q(num_sols*6+5) = q6;
              num_sols++;
          }
        }
      }
    }
    return num_sols;
}
int Kinematic_func::Ycontact_InverseK_min(CArm *A)
{
  int ret;
  if(ret=Ycontact_InverseK(A)){
    double minerrsum=100000000;
    int idx;
    for(int i=0;i<ret;i++){
      double errsum=0;
      for(int j=0;j<6;j++){
				double qother;
				if(A->q(i*6+j)>0)
					qother=A->q(i*6+j)-2*PI;
				else if(A->q(i*6+j)<0)
					qother=A->q(i*6+j)+2*PI;

        double err1=A->qc(j)-qother;
        double err=A->qc(j)-A->q(i*6+j);
				if((err1*err1)<(err*err)){
					A->q(i*6+j)=qother;
					err=err1;
				}
        errsum = errsum + (err*err);
        //errsum += err*err;
      }
      if(minerrsum>errsum){
        minerrsum=errsum;
        idx=i;
      }
    }
    for(int i=0;i<6;i++)
    A->qd(i)=A->q(idx*6+i); // output of inverse kinematics
  }
  return ret;
}

void Kinematic_func::Jacobian(CArm *A)
{	

	s1 = sin(A->qc(0));
	c1 = cos(A->qc(0));
	s2 = sin(A->qc(1));
	c2 = cos(A->qc(1));
	s3 = sin(A->qc(2));
	c3 = cos(A->qc(2));
	s4 = sin(A->qc(3));
	c4 = cos(A->qc(3));
	s5 = sin(A->qc(4));
	c5 = cos(A->qc(4));
	s6 = sin(A->qc(5));
	c6 = cos(A->qc(5));
	
	s234 = sin(A->qc(1) + A->qc(2) + A->qc(3));
	c234 = cos(A->qc(1) + A->qc(2) + A->qc(3));

	s23 = sin(A->qc(1) + A->qc(2));
	c23 = cos(A->qc(1) + A->qc(2));
	
	A->Jp(0,0) = c1*(d6*c5 + d4) + s1*(d6*s5*c234 - d5*s234 - a3*c23 - a2*c2);
	A->Jp(0,1) = c1*(d6*s234*s5 + d5*c234 - a3*s23 - a2*s2);
	A->Jp(0,2) = c1*(d6*s234*s5 + d5*c234 - a3*s23);
	A->Jp(0,3) = c1*(d6*s234*s5 + d5*c234);
	A->Jp(0,4) = -c1*d6*c234*c5 - s1*d6*s5;
	A->Jp(0,5) = 0;

	A->Jp(1,0) = s1*(d6*c5 + d4) + c1*(-d6*s5*c234 + d5*s234 + a3*c23 + a2*c2);
	A->Jp(1,1) = s1*(d6*s234*s5 + d5*c234 - a3*s23 - a2*s2);
	A->Jp(1,2) = s1*(d6*s234*s5 + d5*c234 - a3*s23);
	A->Jp(1,3) = s1*(d6*s234*s5 + d5*c234);
	A->Jp(1,4) = -s1*d6*c234*c5 + c1*d6*s5;
	A->Jp(1,5) = 0;

	A->Jp(2,0) = 0;
	A->Jp(2,1) = a3*c23 + a2*c2 + d5*s234 - d6*c234*s5;
	A->Jp(2,2) = a3*c23 + d5*s234 - d6*c234*s5;
	A->Jp(2,3) = d5*s234 - d6*c234*s5;
	A->Jp(2,4) = -d6*s234*c5;
	A->Jp(2,5) = 0;

	
	A->Jw(0,0) = 0;
	A->Jw(0,1) = s1;
	A->Jw(0,2) = s1;
	A->Jw(0,3) = s1;
	A->Jw(0,4) = c1*s234;
	A->Jw(0,5) = c5*s1 - c1*c234*s5;

	A->Jw(1,0) = 0;
	A->Jw(1,1) = -c1;
	A->Jw(1,2) = -c1;
	A->Jw(1,3) = -c1;
	A->Jw(1,4) = s1*s234;
	A->Jw(1,5) = -s1*c234*s5 - c1*c5;

	A->Jw(2,0) = 1;
	A->Jw(2,1) = 0;
	A->Jw(2,2) = 0;
	A->Jw(2,3) = 0;
	A->Jw(2,4) = -c234;
	A->Jw(2,5) = -s234*s5;

	for(int i=0;i<3;i++){
		for(int j=0;j<6;j++){
			A->J(i,j)=A->Jp(i,j);
			A->J(3+i,j)=A->Jw(i,j);
		}
	}
}



void Kinematic_func::Jacobian_p(CArm *A)
{
	s23 = sin(A->qc(1) + A->qc(2));
	c23 = cos(A->qc(1) + A->qc(2));
	
	A->Jp(0,0) = c1*(d6*c5 + d4) + s1*(d6*s5*c234 - d5*s234 - a3*c23 - a2*c2);
	A->Jp(0,1) = c1*(d6*s234*s5 + d5*c234 - a3*s23 - a2*s2);
	A->Jp(0,2) = c1*(d6*s234*s5 + d5*c234 - a3*s23);
	A->Jp(0,3) = c1*(d6*s234*s5 + d5*c234);
	A->Jp(0,4) = -c1*d6*c234*c5 - s1*d6*s5;
	A->Jp(0,5) = 0;

	A->Jp(1,0) = s1*(d6*c5 + d4) + c1*(-d6*s5*c234 + d5*s234 + a3*c23 + a2*c2);
	A->Jp(1,1) = s1*(d6*s234*s5 + d5*c234 - a3*s23 - a2*s2);
	A->Jp(1,2) = s1*(d6*s234*s5 + d5*c234 - a3*s23);
	A->Jp(1,3) = s1*(d6*s234*s5 + d5*c234);
	A->Jp(1,4) = -s1*d6*c234*c5 + c1*d6*s5;
	A->Jp(1,5) = 0;

	A->Jp(2,0) = 0;
	A->Jp(2,1) = a3*c23 + a2*c2 + d5*s234 - d6*c234*s5;
	A->Jp(2,2) = a3*c23 + d5*s234 - d6*c234*s5;
	A->Jp(2,3) = d5*s234 - d6*c234*s5;
	A->Jp(2,4) = -d6*s234*c5;
	A->Jp(2,5) = 0;
}

void Kinematic_func::Jacobian_w(CArm *A)
{
	A->Jw(0,0) = 0;
	A->Jw(0,1) = s1;
	A->Jw(0,2) = s1;
	A->Jw(0,3) = s1;
	A->Jw(0,4) = c1*s234;
	A->Jw(0,5) = c5*s1 - c1*c234*s5;

	A->Jw(1,0) = 0;
	A->Jw(1,1) = -c1;
	A->Jw(1,2) = -c1;
	A->Jw(1,3) = -c1;
	A->Jw(1,4) = s1*s234;
	A->Jw(1,5) = -s1*c234*s5 - c1*c5;

	A->Jw(2,0) = 1;
	A->Jw(2,1) = 0;
	A->Jw(2,2) = 0;
	A->Jw(2,3) = 0;
	A->Jw(2,4) = -c234;
	A->Jw(2,5) = -s234*s5;
}
	
	
	
Matrix3d Kinematic_func::RotZ(double th) // input unit : rad
{
	Matrix3d RotX_cal;

	RotX_cal << cos(th), -sin(th), 0,
	            sin(th),  cos(th), 0,
				0      ,        0, 1;

	return RotX_cal;
}

Matrix3d Kinematic_func::RotY(double th) // input unit : rad
{
	Matrix3d RotY_cal;

	RotY_cal <<  cos(th), 0, sin(th),
	                   0, 1,       0,
				-sin(th), 0, cos(th);
	return RotY_cal;
}

Matrix3d Kinematic_func::RotX(double th) // input unit : rad
{
	Matrix3d RotZ_cal;

	RotZ_cal << 1,        0,        0,
				0,  cos(th), -sin(th),
				0,  sin(th),  cos(th);

	return RotZ_cal;
}

#if 0 // previous angle-axis code
Matrix3d Kinematic_func::angle_axis_representation(Eigen::Vector3d rot_axis,double rot_angle)
{
	double v0;
	Eigen::Matrix3d pre_rot_mat = Eigen::MatrixXd::Zero(3,3);
	v0 = 1-cos(rot_angle);

	pre_rot_mat(0,0)= rot_axis(0)*rot_axis(0)*v0 + cos(rot_angle);
	pre_rot_mat(0,1)= rot_axis(0)*rot_axis(1)*v0 - rot_axis(2)*sin(rot_angle);
	pre_rot_mat(0,2)= rot_axis(0)*rot_axis(2)*v0 + rot_axis(1)*sin(rot_angle);

	pre_rot_mat(1,0)= rot_axis(0)*rot_axis(0)*v0 + rot_axis(2)*sin(rot_angle);
	pre_rot_mat(1,1)= rot_axis(0)*rot_axis(1)*v0 + cos(rot_angle);
	pre_rot_mat(1,2)= rot_axis(1)*rot_axis(2)*v0 - rot_axis(0)*sin(rot_angle);

	pre_rot_mat(2,0)= rot_axis(0)*rot_axis(2)*v0 - rot_axis(1)*sin(rot_angle);
	pre_rot_mat(2,1)= rot_axis(1)*rot_axis(2)*v0 + rot_axis(0)*sin(rot_angle);
	pre_rot_mat(2,2)= rot_axis(2)*rot_axis(2)*v0 + cos(rot_angle);

	return pre_rot_mat;
}
#endif

/*Modified code by GPT */
Eigen::Matrix3d Kinematic_func::angle_axis_representation(Eigen::Vector3d rot_axis, double rot_angle)
{
    Eigen::Matrix3d pre_rot_mat = Eigen::Matrix3d::Identity(); // Identity matrix initialization
    
    double c = cos(rot_angle);
    double s = sin(rot_angle);
    double t = 1 - c;
    
    double x = rot_axis(0);
    double y = rot_axis(1);
    double z = rot_axis(2);
    
    pre_rot_mat(0, 0) = t * x * x + c;
    pre_rot_mat(0, 1) = t * x * y - s * z;
    pre_rot_mat(0, 2) = t * x * z + s * y;
    
    pre_rot_mat(1, 0) = t * x * y + s * z;
    pre_rot_mat(1, 1) = t * y * y + c;
    pre_rot_mat(1, 2) = t * y * z - s * x;
    
    pre_rot_mat(2, 0) = t * x * z - s * y;
    pre_rot_mat(2, 1) = t * y * z + s * x;
    pre_rot_mat(2, 2) = t * z * z + c;
    
    return pre_rot_mat;
}

Matrix3d Kinematic_func::Qua2Rot(double w,double x, double y, double z)
{
	Matrix3d Rot_out;
	Rot_out(0,0) = (1 - 2*y*y - 2*z*z);
	Rot_out(0,1) = (2*x*y - 2*z*w);
	Rot_out(0,2) = (2*x*z + 2*y*w);

	Rot_out(1,0) = (2*x*y + 2*z*w);
	Rot_out(1,1) = (1 - 2*x*x - 2*z*z);
	Rot_out(1,2) = (2*y*z - 2*x*w);

	Rot_out(2,0) = (2*x*z - 2*y*w);
	Rot_out(2,1) = (2*y*z + 2*x*w);
	Rot_out(2,2) = (1 - 2*x*x - 2*y*y);

	return Rot_out;
}

Quaterniond Kinematic_func::Rot2Qua(const Matrix3d& rotationMatrix) {
	Quaterniond q(rotationMatrix);
	return q;
}