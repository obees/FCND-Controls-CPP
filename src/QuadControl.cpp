#include "Common.h"
#include "QuadControl.h"

#include "Utility/SimpleConfig.h"

#include "Utility/StringUtils.h"
#include "Trajectory.h"
#include "BaseController.h"
#include "Math/Mat3x3F.h"

#ifdef __PX4_NUTTX
#include <systemlib/param/param.h>
#endif

void QuadControl::Init()
{
  BaseController::Init();

  // variables needed for integral control
  integratedAltitudeError = 0;
    
#ifndef __PX4_NUTTX
  // Load params from simulator parameter system
  ParamsHandle config = SimpleConfig::GetInstance();
   
  // Load parameters (default to 0)
  kpPosXY = config->Get(_config+".kpPosXY", 0);
  kpPosZ = config->Get(_config + ".kpPosZ", 0);
  KiPosZ = config->Get(_config + ".KiPosZ", 0);
     
  kpVelXY = config->Get(_config + ".kpVelXY", 0);
  kpVelZ = config->Get(_config + ".kpVelZ", 0);

  kpBank = config->Get(_config + ".kpBank", 0);
  kpYaw = config->Get(_config + ".kpYaw", 0);

  kpPQR = config->Get(_config + ".kpPQR", V3F());

  maxDescentRate = config->Get(_config + ".maxDescentRate", 100);
  maxAscentRate = config->Get(_config + ".maxAscentRate", 100);
  maxSpeedXY = config->Get(_config + ".maxSpeedXY", 100);
  maxAccelXY = config->Get(_config + ".maxHorizAccel", 100);

  maxTiltAngle = config->Get(_config + ".maxTiltAngle", 100);

  minMotorThrust = config->Get(_config + ".minMotorThrust", 0);
  maxMotorThrust = config->Get(_config + ".maxMotorThrust", 100);
  //cx = config->Get(_config + ".cx", 0.f);
  //cy = config->Get(_config + ".cy", 0.f);
#else
  // load params from PX4 parameter system
  //TODO
  param_get(param_find("MC_PITCH_P"), &Kp_bank);
  param_get(param_find("MC_YAW_P"), &Kp_yaw);
#endif
}

VehicleCommand QuadControl::GenerateMotorCommands(float collThrustCmd, V3F momentCmd)
{
  // Convert a desired 3-axis moment and collective thrust command to 
  //   individual motor thrust commands
  // INPUTS: 
  //   desCollectiveThrust: desired collective thrust [N]
  //   desMoment: desired rotation moment about each axis [N m]
  // OUTPUT:
  //   set class member variable cmd (class variable for graphing) where
  //   cmd.desiredThrustsN[0..3]: motor commands, in [N]

  // HINTS: 
  // - you can access parts of desMoment via e.g. desMoment.x
  // You'll need the arm length parameter L, and the drag/thrust ratio kappa
    // kappa: torque (Nm) produced by motor per N of thrust produced

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
    
    // cmd.desiredThrustsN[0] is for front left
    // cmd.desiredThrustsN[1] is for front right
    // cmd.desiredThrustsN[2] is for rear left
    // cmd.desiredThrustsN[3] is for rear right
    // Drone motors ids
    // 1  2
    // 3  4

    // totalThrust = F1 + F2 + F3 + F4
    // T1 = -kappa * F1
    // T2 = kappa * F2
    // T3 = kappa * F3
    // T4 = -kappa * F4
    
    // Tx = (F1 + F3 - F2 - F4) * (SQRT(2)/2*L)
    // Ty = (F1 + F2 - F3 - F4) * (SQRT(2)/2*L)
    // Tz = T1 + T2 + T3 + T4
    
    // totalThrust        = F1 + F2 + F3 + F4      (L1)
    // Tx / (SQRT(2)/2*L) = F1 - F2 + F3 - F4      (L2)
    // Ty / (SQRT(2)/2*L) = F1 + F2 - F3 - F4      (L3)
    // Tz / -kappa        = F1 - F2 + F4 - F3      (L4)
    
    // 4 variables and 4 equations, there's a unique solution
    
    // F1 = totalThrust - F2 - F3 - F4
    
    // L1 + L2 + L3 + L4
    float F1 = (collThrustCmd + momentCmd.x / (sqrt(2)/2*L) + momentCmd.y / (sqrt(2)/2*L) + momentCmd.z / (-kappa)) / 4;
    
    // L1 + L4 with resolved F1
    float F4 = (collThrustCmd + momentCmd.z / (-kappa) - 2 * F1) / 2;
    
    // L1 + L2 with known F1
    float F3 = (collThrustCmd + momentCmd.x / (sqrt(2)/2*L) - 2 * F1) / 2;
    
    // From L1 with known F1 F3 F4
    float F2 = collThrustCmd - F1 - F3 - F4;

    cmd.desiredThrustsN[0] = CONSTRAIN(F1, minMotorThrust, maxMotorThrust);
    cmd.desiredThrustsN[1] = CONSTRAIN(F2, minMotorThrust, maxMotorThrust);
    cmd.desiredThrustsN[2] = CONSTRAIN(F3, minMotorThrust, maxMotorThrust);
    cmd.desiredThrustsN[3] = CONSTRAIN(F4, minMotorThrust, maxMotorThrust);

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return cmd;
}

V3F QuadControl::BodyRateControl(V3F pqrCmd, V3F pqr)
{
  // Calculate a desired 3-axis moment given a desired and current body rate
  // INPUTS: 
  //   pqrCmd: desired body rates [rad/s]
  //   pqr: current or estimated body rates [rad/s]
  // OUTPUT:
  //   return a V3F containing the desired moments for each of the 3 axes

  // HINTS: 
  //  - you can use V3Fs just like scalars: V3F a(1,1,1), b(2,3,4), c; c=a-b;
  //  - you'll need parameters for moments of inertia Ixx, Iyy, Izz
  //  - you'll also need the gain parameter kpPQR (it's a V3F)

  V3F momentCmd;

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

    // a proportional controller on body rates to commanded moments
    // kg * m^2 * rad / sec^2 = Newtons*meters

    // V3F structure used to store moments of inertia in every axis
    V3F moi = V3F(Ixx,Iyy,Izz);

    // kpPQR is a V3F used to store proportional gains on angular velocity on all axes
    // Calculation is based on Moment = (Moment of Inertia) x (Angular Acceleration)
    // Angular acceleration  equals kpPQR x (body rate desired - observed)
    momentCmd = kpPQR * moi * (pqrCmd - pqr);

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return momentCmd;
}

// returns a desired roll and pitch rate 
V3F QuadControl::RollPitchControl(V3F accelCmd, Quaternion<float> attitude, float collThrustCmd)
{
  // Calculate a desired pitch and roll angle rates based on a desired global
  //   lateral acceleration, the current attitude of the quad, and desired
  //   collective thrust command
  // INPUTS: 
  //   accelCmd: desired acceleration in global XY coordinates [m/s2]
  //   attitude: current or estimated attitude of the vehicle
  //   collThrustCmd: desired collective thrust of the quad [N]
  // OUTPUT:
  //   return a V3F containing the desired pitch and roll rates. The Z
  //     element of the V3F should be left at its default value (0)

  // HINTS: 
  //  - we already provide rotation matrix R: to get element R[1,2] (python) use R(1,2) (C++)
  //  - you'll need the roll/pitch gain kpBank
  //  - collThrustCmd is a force in Newtons! You'll likely want to convert it to acceleration first

  V3F pqrCmd;
  Mat3x3F R = attitude.RotationMatrix_IwrtB();

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

    // proportional controller
    
    float b_x_c, b_y_c;
    float b_x_c_dot, b_y_c_dot;
    float p_c, q_c;
    float roll, pitch;
    
    // Avoid division by zero
    if (collThrustCmd != 0){
        // Based on equation x_dot_dot = (1/m) * R[1,3] * (-F),
        // given
        // x_dot_dot = north_acceleration_cmd
        // F = thrust_cmd
        // m = DRONE_MASS_KG
        // R[1,3] = b_x_c, x scaling factor in inertial frame of the commanded acceleration (DRONE_MASS_KG/thrust_cmd) in body frame
        b_x_c = accelCmd[0] / -collThrustCmd * mass;
        
        // Same for b_y_c
        b_y_c = accelCmd[1] / -collThrustCmd * mass;
    }else{
        b_x_c = 0.f;
        b_y_c = 0.f;
    }

    // b_x_a is given by R(0,2)
    // rate of change of x scaling factor in inertial frame
    b_x_c_dot = kpBank * (b_x_c - R(0,2));
    
    // from b_y_c_dot = kpBank * (b_y_c - b_y_a);
    b_y_c_dot = kpBank * (b_y_c - R(1,2));
    
    // Calculation of p_c and q_c by hand instead of creating a 2x2 matrix based on first 4 elements of R
    p_c = (R(1,0) * b_x_c_dot - R(0,0) * b_y_c_dot) / R(2,2);
    q_c = (R(1,1) * b_x_c_dot - R(0,1) * b_y_c_dot) / R(2,2);

    
    // Get Euler roll and pitch from attitude quaternion
    roll = attitude.Roll();
    pitch = attitude.Pitch();
    
    // making sure we are not going over the roll limits
    if (roll >= maxTiltAngle && p_c > 0){
        p_c = -M_PI/4;
    } else if (roll < (-maxTiltAngle) && p_c < 0){
        p_c = M_PI/4;
    }
    
    // making sure we are not going over the pitch limits
    if (pitch >= maxTiltAngle && q_c > 0){
        q_c = -M_PI/4;
    } else if (pitch < (-maxTiltAngle) && q_c < 0){
        q_c = M_PI/4;
    }
    
    // populate the return variable with p_c and q_c
    pqrCmd.x = p_c;
    pqrCmd.y = q_c;
    

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return pqrCmd;
}

float QuadControl::AltitudeControl(float posZCmd, float velZCmd, float posZ, float velZ, Quaternion<float> attitude, float accelZCmd, float dt)
{
  // Calculate desired quad thrust based on altitude setpoint, actual altitude,
  //   vertical velocity setpoint, actual vertical velocity, and a vertical 
  //   acceleration feed-forward command
  // INPUTS: 
  //   posZCmd, velZCmd: desired vertical position and velocity in NED [m]
  //   posZ, velZ: current vertical position and velocity in NED [m]
  //   accelZCmd: feed-forward vertical acceleration in NED [m/s2]
  //   dt: the time step of the measurements [seconds]
  // OUTPUT:
  //   return a collective thrust command in [N]

  // HINTS: 
  //  - we already provide rotation matrix R: to get element R[1,2] (python) use R(1,2) (C++)
  //  - you'll need the gain parameters kpPosZ and kpVelZ
  //  - maxAscentRate and maxDescentRate are maximum vertical speeds. Note they're both >=0!
  //  - make sure to return a force, not an acceleration
  //  - remember that for an upright quad in NED, thrust should be HIGHER if the desired Z acceleration is LOWER

  Mat3x3F R = attitude.RotationMatrix_IwrtB();
  float thrust = 0;
  
  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

    // PID Controller
    
    float u_1_bar, c;

    // Ensure z velocity command is within limits
    if ( -velZCmd > maxAscentRate){
        velZCmd = -maxAscentRate;
    } else if (velZCmd > maxDescentRate){
        velZCmd = maxDescentRate;
    }
   
    // Error integration part of the PID
    integratedAltitudeError = integratedAltitudeError + (posZCmd - posZ) * dt;
    
    // desired vertical acceleration in NED frame is given by the following PD controller formula
    u_1_bar = kpPosZ * (posZCmd - posZ) + kpVelZ * ( velZCmd - velZ) + accelZCmd + KiPosZ * integratedAltitudeError;
    
    // desired vertical acceleration in body frame
    c = -(u_1_bar - CONST_GRAVITY) / R(2,2);
    
    // Thrust relies on F=ma equation
    thrust = mass * c;


  /////////////////////////////// END STUDENT CODE ////////////////////////////
  
  return thrust;
}

// returns a desired acceleration in global frame
V3F QuadControl::LateralPositionControl(V3F posCmd, V3F velCmd, V3F pos, V3F vel, V3F accelCmdFF)
{
  // Calculate a desired horizontal acceleration based on 
  //  desired lateral position/velocity/acceleration and current pose
  // INPUTS: 
  //   posCmd: desired position, in NED [m]
  //   velCmd: desired velocity, in NED [m/s]
  //   pos: current position, NED [m]
  //   vel: current velocity, NED [m/s]
  //   accelCmdFF: feed-forward acceleration, NED [m/s2]
  // OUTPUT:
  //   return a V3F with desired horizontal accelerations. 
  //     the Z component should be 0
  // HINTS: 
  //  - use the gain parameters kpPosXY and kpVelXY
  //  - make sure you limit the maximum horizontal velocity and acceleration
  //    to maxSpeedXY and maxAccelXY

  // make sure we don't have any incoming z-component
  accelCmdFF.z = 0;
  velCmd.z = 0;
  posCmd.z = pos.z;

  // we initialize the returned desired acceleration to the feed-forward value.
  // Make sure to _add_, not simply replace, the result of your controller
  // to this variable
  V3F accelCmd = accelCmdFF;

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

    // PD controller
    
    // Ensure x-y plan velocities are within limits
    velCmd.x = CONSTRAIN(velCmd.x, -maxSpeedXY, maxSpeedXY);
    velCmd.y = CONSTRAIN(velCmd.y, -maxSpeedXY, maxSpeedXY);
    
    // PD controller formula to calculate desired x acceleration in NED
    accelCmd.x = kpPosXY * (posCmd.x - pos.x) + kpVelXY * (velCmd.x - vel.x) + accelCmd.x;
    
    // PD controller formula to calculate desired y acceleration in NED
    accelCmd.y = kpPosXY * (posCmd.y - pos.y) + kpVelXY * (velCmd.y - vel.y) + accelCmd.y;
    
    // Ensure x-y plan accelerations are within limits
    accelCmd.x = CONSTRAIN(accelCmd.x, -maxAccelXY, maxAccelXY);
    accelCmd.y = CONSTRAIN(accelCmd.y, -maxAccelXY, maxAccelXY);

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return accelCmd;
}

// returns desired yaw rate
float QuadControl::YawControl(float yawCmd, float yaw)
{
  // Calculate a desired yaw rate to control yaw to yawCmd
  // INPUTS: 
  //   yawCmd: commanded yaw [rad]
  //   yaw: current yaw [rad]
  // OUTPUT:
  //   return a desired yaw rate [rad/s]
  // HINTS: 
  //  - use fmodf(foo,b) to unwrap a radian angle measure float foo to range [0,b]. 
  //  - use the yaw control gain parameter kpYaw

  float yawRateCmd=0;
  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
    
    // proportional controller

    float yawDelta;

    // calculate yaw delta between commanded and observed yaw values
    yawDelta = yawCmd - yaw;

    // Ensure you always travel the smallest angle from observed yaw to commanded yaw
    if (fabsf(yawDelta) > M_PI){
        yawDelta = -fmodf(yawDelta,M_PI);
    }
    
    // Calculate yaw rate with the proportional yaw constant
    yawRateCmd = kpYaw * yawDelta;
    
  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return yawRateCmd;

}

VehicleCommand QuadControl::RunControl(float dt, float simTime)
{
  curTrajPoint = GetNextTrajectoryPoint(simTime);

  float collThrustCmd = AltitudeControl(curTrajPoint.position.z, curTrajPoint.velocity.z, estPos.z, estVel.z, estAtt, curTrajPoint.accel.z, dt);

  // reserve some thrust margin for angle control
  float thrustMargin = .1f*(maxMotorThrust - minMotorThrust);
  collThrustCmd = CONSTRAIN(collThrustCmd, (minMotorThrust+ thrustMargin)*4.f, (maxMotorThrust-thrustMargin)*4.f);
  
  V3F desAcc = LateralPositionControl(curTrajPoint.position, curTrajPoint.velocity, estPos, estVel, curTrajPoint.accel);
  
  V3F desOmega = RollPitchControl(desAcc, estAtt, collThrustCmd);
  desOmega.z = YawControl(curTrajPoint.attitude.Yaw(), estAtt.Yaw());

  V3F desMoment = BodyRateControl(desOmega, estOmega);

  return GenerateMotorCommands(collThrustCmd, desMoment);
}
