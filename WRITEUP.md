# Building a Controller in python and c++

## Python
### body rate control
#### Specifications
- The controller should be a proportional controller on body rates to commanded moments
- The controller should take into account the moments of inertia of the drone when calculating the commanded moments
#### body rate control details
- Purpose is to calculate the commanded moment resulting from the gap between the desired and the observed body rates which are given as inputs

Commented code:
```
def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            attitude: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        # a proportional controller on body rates to commanded moments
        # kg * m^2 * rad / sec^2 = Newtons*meters

        # Only proportional gain used, no derivative term
        gains = np.array([self.k_p_p, self.k_p_q, self.k_p_r])

        # Calculation is based on Moment = (Moment of Inertia) x (Angular Acceleration)
        # Angular acceleration  equals (Gain) x (body rate desired - observed)
        [tau_x_c, tau_y_c, tau_z_c] = MOI.T * (gains.T * (body_rate_cmd.T - body_rate.T))

        # Maximum commanded moments are limited to MAX_TORQUE
        tau_x_c_c = np.clip(tau_x_c, -MAX_TORQUE, MAX_TORQUE)
        tau_y_c_c = np.clip(tau_y_c, -MAX_TORQUE, MAX_TORQUE)
        tau_z_c_c = np.clip(tau_z_c, -MAX_TORQUE, MAX_TORQUE)

        return np.array([tau_x_c_c, tau_y_c_c, tau_z_c_c])
```


### roll pitch control
#### Specifications
- The controller should use the acceleration and thrust commands, in addition to the vehicle attitude to output a body rate command
- The controller should account for the non-linear transformation from local accelerations to body rates
- Note that the drone's mass should be accounted for when calculating the target angles
#### roll pitch control details
- Purpose is to calculate the desired rollrate and pitchrate from commanded acceleration, commanded thrust and observed attitude

Commented code:
```
def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """

        # Extract variable values from parameters
        [north_acceleration_cmd, east_acceleration_cmd] = acceleration_cmd
        [roll, pitch, yaw] = attitude

        # Avoid division by zero
        if thrust_cmd != 0:

            # Based on equation x_dot_dot = (1/m) * R[1,3] * (-F),
            # given
            # x_dot_dot = north_acceleration_cmd
            # F = thrust_cmd
            # m = DRONE_MASS_KG
            # R[1,3] = b_x_c, x scaling factor in inertial frame of the commanded acceleration (DRONE_MASS_KG/thrust_cmd) in body frame
            b_x_c = north_acceleration_cmd / -thrust_cmd * DRONE_MASS_KG

            # Same for b_y_c
            b_y_c = east_acceleration_cmd / -thrust_cmd * DRONE_MASS_KG
        else:
            b_x_c = 0
            b_y_c = 0

        # Rotation matrix To transform between body-frame accelerations and world frame accelerations
        rot_mat = euler2RM(roll, pitch, yaw)

        # x scaling factor in inertial frame from actual attitude
        b_x_a = rot_mat[0][2]

        # y scaling factor in inertial frame from actual attitude
        b_y_a = rot_mat[1][2]

        # rate of change of x scaling factor in inertial frame
        b_x_c_dot = self.k_p_roll * (b_x_c - b_x_a)

        # rate of change of y scaling factor in inertial frame
        b_y_c_dot = self.k_p_pitch * (b_y_c - b_y_a)

        # creation of rate of change vector from its x and y components
        b_dot_vector = np.array([b_x_c_dot, b_y_c_dot]).T

        # sub rotation matrix containing the first 4 cells
        sub_rot_mat = np.zeros([2, 2])
        sub_rot_mat[0][0] = rot_mat[1][0]
        sub_rot_mat[1][0] = rot_mat[1][1]

        sub_rot_mat[0][1] = -rot_mat[0][0]
        sub_rot_mat[1][1] = -rot_mat[0][1]

        b_rot = (1 / rot_mat[2][2]) * sub_rot_mat

        # Calculation of p_c and  q_c
        [p_c, q_c] = b_rot.dot(b_dot_vector)

        # making sure we are not going over the roll limits
        if roll > (math.pi / 4) and p_c > 0:
            p_c = -math.pi/3

        elif roll < (-math.pi / 4) and p_c < 0:
            p_c = math.pi/3

        # making sure we are not going over the pitch limits
        if pitch > (math.pi / 4) and q_c > 0:
            q_c = -math.pi/3

        elif pitch < (-math.pi / 4) and q_c < 0:
            q_c = math.pi/3

        return np.array([p_c, q_c])
```

### altitude control
#### Specifications
- The controller should use both the down position and the down velocity to command thrust
- Ensure that the output value is indeed thrust (the drone's mass needs to be accounted for)
- Ensure that the thrust includes the non-linear effects from non-zero roll/pitch angles
#### altitude control details
- Purpose is to calculate the commanded thrust (in Newton) from the given parameters the altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff

Commented code
```
    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """

        # Extract roll, pitch, yaw variables from attitude parameter
        [roll, pitch, yaw] = attitude

        # calculate rotation matrix from roll pitch yaw
        rot_mat = euler2RM(roll, pitch, yaw)

        # desired vertical acceleration in inertial frame is given by the following PD controller formula
        u_1_bar = self.z_k_p * (altitude_cmd - altitude) + self.z_k_d * (vertical_velocity_cmd - vertical_velocity) + acceleration_ff

        # calculate vertical acceleration in body frame, rot_mat[2][2] is z scaling factor in inertial frame
        c = (u_1_bar - 9.81) / rot_mat[2][2]

        # since thrust is in Newton, from F=ma we need to multiply by drone's mass
        thrust = DRONE_MASS_KG * c

        # ensure thrust stays below the MAX_THRUST limit
        thrust_limited = np.clip(thrust, 0.0, MAX_THRUST)

        return thrust_limited  # thrust command in Newton
```
### lateral position control
#### Specifications
- The controller should use the local NE position and velocity to generate a commanded local acceleration
#### lateral position control details
- Purpose is to calculate horizontal acceleration commands on north and east axis from local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff parameters
- This will complete the thrust command to cover all 3 linear commanded linear accelerations

Commented code
```
    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """

        # PD controller

        # extract x-north axis variables from parameters
        x_north_cmd = local_position_cmd[0]
        x_north = local_position[0]
        x_dot_north_cmd = local_velocity_cmd[0]
        x_dot_north = local_velocity[0]
        x_dot_dot_north_ff = acceleration_ff[0]

        # extract y-east axis variables from parameters
        y_east_cmd = local_position_cmd[1]
        y_east = local_position[1]
        y_dot_east_cmd = local_velocity_cmd[1]
        y_dot_east = local_velocity[1]
        y_dot_dot_east_ff = acceleration_ff[1]

        # PD controller formula to calculate desirednorth acceleration in local frame
        acc_north_cmd = self.x_k_p * (x_north_cmd - x_north) + self.x_k_d * (x_dot_north_cmd - x_dot_north) + x_dot_dot_north_ff

        # PD controller formula to calculate desired east acceleration in local frame
        acc_east_cmd = self.y_k_p * (y_east_cmd - y_east) + self.y_k_d * (y_dot_east_cmd - y_dot_east) + y_dot_dot_east_ff

        return np.array([acc_north_cmd, acc_east_cmd])
```
### yaw control
#### Specifications
- The controller can be a linear/proportional heading controller to yaw rate commands (non-linear transformation not required)
#### yaw details
- Purpose is to calculate the desired yaw rate from the commanded yaw and the yaw parameters

Commented code
```
def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        # proportional controller

        # calculate yaw delta between commanded and observed yaw values
        yaw_delta = yaw_cmd - yaw

        # create new yaw delta variable to store temp values
        yaw_delta_2 = yaw_delta

        # Ensure you always pick the shortest angle towards commanded yaw from observed yaw
        if abs(yaw_delta) > (math.pi):
            if yaw_delta < 0:
                yaw_delta_2 = (2 * math.pi) + yaw_delta
            else:
                yaw_delta_2 = yaw_delta - (2 * math.pi)

        # Calculate yaw rate with the proportional yaw constant
        yawrate = self.k_p_yaw * (yaw_delta_2)

        return yawrate
```
### flight performance metrics
#### Specifications
- Ensure that the drone looks stable and performs the required task
#### details
- Purpose is follow the path correctly

#### Results
- Constraints are respected, and the drone flies it's path


## C++
### body rate control
#### Specifications
- The controller should be a proportional controller on body rates to commanded moments
- The controller should take into account the moments of inertia of the drone when calculating the commanded moments
#### body rate control details
- Purpose is to calculate the commanded moment resulting from the gap between the desired and the observed body rates which are given as inputs
```
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
```

### roll pitch control
#### Specifications
- The controller should use the acceleration and thrust commands, in addition to the vehicle attitude to output a body rate command
- The controller should account for the non-linear transformation from local accelerations to body rates
- Note that the drone's mass should be accounted for when calculating the target angles
- Additionally, the C++ altitude controller should contain an integrator to handle the weight non-idealities presented in scenario 4
#### roll pitch control details
- Purpose is to calculate a desired pitch and roll angle rates based on a desired global lateral acceleration, the current attitude of the quad, and desired collective thrust command

Commented code
```
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
```
### altitude Controller
#### Specifications
- The controller should use both the down position and the down velocity to command thrust
- Ensure that the output value is indeed thrust (the drone's mass needs to be accounted for)
- Ensure that the thrust includes the non-linear effects from non-zero roll/pitch angles
#### altitude control details
- Purpose is to calculate desired quad thrust based on altitude setpoint, actual altitude, vertical velocity setpoint, actual vertical velocity, and a vertical acceleration feed-forward command

Commented code
```
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
```
### lateral position control
#### Specifications
- The controller should use the local NE position and velocity to generate a commanded local acceleration
#### lateral position control details
- Purpose is to calculate the desired acceleration in the global frame based on desired lateral position/velocity/acceleration and current pose parameters

Commented code
```
V3F QuadControl::LateralPositionControl(V3F posCmd, V3F velCmd, V3F pos, V3F vel, V3F accelCmd)
{
  // Calculate a desired horizontal acceleration based on 
  //  desired lateral position/velocity/acceleration and current pose
  // INPUTS: 
  //   posCmd: desired position, in NED [m]
  //   velCmd: desired velocity, in NED [m/s]
  //   pos: current position, NED [m]
  //   vel: current velocity, NED [m/s]
  //   accelCmd: desired acceleration, NED [m/s2]
  // OUTPUT:
  //   return a V3F with desired horizontal accelerations. 
  //     the Z component should be 0
  // HINTS: 
  //  - use fmodf(foo,b) to constrain float foo to range [0,b]
  //  - use the gain parameters kpPosXY and kpVelXY
  //  - make sure you cap the horizontal velocity and acceleration
  //    to maxSpeedXY and maxAccelXY

  // make sure we don't have any incoming z-component
  accelCmd.z = 0;
  velCmd.z = 0;
  posCmd.z = pos.z;

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
```
### yaw control
#### Specifications
- The controller can be a linear/proportional heading controller to yaw rate commands (non-linear transformation not required)
#### yaw details
- Purpose is to calculate the desired yaw rate from the commanded yaw and the yaw parameters

Commented code
```
float QuadControl::YawControl(float yawCmd, float yaw)
{
  // Calculate a desired yaw rate to control yaw to yawCmd
  // INPUTS: 
  //   yawCmd: commanded yaw [rad]
  //   yaw: current yaw [rad]
  // OUTPUT:
  //   return a desired yaw rate [rad/s]
  // HINTS: 
  //  - use fmodf(foo,b) to constrain float foo to range [0,b]
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

```
### calculating the motor commands given commanded thrust and moments
#### Specifications
- The thrust and moments should be converted to the appropriate 4 different desired thrust forces for the moments
- Ensure that the dimensions of the drone are properly accounted for when calculating thrust from moments.
#### details
- Purpose is to calculate each motor command thrust from commanded collective thrust ans commanded moment

Commented code
```
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
```
### flight performance metrics
#### Specifications
- Ensure that in each scenario the drone looks stable and performs the required task. Specifically check that the student's controller is able to handle the non-linearities of scenario 4 (all three drones in the scenario should be able to perform the required task with the same control gains used).
#### results
- All scenarios passed with the same gains