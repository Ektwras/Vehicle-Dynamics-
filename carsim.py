import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, solve_continuous_are

# Constants
mass = 1470     # Mass of the car (kg)
I_z = 1900      # Moment of inertia about the z-axis (kg*m^2)
l_f = 1.04      # Distance from the center of mass to the front axle (m)
l_r = 1.56      # Distance from the center of mass to the rear axle (m)
C_f = 71000     # Front tire cornering stiffness (N/rad)
C_r = 47000     # Rear tire cornering stiffness (N/rad)
V_x = 10        # Velocity of the car in the x-direction (m/s)

# Initial conditions
X_0 = 0          # Initial x-position (m)
Y_0 = 0          # Initial y-position (m)
Psi_0 = 0        # Initial yaw angle (rad)

x_dot0 = 10      # Initial longitudinal velocity (m/s)
y_dot0 = 0       # Initial lateral velocity (m/s)
psi_dot0 = 0     # Initial yaw rate (rad/s)

# Time parameters
dt = 0.01       # Time step (s)
t_end = 10      # End time (s)
timesteps = int(t_end / dt)  # Number of time steps

# Function to calculate the derivatives of the state variables for state state space


# Construct the input matrix B
B_SS = np.array([[0],
  [C_f / mass],
  [0],
  [C_f * l_f / I_z]
  ])

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html
B = B_SS                              # Input matrix

doYawControl=False
if doYawControl==True:
  Q = np.array([[0.0,0,0,0],[0,0.01,0,0],[0,0.0,5,0],[0,0,0,0.3]])      # State cost matrix yaw tracking
else:
  Q = np.array([[0.5,0,0,0],[0,0.01,0,0],[0,0.0,0.0,0],[0,0,0,0.3]])     # State cost matrix lateral tracking

R = np.array([[1]])                   # Input cost matrix



def dynamics_SS(state,delta):
    y, y_dot, psi, psi_dot = state
    # System state vector
    stateVector = np.array([y, y_dot, psi, psi_dot])

    # Compute the derivative of the state vector
    stateVector_dot = A @ stateVector + B_SS.flatten() * delta

    # Return the derivatives of the state
    return np.array([stateVector_dot[0], stateVector_dot[1],stateVector_dot[2],stateVector_dot[3]])

# Non-linear dynamics
def dynamics(state,delta):
    x_dot, y_dot, psi_dot = state

    a_f = delta - np.arctan2((y_dot + psi_dot * l_f), x_dot)
    a_r = -np.arctan2((y_dot - psi_dot * l_r), x_dot)

    # Calculate lateral forces
    F_yf = C_f * a_f
    F_yr = C_r * a_r

    # Calculate derivatives
    x_ddot = 0                                                                      # Fixed speed
    y_ddot = ((1 / mass) * (F_yf * np.cos(delta) + F_yr)) - psi_dot * x_dot         # Lateral component
    psi_ddot = (1 / I_z) * (l_f * F_yf - l_r * F_yr)                                # Yaw component

    return np.array([x_ddot, y_ddot, psi_ddot])

# Function to calculate the trajectory
def trajectory_cal(state, trajVector):
    x_dot, y_dot, psi_dot = state
    X, Y, Psi = trajVector

    # Calculate derivatives
    X_dot = x_dot * np.cos(Psi) - y_dot * np.sin(Psi)
    Y_dot = x_dot * np.sin(Psi) + y_dot * np.cos(Psi)
    Psi_dot = psi_dot

    return np.array([X_dot, Y_dot, Psi_dot])

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.dt=dt
    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error*dt
        derivative = (error - self.prev_error)/dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
# Define setpoints
YPos_Set=Y_0
YPos_Set_SS=Y_0
YawAngle_Set=Psi_0
YawAngle_Set_SS=Psi_0

def saturate(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Generate sinusoidal steering input
def generate_steering_input(timesteps, dt):
    freq = 0.2  # Frequency of the sinusoidal input (Hz)
    amplitude = 0.1  # Amplitude of the sinusoidal input (rad)
    time_array = np.linspace(0, t_end, timesteps)
    steering_input = amplitude * np.sin(2 * np.pi * freq * time_array)
    return steering_input


def my_riccati(A, B, Q, R, P):

    P_dot = (A.T @ P + P @ A - (P @ B) @ np.linalg.inv(R) @ B.T @ P + Q)

    return P_dot


# State space system
# Initialize state vector
stateVector_SS = np.array([Y_0, y_dot0, Psi_0,psi_dot0], dtype=float)
# Initialize trajectory vector
trajVector_SS = np.array([X_0, Y_0, Psi_0], dtype=float)
stateVector_SSTmp=np.array([0, 0, 0], dtype=float)                    # This is for compatibility with the NL
# Initialize arrays to store results
stateArray_SS = np.zeros((timesteps, 3))           # Array to store the states
trajectoryArray_SS = np.zeros((timesteps, 3))      # Array to store the trajectory (also states)
acc_y_array_SS = np.zeros(timesteps)               # Array to store lateral acceleration
acc_x_array_SS = np.zeros(timesteps)               # Array to store longitudinal acceleration
time_array_SS = np.linspace(0, t_end, timesteps)   # Time array
delta_array_SS = np.zeros(timesteps)               # Array to store delta

# non linear
# Initialize state vector
stateVector = np.array([x_dot0, y_dot0, psi_dot0], dtype=float)
# Initialize trajectory vector
trajVector = np.array([X_0, Y_0, Psi_0], dtype=float)
# Initialize arrays to store results
stateArray = np.zeros((timesteps, 3))           # Array to store the states
trajectoryArray = np.zeros((timesteps, 3))      # Array to store the trajectory (also states)
acc_y_array = np.zeros(timesteps)               # Array to store lateral acceleration
acc_x_array = np.zeros(timesteps)               # Array to store longitudinal acceleration
time_array = np.linspace(0, t_end, timesteps)   # Time array
delta_array = np.zeros(timesteps)               # Array to store delta

stateVector_LQR = np.array([0, 0, 0, 0], dtype=float)

# Generate steering input
steering_input = generate_steering_input(timesteps, dt)

# Initialize PID controllers

if doYawControl==True:
  Kp=0.25*0
  Ki=0.00*0
  Kd=0.1*0
else:
  Kp=0.25
  Ki=0.00
  Kd=0.1

lateral_position_controller = PIDController(Kp,Ki,Kd, setpoint=YPos_Set)
lateral_position_controller_SS = PIDController(Kp,Ki,Kd, setpoint=YPos_Set_SS)

if doYawControl==True:
  Kp=5
  Ki=0.5
  Kd=0.5
else:
  Kp=0.25
  Ki=0.00
  Kd=0.01


yaw_angle_controller = PIDController(Kp, Ki, Kd, setpoint=YawAngle_Set)
yaw_angle_controller_SS = PIDController(Kp, Ki, Kd, setpoint=YawAngle_Set_SS)


# Main loop
maxDelta=0.3

last_update_time = 0
K_LQR = np.zeros((1, 4))
P = np.zeros((4, 4))

for i in range(timesteps):
    current_time = i * dt
    x_dot = x_dot0 + 8 * np.sin(2 * np.pi * 0.2 * current_time)                 # x_dot and matrix A are recalculated st every timestep
    A = np.array([
      [0, 1, x_dot,0],
      [0,-(C_f + C_r) / (mass * x_dot),0, -(C_f * l_f - C_r * l_r) / (mass * x_dot) - x_dot],
      [0,0,0,1],
      [0,-(C_f * l_f - C_r * l_r) / (I_z * x_dot), 0, -(C_f * l_f**2 + C_r * l_r**2) / (I_z * x_dot)]
      ])

    if current_time - last_update_time >= 0.2:                                  # calculates Riccati matrix using solve_continupus_are every 0.2 seconds. change to 0.5 or 1.0
      P_cont = solve_continuous_are(A, B, Q, R)
      K_LQR_cont = np.linalg.inv(R) @ (B.T @ P_cont)

      for time in np.arange(0, t_end, dt):                                 # calculates Riccati matrix using numerical integration.
        P = P + my_riccati(A, B, Q, R, P) * dt
        K_LQR = np.linalg.inv(R) @ (B.T @ P)

      last_update_time = current_time

    Y_ref = 2 * np.sin(2 * np.pi * 0.25 * current_time)                         # Y_ref updates at every timestep.

    if doYawControl==True:
        YawAngle_Set=0.1
        YawAngle_Set_SS=0.1
    else:
        YawAngle_Set=Psi_0
        YawAngle_Set_SS=Psi_0


    Debug=0 # If Debug=1 uses the steering_input for model development and verification

    # Store current state
    stateArray_SS[i,0] = x_dot # To come back to that
    stateArray_SS[i,1] = stateVector_SS[1]
    stateArray_SS[i,2] = stateVector_SS[3]
    trajectoryArray_SS[i] = trajVector_SS

    EGOPosX_m=trajectoryArray_SS[i, 0]
    EGOPosY_m=trajectoryArray_SS[i, 1]
    EGOYaw_rad=trajectoryArray_SS[i, 2]

    # State space system
    stateVector_LQR=stateVector_SS

    stateVector_LQR[0]=EGOPosY_m-Y_ref
    stateVector_LQR[2]=EGOYaw_rad-YawAngle_Set

    deltaTmp=saturate(-K_LQR@stateVector_LQR,-maxDelta, maxDelta)

    if Debug==1:
      deltaTmp=steering_input[i]

    delta_array_SS[i]=deltaTmp

    # Calculate derivatives
    derivativesState_SS = dynamics_SS(stateVector_SS,deltaTmp)

    # Update state using Euler integration
    stateVector_SS += derivativesState_SS * dt

    # Just for compatibility with NL
    stateVector_SSTmp[0]=x_dot
    stateVector_SSTmp[1]=stateVector_SS[1]  # y_dot
    stateVector_SSTmp[2]=stateVector_SS[3]  # psi_dot

    # Calculate lateral and longitudinal accelerations
    # x_dot, x_dot, psi_dot = stateVector_SSTmp
    # y_ddot=derivativesState_SS[1]
    # x_ddot=0
    acc_x_SS = 0 - stateVector_SSTmp[2] * stateVector_SSTmp[1]
    acc_y_SS = derivativesState_SS[1] + stateVector_SSTmp[2] * stateVector_SSTmp[0]
    acc_y_array_SS[i] = acc_y_SS
    acc_x_array_SS[i] = acc_x_SS

    # Calculate derivatives for trajectory
    derivativesTraj_SS = trajectory_cal(stateVector_SSTmp, trajVector_SS)
    trajVector_SS += derivativesTraj_SS * dt
    #trajVector_SS[1]=stateVector_SS[0] # Y from state space
    #trajVector_SS[2]=stateVector_SS[2] # Psi from state space

    ######
    # Non Linear dynamics

    # Store current state (this was the problem; we need to store first)
    stateArray[i] = stateVector
    stateArray[i, 0] = x_dot
    trajectoryArray[i] = trajVector

    EGOPosX_m=trajectoryArray[i, 0]
    EGOPosY_m=trajectoryArray[i, 1]
    EGOYaw_rad=trajectoryArray[i, 2]

    lateral_position_controller.setpoint=Y_ref
    yaw_angle_controller.setpoint=YawAngle_Set

    # Update PID controllers
    lateral_position_control_output = lateral_position_controller.update(EGOPosY_m)
    yaw_angle_control_output = yaw_angle_controller.update(EGOYaw_rad)
    # Apply control outputs to the bicycle model
    deltaTmp=saturate(lateral_position_control_output+yaw_angle_control_output,-maxDelta, maxDelta)

    if Debug==1:
      deltaTmp=steering_input[i]

    delta_array[i]=deltaTmp

    # Calculate derivatives
    derivativesState = dynamics(stateVector,deltaTmp)

    # Update state using Euler integration
    stateVector += derivativesState * dt

    # Calculate lateral and longitudinal accelerations
    #x_ddot, y_ddot, psi_ddot = dynamics(stateVector)
    acc_x = derivativesState[0] - stateVector[2] * stateVector[1]
    acc_y = derivativesState[1] + stateVector[2] * stateVector[0]
    acc_y_array[i] = acc_y
    acc_x_array[i] = acc_x

    # Calculate derivatives for trajectory
    derivativesTraj = trajectory_cal(stateVector, trajVector)
    trajVector += derivativesTraj * dt


print("\n")
print(f"K optimal gains from continuous ARE: {K_LQR_cont}")
print(f"K optimal gains from numerical integration: {K_LQR}")
print("\n")

# Plot results
plt.figure(figsize=(14, 10))

plt.subplot(6, 1, 1)
plt.plot(time_array_SS, stateArray_SS[:, 0], label='SS')
plt.plot(time_array, stateArray[:, 0], label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.grid(True)
plt.legend()

plt.subplot(6, 1, 2)
plt.plot(time_array_SS, stateArray_SS[:, 2], label='SS')
plt.plot(time_array, stateArray[:, 2], label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Yaw Rate [rad/s]]')
plt.grid(True)
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(time_array_SS, acc_y_array_SS, label='SS')
plt.plot(time_array, acc_y_array, label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Lat. Acc. [m/s^2]')
plt.grid(True)
plt.legend()

plt.subplot(6, 1, 4)
'''
plt.plot(time_array_SS, acc_x_array_SS, label='SS')
plt.plot(time_array, acc_x_array, label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Long. Acc. [m/s^2]')
plt.grid(True)
plt.legend()
'''
plt.plot(time_array_SS, delta_array_SS, label='SS')
plt.plot(time_array, delta_array, label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Delta [rad]')
plt.grid(True)
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(trajectoryArray_SS[:, 0], trajectoryArray_SS[:, 1], label='SS')
plt.plot(trajectoryArray[:, 0], trajectoryArray[:, 1], label='NL',linestyle='dashed')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
#plt.title('Single Track Car Trajectory')
plt.grid(True)
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(time_array_SS, trajectoryArray_SS[:, 2], label='SS')
plt.plot(time_array, trajectoryArray[:, 2], label='NL',linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Yaw angle[rad]')
plt.title('Single Track Car Trajectory')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
