"""
This script implements a 'babbling' experiment for system identification
of a robotic or motor control system. It generates specific activation
signals (inputs) and sends them to a real-time bridge for a physical
or simulated system, intended for collecting corresponding system outputs.

The collected input-output data can then be used to train a system
identification model (e.g., a neural network) that learns the dynamics
of the system.
"""

import numpy as np
import os
import time
# Assuming 'functions' contains systemID_input_gen_func and other utilities
from functions import systemID_input_gen_func
# import RTBridge as rtb # This library is highly specific to a real-time
                        # system and likely not publicly available.
                        # Uncomment and ensure it's in your Python path
                        # if running on the target hardware setup.

# --- Experiment Configuration ---
directory_path = './datalog/'
Experiment_ID = 'v2_01'  # Unique identifier for this experiment run
babbling_duration = 8    # Total duration of the babbling sequence in seconds
fs = 100                 # Sampling frequency of the activation signals (Hz)
minimum_input = 0.96     # Minimum activation level for limbs
maximum_input = 0.99     # Maximum activation level for limbs
steps = 0.5              # Duration of each constant activation step within the sequence (seconds)

# --- Setup Data Logging Directory ---
# Create a dedicated directory for this experiment's data if it doesn't exist
experiment_data_path = os.path.join(directory_path, f'Experiment_{Experiment_ID}')
if not os.path.exists(experiment_data_path):
    os.makedirs(experiment_data_path)
    print(f"Created directory: {experiment_data_path}")
else:
    print(f"Directory already exists: {experiment_data_path}. Data will be saved here.")

# --- Generate Babbling Activation Signals ---
# These signals serve as inputs to the system for identification.
# The 'systemID_input_gen_func' is assumed to be defined in 'functions.py'
print(f"Generating babbling activation signals for {babbling_duration} seconds at {fs} Hz...")
activations = systemID_input_gen_func(
    duration=babbling_duration,
    fs=fs,
    steps=steps,
    min_input=minimum_input,
    max_input=maximum_input
)
print(f"Generated activation signals with shape: {activations.shape}")

# Save the generated activations to a text file for record-keeping
np.savetxt(os.path.join(experiment_data_path, 'activations.txt'), activations)
print(f"Activations saved to {os.path.join(experiment_data_path, 'activations.txt')}")

# Optional: Plot activations if matplotlib is available and you want to visualize
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# for i in range(activations.shape[0]):
#     plt.plot(activations[i, :], label=f'Limb {i+1} Activation')
# plt.title('Generated Babbling Activation Signals')
# plt.xlabel('Time Sample')
# plt.ylabel('Activation Value')
# plt.legend()
# plt.grid(True)
# plt.show()


# --- RTBridge Setup ---
# This section assumes the use of a specific Real-Time Bridge (RTBridge)
# for communication with a physical or simulated robotic system.
# Replace with your actual bridge setup if different.
# NOTE: The 'RTBridge' library is proprietary or specific to a certain setup
# and may not be publicly available. This part will cause an ImportError
# if 'RTBridge' is not installed or accessible.

# try:
#     # PXI controller IP and port for receiving data
#     pxiWin10 = "169.254.172.223:5555"
#     # Local port for publishing commands to the system
#     pubPort = "5557"
#     # Initialize the bridge with specific setup (e.g., hand_4_4 implies 4 input/4 output hand system)
#     bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, milliTimeStep=int(10))
#     print(f"RTBridge setup complete. Publishing to {pubPort}, receiving from {pxiWin10}")

#     # --- Establish Connection ---
#     print("Starting RTBridge connection...")
#     bridge.startConnection()
#     print("RTBridge connected.")

#     # Send an initial command to ensure the system is ready (e.g., reset, set initial state)
#     # This command sends a list of 4 float values (e.g., initial limb activations)
#     # and waits for 2 responses/cycles.
#     _ = bridge.sendAndReceive([0.05] * 4, 2)
#     print("Initial command sent to bridge.")

#     # --- Send Activation Data to System ---
#     print("Sending activation data to the system via RTBridge...")
#     start_time = time.time() # Record start time for timing the experiment

#     # Iterate through each column of the activations array (each time step)
#     # activations are typically (num_limbs, num_samples), so we iterate over samples
#     # and send activations for all limbs at that sample.
#     for sample_idx in range(activations.shape[1]):
#         activation_set = activations[:, sample_idx] # Get activations for all limbs at current sample
#         # Send the current set of activations and receive feedback (if any)
#         # The '_' variable stores the received data, which might contain sensor readings.
#         # In a real system ID, this received data would be logged as system output.
#         received_data = bridge.sendAndReceive(activation_set.tolist(), 1) # Send 1 set, expect 1 response
#         # Log 'activation_set' (input) and 'received_data' (output) here for system ID
#         # For example:
#         # with open(os.path.join(experiment_data_path, 'log.txt'), 'a') as f:
#         #     f.write(f"{time.time() - start_time}, {activation_set.tolist()}, {received_data}\n")

#     end_time = time.time()
#     print(f"Finished sending activations. Total send duration: {end_time - start_time:.2f} seconds.")

#     # --- Close Connection ---
#     print("Stopping RTBridge connection...")
#     bridge.stopConnection()
#     print("RTBridge disconnected.")

# except ImportError:
#     print("RTBridge library not found. Skipping real-time bridge operations.")
#     print("This script is designed for specific hardware interaction and requires RTBridge.")
# except Exception as e:
#     print(f"An error occurred during RTBridge operation: {e}")
#     if 'bridge' in locals() and bridge:
#         bridge.stopConnection() # Attempt to clean up connection if error occurs after starting.

print("\nBabbling experiment script finished.")
print("Generated activation signals are available in the datalog directory.")
print("To run this with a physical system, ensure 'RTBridge' is correctly installed and configured, then uncomment the RTBridge-related code.")