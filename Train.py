"""
This script implements the training and deployment of an inverse mapping
model. This model takes in musical features (e.g., a spectrogram of music)
and outputs predicted robot limb activations. These predicted activations
are then sent in real-time to a robotic or simulated system via an RTBridge.

This workflow is typically used for applications like music-driven robotic
performance or musical human-robot interaction.
"""

import numpy as np
import os
import time
# Assuming 'functions' contains utility functions like create_simple_nn
from functions import create_simple_nn
# import RTBridge as rtb # This library is highly specific to a real-time
                        # system and likely not publicly available.
                        # Uncomment and ensure it's in your Python path
                        # if running on the target hardware setup.

# --- Configuration and Initialization ---

# Base directory for logs and experiment data
LOG_DIRECTORY_BASE = './datalog/Experiment_'
# Identifier for the specific experiment run whose data we are using/generating
EXPERIMENT_ID = 'v0_03' # This ID refers to the current experiment run
BABBLING_EXPERIMENT_ID = 'v2_01' # This ID refers to the previous babbling experiment for data

print(f"Initializing training for Experiment ID: {EXPERIMENT_ID}")

# --- Load Data for Model Training ---
# The inverse mapping model requires:
# 1. Music features (e.g., spectrogram)
# 2. Corresponding robot activations (from a babbling or previous recording session)

# Path to the specific music file for processing
music_file_path = os.path.join(LOG_DIRECTORY_BASE.replace('/Experiment_','/'), f'Babbling_{EXPERIMENT_ID.replace("v0_","v0_05")}.wav')
# Note: The original code used 'Babbling_v0_05.wav' specifically. Adjust if generic.

# Load activations from a previous babbling experiment.
# These are the robot's responses to known inputs, serving as labels for training.
# Assuming 'activations.txt' contains data in (num_limbs, num_samples) format,
# we transpose it to (num_samples, num_limbs) for common ML model input/output.
try:
    ground_truth_activations = np.loadtxt(os.path.join(LOG_DIRECTORY_BASE + BABBLING_EXPERIMENT_ID, 'activations.txt')).T
    print(f"Loaded ground truth activations from {os.path.join(LOG_DIRECTORY_BASE + BABBLING_EXPERIMENT_ID, 'activations.txt')}")
    print(f"Ground truth activations shape: {ground_truth_activations.shape}")
except FileNotFoundError:
    print(f"Error: Ground truth activations file not found at {os.path.join(LOG_DIRECTORY_BASE + BABBLING_EXPERIMENT_ID, 'activations.txt')}")
    print("Please ensure the babbling experiment has been run and saved its activations.")
    exit() # Exit if critical data is missing

# --- Music Spectrogram Preprocessing ---
# This part assumes a function `spect_preprocessing` exists.
# It should extract relevant features (like a spectrogram) from an audio file.
# The parameters (14800, 6758325) likely represent start_sample and end_sample
# or similar trimming parameters for the audio.
print(f"Processing music spectrogram from: {music_file_path}...")
# Placeholder for the actual music spectrogram
# You would need to define spect_preprocessing function elsewhere (e.g., in functions.py)
# Example: music_spectrogram = spect_preprocessing(music_file_path, start_sample=14800, end_sample=6758325)
# For demonstration, creating a dummy spectrogram that matches expected dimensions
# Assuming spect_preprocessing outputs (num_features, num_time_steps)
num_music_features = 128 # Example number of frequency bins/features
num_music_time_steps = ground_truth_activations.shape[0] # Match time steps with activations
music_spectrogram_raw = np.random.rand(num_music_features, num_music_time_steps) # Dummy data

# Transpose spectrogram to (num_time_steps, num_features) for common NN input
music_spectrogram_processed = np.transpose(music_spectrogram_raw)
print(f"Processed music spectrogram shape: {music_spectrogram_processed.shape}")


# --- Inverse Mapping Model Training ---
# This part assumes an `inverse_mapping_func` is defined.
# This function likely trains a model (e.g., a neural network) that learns
# to map musical features (spectrogram) to robot activations.
print("Training inverse mapping model...")
# Placeholder for the actual model training function
# You would need to define inverse_mapping_func elsewhere (e.g., in functions.py)
# This function should typically return a trained Keras/TensorFlow model or similar.
# Example:
# model = inverse_mapping_func(music_spectrogram_processed, ground_truth_activations, learning_rate=0.1)

# Using create_simple_nn from functions.py for demonstration of model structure
# Input to model: music_spectrogram_processed (num_time_steps, num_music_features)
# Output from model: predicted_activations (num_time_steps, num_limbs)
input_dim_model = music_spectrogram_processed.shape[1] # num_music_features
output_dim_model = ground_truth_activations.shape[1] # num_limbs
inverse_model = create_simple_nn(input_dim_model, output_dim_model, hidden_layers=(256, 128, 64))
inverse_model.compile(optimizer='adam', loss='mse') # Example: Adam optimizer, Mean Squared Error loss

# Train the inverse model with music features as input and robot activations as target
# This is a supervised learning task
history = inverse_model.fit(music_spectrogram_processed, ground_truth_activations,
                            epochs=100, batch_size=32, verbose=0)
print(f"Inverse mapping model training finished. Final loss: {history.history['loss'][-1]:.4f}")


# --- Generate Predicted Activations ---
# Use the trained inverse mapping model to predict new activations based on the music.
print("Generating predicted activations from the trained model...")
predicted_activations = inverse_model.predict(music_spectrogram_processed)
print(f"Predicted activations shape: {predicted_activations.shape}")

# Save the predicted activations
predicted_activations_path = os.path.join(LOG_DIRECTORY_BASE.replace('/Experiment_','/'), 'predicted_activations.txt')
np.savetxt(predicted_activations_path, predicted_activations)
print(f"Predicted activations saved to {predicted_activations_path}")

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

#     # Send an initial command (e.g., to reset robot to a safe pose)
#     # This sends a list of 4 float values (e.g., initial limb activations) and waits for 2 responses.
#     _ = bridge.sendAndReceive([0.05] * 4, 2)
#     print("Initial command sent to bridge.")

#     # --- Send Predicted Activations to System ---
#     print("Sending predicted activations to the system via RTBridge...")
#     start_time_send = time.time() # Record start time for timing the deployment

#     # Iterate through each time step of the predicted activations
#     # predicted_activations should be (num_time_steps, num_limbs)
#     for activation_set in predicted_activations:
#         # Send the current set of activations for all limbs at this time step
#         # The '_' variable stores the received data, which might contain sensor readings.
#         received_data = bridge.sendAndReceive(activation_set.tolist(), 1) # Send 1 set, expect 1 response
#         # In a real-time deployment, you might log 'activation_set' (command)
#         # and 'received_data' (robot's actual state/feedback) for monitoring.

#     end_time_send = time.time()
#     time_difference = end_time_send - start_time_send
#     print(f"Finished sending predicted activations. Total deployment duration: {time_difference:.2f} seconds.")

#     # Send a final command to return the robot to a safe state
#     _ = bridge.sendAndReceive([0.05] * 4, 2)
#     print("Final command sent to bridge.")

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

print("\nInverse mapping model training and (simulated) deployment script finished.")
print("Predicted activations are available in the datalog directory.")
print("To run this with a physical system, ensure 'RTBridge' is correctly installed and configured, then uncomment the RTBridge-related code.")