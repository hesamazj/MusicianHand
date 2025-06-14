from scipy.signal import stft, resample
from scipy.io import loadmat
import tensorflow as tf # Prefer tf.keras for consistent Keras usage
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
# Use tf.keras for Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import math
import zmq
# import RTBridge as rtb # This library seems specific and might not be publicly available or needed.
from scipy.io import wavfile
import scipy.signal as signal
from tensorflow.keras import backend as K # Using tf.keras.backend
from scipy.stats import beta
from scipy.fft import fft
import pyaudio
import mido
import wave
import struct
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Added for train_test_split


# --- Helper Functions ---

def systemID_input_gen_func(duration, fs, steps, min_input, max_input):
    """
    Generates input signals for system identification, simulating limb activations.
    These signals consist of stepped, smoothly transitioning activation levels
    for multiple "limbs" or control outputs.

    Args:
        duration (float): Total duration of the generated signal in seconds.
        fs (int): Sampling frequency in Hz.
        steps (float): Duration of each 'step' (constant activation block) in seconds.
        min_input (float): Minimum activation value for any limb.
        max_input (float): Maximum activation value for any limb.

    Returns:
        numpy.ndarray: A 2D array of activation signals.
                       Shape: (number_of_limbs, number_of_samples).
                       Each row represents a limb's activation over time.
    """
    samples_per_step = int(fs * steps)
    number_of_values = int(duration / steps)  # Number of activation "blocks" for each limb

    # Initialize limb activations to min_input for all blocks
    # Shape: (number_of_blocks, number_of_limbs)
    limb_activations_blocks = np.ones((number_of_values, 4)) * min_input

    # Apply pulsed activations for each limb, cycling every 4 blocks
    # This creates a structured yet varying activation pattern for system ID
    for i in range(4):  # Iterate through each limb (assuming 4 limbs)
        for c in range(0, number_of_values, 4):  # Cycle every 4 blocks to activate one limb at a time
            # Set a random activation value between 90% and 100% of max_input
            # This introduces slight randomness in the high activation states
            limb_activations_blocks[c + i, i] = min_input + ((max_input - min_input) * np.random.uniform(0.9, 1, 1))

    # Initialize the full activation array for all samples
    # Shape: (number_of_limbs, total_number_of_samples)
    activations_full = np.ones((4, samples_per_step * number_of_values)) * min_input
    # The original code had (number_of_values + 1) in samples_per_step * (number_of_values + 1)
    # but the loop only goes up to number_of_values-1 for j, so adjusting to match.

    x_transition = np.linspace(-0.25, 0.25, samples_per_step)
    a_steepness = 30  # Controls the steepness of the sigmoid-like transition curve

    # Apply activations with a smooth transition for each step
    for i in range(4): # For each limb
        for j in range(number_of_values): # For each activation block
            # Create a smooth transition curve (sigmoid-like shape)
            y_transition = 1 / (1 + np.exp(-a_steepness * x_transition))
            # Scale the transition to go from min_input to the specific limb_activation value
            scaled_transition = min_input + (limb_activations_blocks[j, i] - min_input) * y_transition

            # Assign the smoothly transitioned segment to the correct part of the full activation array
            start_idx = j * samples_per_step
            end_idx = (j + 1) * samples_per_step
            if end_idx > activations_full.shape[1]: # Prevent index out of bounds for the last segment
                end_idx = activations_full.shape[1]
                scaled_transition = scaled_transition[:(end_idx - start_idx)] # Trim if needed
            activations_full[i, start_idx:end_idx] = scaled_transition

    return activations_full

def compute_stft(y, sr, n_fft=2048, hop_length=512):
    """
    Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    Args:
        y (numpy.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        n_fft (int): FFT window size (number of samples per segment).
        hop_length (int): Number of audio samples between adjacent STFT columns (overlap).

    Returns:
        numpy.ndarray: Complex-valued spectrogram (frequency x time).
    """
    # Using scipy.signal.stft. It returns frequencies, segment times, and the STFT matrix.
    f, t, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    return Zxx

def note_to_freq(note_midi):
    """
    Converts a MIDI note number to its corresponding frequency in Hz.
    Uses the standard A4 (MIDI 69) = 440 Hz reference.

    Args:
        note_midi (int): MIDI note number (0-127).

    Returns:
        float: Frequency in Hz.
    """
    return 440 * (2 ** ((note_midi - 69) / 12))

def create_simple_nn(input_dim, output_dim, hidden_layers=(64, 32)):
    """
    Creates a simple Keras Sequential neural network model.

    Args:
        input_dim (int): Dimension of the input layer (number of features).
        output_dim (int): Dimension of the output layer (number of prediction targets).
        hidden_layers (tuple): A tuple specifying the number of units in each hidden layer.

    Returns:
        tf.keras.models.Sequential: The compiled Keras model.
    """
    model = Sequential()
    # Input layer with ReLU activation
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    # Additional hidden layers with ReLU activation
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    # Output layer. 'linear' activation is common for regression tasks.
    model.add(Dense(output_dim, activation='linear'))
    return model

def generate_and_save_midi_audio(WAVE_OUTPUT_FILENAME="output.wav", RECORD_SECONDS=5, play_live=False):
    """
    Generates audio from MIDI input (live or simulated) and saves it to a WAV file.
    If 'play_live' is True, it attempts to listen to a live MIDI device.
    If no MIDI device is found or 'play_live' is False, it generates a simple sine wave.

    Args:
        WAVE_OUTPUT_FILENAME (str): Name of the output WAV file.
        RECORD_SECONDS (int): Desired duration of the generated audio in seconds.
        play_live (bool): If True, listens for live MIDI input. If False,
                          a simple sine wave at a fixed frequency is generated.
    """
    FORMAT = pyaudio.paInt16 # 16-bit integers for audio samples
    CHANNELS = 1             # Mono audio
    RATE = 44100             # Sampling rate in Hz
    CHUNK = 1024             # Number of frames processed per buffer

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

    print('Starting audio generation/recording...')
    audio_frames = []
    total_duration_generated = 0 # Tracks the actual duration of audio added to frames
    start_time_total = time.time() # To monitor overall script execution time

    midi_in = None
    if play_live:
        # Attempt to open MIDI input device
        try:
            midi_in = mido.open_input()
            print(f"MIDI input opened: {midi_in.name}. Playing live MIDI input.")
        except IOError:
            print("No MIDI input device found or accessible. Falling back to simulated tone.")
            play_live = False # Disable live play if no MIDI device

    if not play_live:
        # Default behavior: generate a fixed sine wave if no live MIDI or device unavailable
        print("Generating a default sine wave tone.")
        frequency = 440.0 # A4 note (example)
        amplitude = 0.5   # Max amplitude (0.0 to 1.0) for float audio before scaling

        # Generate audio for the full duration
        num_chunks_to_generate = int(RATE / CHUNK * RECORD_SECONDS)
        for _ in range(num_chunks_to_generate):
            t_values = np.arange(CHUNK) / RATE # Time points for current chunk
            y = amplitude * np.sin(2 * np.pi * frequency * t_values)
            # Convert float audio to 16-bit integer format and then to bytes
            data = (y * 32767).astype(np.int16).tobytes()
            audio_frames.append(data)
            stream.write(data) # Play audio chunk
            total_duration_generated += CHUNK / RATE # Accumulate duration
    else: # Live MIDI input
        current_frequency = 0.0
        current_amplitude = 0.0
        last_chunk_time = time.time() # To track when to generate next audio chunk

        # Loop until desired record duration is met
        while total_duration_generated < RECORD_SECONDS:
            # Process any pending MIDI messages
            for msg in midi_in.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0: # Check velocity to distinguish from note_off with velocity 0
                    current_frequency = note_to_freq(msg.note)
                    current_amplitude = msg.velocity / 127.0
                    # print(f"Note On: {msg.note}, Freq: {current_frequency:.2f} Hz, Amp: {current_amplitude:.2f}")
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    current_frequency = 0.0 # Stop tone
                    current_amplitude = 0.0
                    # print("Note Off.")

            # Generate audio chunk based on current note state
            current_time = time.time()
            if current_time - last_chunk_time >= CHUNK / RATE:
                t_values = np.arange(CHUNK) / RATE
                if current_frequency > 0 and current_amplitude > 0:
                    y = current_amplitude * np.sin(2 * np.pi * current_frequency * t_values)
                else:
                    y = np.zeros_like(t_values) # Generate silence if no note
                
                data = (y * 32767).astype(np.int16).tobytes()
                audio_frames.append(data)
                stream.write(data)
                total_duration_generated += CHUNK / RATE
                last_chunk_time = current_time

            # Small sleep to prevent busy-waiting, but balanced for real-time
            time.sleep(0.001) # Sleep for 1ms


    # Stop audio stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
    if midi_in:
        midi_in.close() # Close MIDI input if it was opened

    # Concatenate all collected audio frames
    full_audio_data = b''.join(audio_frames)

    # Save concatenated audio to WAV file
    try:
        wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(full_audio_data)
        wave_file.close()
        print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

# --- Neural Network Models for Inverse Mapping ---

def music_amp_neural_net(music_spect, labels, test_size):
    """
    Creates, compiles, and trains a simple feed-forward neural network
    for mapping music spectrograms to limb activations, potentially
    for amplitude or categorical outputs.

    Args:
        music_spect (numpy.ndarray): Input music spectrogram data.
                                     Expected shape: (num_samples, num_freq_bins, num_time_frames).
                                     This will be flattened by the first layer.
        labels (numpy.ndarray): Corresponding target limb activation labels.
                                If using 'softmax' and 'categorical_crossentropy', these should be
                                one-hot encoded (e.g., (num_samples, num_classes)).
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tf.keras.models.Sequential: The trained Keras model.
    """
    # Normalize music spectrogram data to range between 0 and 1
    # This helps with neural network training stability
    music_spect = music_spect / np.max(np.abs(music_spect))

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        music_spect, labels, test_size=test_size, random_state=42
    )

    # Define the neural network architecture
    layers = [
        # Flatten the input spectrogram (e.g., from (257, 20) to 5140 features)
        tf.keras.layers.Flatten(input_shape=(music_spect.shape[1], music_spect.shape[2])),
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with ReLU activation
        # Output layer. 'softmax' implies classification (e.g., 4 categories of limb states).
        # If the goal is regression (continuous activation values), 'linear' activation
        # and 'mse' loss would be more appropriate.
        tf.keras.layers.Dense(4, activation='softmax')
    ]
    model = Sequential(layers)

    # Compile the model
    # 'categorical_crossentropy' is for multi-class classification with one-hot encoded labels.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training music_amp_neural_net...")
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=1)
    print(f"Music Amp NN training finished. Final loss: {history.history['loss'][-1]:.4f}, Final accuracy: {history.history['accuracy'][-1]:.4f}")

    return model

def music_notes_neural_net(music_spect, labels, test_size):
    """
    Creates, compiles, and trains a Convolutional Neural Network (CNN)
    for mapping music spectrograms to limb activations, suitable for
    tasks where spatial features (e.g., frequency patterns) in the spectrogram
    are important.

    Args:
        music_spect (numpy.ndarray): Input music spectrogram data.
                                     Expected shape: (num_samples, num_freq_bins, num_time_frames, 1).
                                     The `1` channel is for grayscale-like data.
        labels (numpy.ndarray): Corresponding target limb activation labels.
                                If using 'softmax' and 'categorical_crossentropy', these should be
                                one-hot encoded (e.g., (num_samples, num_classes)).
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tf.keras.models.Sequential: The trained Keras CNN model.
    """
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        music_spect, labels, test_size=test_size, random_state=42 # Added random_state for reproducibility
    )

    # Define the CNN architecture
    layers = [
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(music_spect.shape[1], music_spect.shape[2], music_spect.shape[3])),
        tf.keras.layers.MaxPooling2D((2, 2)), # Downsampling
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)), # Downsampling
        tf.keras.layers.Flatten(), # Flatten for Dense layers
        tf.keras.layers.Dense(64, activation='relu'), # Hidden dense layer
        # Output layer. 'softmax' implies classification.
        # As with `music_amp_neural_net`, if targeting continuous activations,
        # 'linear' activation and 'mse' loss would be more suitable.
        tf.keras.layers.Dense(4, activation='softmax') # Assuming 4 output classes/categories
    ]
    model = Sequential(layers)

    # Compile the model
    # 'categorical_crossentropy' is for multi-class classification.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training music_notes_neural_net (CNN)...")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
    print(f"Music Notes CNN training finished. Final loss: {history.history['loss'][-1]:.4f}, Final accuracy: {history.history['accuracy'][-1]:.4f}")

    return model

# --- Main Experiment/Workflow Functions (from previous revision) ---

def run_system_identification_experiment(output_dir="data", model_filename="system_id_model.pkl"):
    """
    Conducts a system identification experiment.
    Generates input signals, simulates output (placeholder), trains a simple NN,
    and saves the trained model.

    Args:
        output_dir (str): Directory to save data and models.
        model_filename (str): Filename for the trained model (pickle format).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parameters for input generation
    duration_sec = 10  # Total duration of input signal
    sampling_rate_hz = 1000  # Hz
    step_duration_sec = 0.5  # Duration of each distinct input step
    min_activation = 0.1
    max_activation = 1.0

    print("Generating system identification input signals...")
    inputs = systemID_input_gen_func(duration_sec, sampling_rate_hz, step_duration_sec, min_activation, max_activation)
    print(f"Generated input signals with shape: {inputs.shape}")

    # --- Simulate System Output (PLACEHOLDER) ---
    # In a real system identification, you would apply 'inputs' to your physical
    # or simulated system and record its 'outputs'.
    # For demonstration, let's create a dummy output that's a simple transformation of the input.
    # This is where your actual system model or data would come in.
    print("Simulating system output (placeholder)...")
    # Example: A simple linear combination and some noise
    # Assuming inputs[0] influences output strongest
    outputs = np.zeros(inputs.shape[1])
    for i in range(inputs.shape[0]):
        # Simple weighted sum, simulating a multi-input system
        outputs += inputs[i, :] * (0.5 + 0.1 * i) # Different weights for each limb
    outputs += np.random.randn(outputs.shape[0]) * 0.05 # Add some noise

    # For system ID, we usually need input-output pairs
    # Reshape inputs for training: (num_samples, num_limbs)
    X_train = inputs.T
    y_train = outputs.reshape(-1, 1) # Reshape output to (num_samples, 1)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # --- Train a Simple Neural Network for System Identification ---
    # The NN learns to map inputs to outputs, representing the system's dynamics.
    print("Training neural network for system identification...")
    input_dim = X_train.shape[1] # Number of input features (limbs)
    output_dim = y_train.shape[1] # Number of output features (e.g., joint angle, force)
    hidden_layers = (64, 32, 16) # Example: 3 hidden layers

    model = create_simple_nn(input_dim, output_dim, hidden_layers)
    model.compile(optimizer='adam', loss='mse') # Adam optimizer, Mean Squared Error loss

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    print(f"Model training finished. Final loss: {history.history['loss'][-1]:.4f}")

    # Save the trained model
    model_path = os.path.join(output_dir, model_filename)
    # For Keras models, it's recommended to save the entire model architecture and weights
    model.save(os.path.join(output_dir, "system_id_model.h5"))
    # The original code used pickle to save weights, which is less ideal for Keras models.
    # with open(model_path, 'wb') as f:
    #     pickle.dump(model.get_weights(), f) # Saving weights, not the full model architecture
    print(f"System ID model saved to {os.path.join(output_dir, 'system_id_model.h5')}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Predict and plot a segment
    y_pred = model.predict(X_train).flatten()
    plt.figure(figsize=(12, 6))
    plt.plot(outputs[:500], label='Actual Output')
    plt.plot(y_pred[:500], label='Predicted Output', linestyle='--')
    plt.title('Actual vs. Predicted System Output (First 500 Samples)')
    plt.xlabel('Time Sample')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
    plt.close()

    print("System identification experiment completed.")


def run_audio_synthesis_experiment(output_audio_name="synthesized_midi_audio.wav", record_duration=10, live_midi=False):
    """
    Runs an audio synthesis experiment.
    Generates an audio file based on MIDI input (live or simulated).

    Args:
        output_audio_name (str): Filename for the generated WAV audio.
        record_duration (int): Duration of audio to generate in seconds.
        live_midi (bool): If True, listens for live MIDI input. Otherwise,
                          generates a default tone.
    """
    print(f"Starting audio synthesis experiment: {output_audio_name}")
    generate_and_save_midi_audio(
        WAVE_OUTPUT_FILENAME=output_audio_name,
        RECORD_SECONDS=record_duration,
        play_live=live_midi
    )
    print("Audio synthesis experiment completed.")

# Example usage (uncomment to run directly)
if __name__ == "__main__":
    # --- System Identification Example ---
    print("--- Running System Identification Experiment ---")
    run_system_identification_experiment(output_dir="system_id_results", model_filename="robot_dynamics_model.h5")

    # --- Audio Synthesis Example ---
    print("\n--- Running Audio Synthesis Experiment ---")
    # To test with live MIDI, set live_midi=True and ensure a MIDI device is connected.
    # Otherwise, it will generate a simple sine wave.
    run_audio_synthesis_experiment(output_audio_name="my_synthesized_tune.wav", record_duration=5, live_midi=False)

    # --- Example of using new neural network functions (dummy data) ---
    print("\n--- Testing New Neural Network Functions with Dummy Data ---")
    # Dummy data for demonstration purposes
    # For music_amp_neural_net: (num_samples, num_freq_bins, num_time_frames)
    dummy_music_spect_amp = np.random.rand(100, 257, 20)
    # Dummy labels: Assuming 4 categories, one-hot encoded
    dummy_labels_amp = tf.keras.utils.to_categorical(np.random.randint(0, 4, 100), num_classes=4)
    print(f"Dummy music_spect_amp shape: {dummy_music_spect_amp.shape}, labels shape: {dummy_labels_amp.shape}")
    
    # Train music_amp_neural_net
    trained_amp_model = music_amp_neural_net(dummy_music_spect_amp, dummy_labels_amp, test_size=0.2)
    print("Trained music_amp_neural_net model summary:")
    trained_amp_model.summary()

    # Dummy data for music_notes_neural_net: (num_samples, num_freq_bins, num_time_frames, 1 channel)
    dummy_music_spect_notes = np.random.rand(100, 257, 20, 1)
    # Dummy labels: Assuming 4 categories, one-hot encoded
    dummy_labels_notes = tf.keras.utils.to_categorical(np.random.randint(0, 4, 100), num_classes=4)
    print(f"Dummy music_spect_notes shape: {dummy_music_spect_notes.shape}, labels shape: {dummy_labels_notes.shape}")

    # Train music_notes_neural_net
    trained_notes_model = music_notes_neural_net(dummy_music_spect_notes, dummy_labels_notes, test_size=0.2)
    print("Trained music_notes_neural_net model summary:")
    trained_notes_model.summary()
    print("--- Dummy Neural Network Testing Complete ---")