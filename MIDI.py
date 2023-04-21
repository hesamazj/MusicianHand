import pyaudio
import mido
import wave
import struct
import time
import numpy as np

# Set up MIDI input
midi_in = mido.open_input('LPK25 0')


# Set up audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1  # mono recording
RATE = 44100  # sample rate
RECORD_SECONDS = 10  # recording duration in seconds
WAVE_OUTPUT_FILENAME = 'output.wav'

# Initialize PyAudio object
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

# Start recording and listening for MIDI input
print('Starting recording and MIDI input...')
audio_frames = []
beat_count = 0
while True:
    # Check for MIDI input
    for msg in midi_in.iter_pending():
        note = msg.note
        velocity = msg.velocity
        frequency = 440 * 2 ** ((note - 69) / 12)
        amplitude = velocity / 127.0
        duration = 0.5  # length of note in seconds
        t = 0
        while t < duration:
            # Generate a sine wave at the frequency of the note
            y = amplitude * np.sin(2 * np.pi * frequency * t)
            # Convert float value to bytes
            data = struct.pack('<h', int(y * 32767))
            # Append to audio frames list
            audio_frames.append(data)
            # Write audio data to stream
            stream.write(data)
            t += 1.0 / RATE
        beat_count += 1

    # Check if we have recorded the desired number of beats
    if beat_count >= 10:  # change this value to record a different number of beats
        break

# Stop audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Open wave file for writing
wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wave_file.setnchannels(CHANNELS)
wave_file.setsampwidth(audio.get_sample_size(FORMAT))
wave_file.setframerate(RATE)
wave_file.writeframes(b''.join(audio_frames))
wave_file.close()

print(f'Recording saved as {WAVE_OUTPUT_FILENAME}.')