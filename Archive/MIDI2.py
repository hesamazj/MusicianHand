import pyaudio
import mido
import wave
import struct
import time
import numpy as np

def note_to_freq(note, concert_A=440.0):
    f = (2.0 ** ((note - 69) / 12.0)) * concert_A
    return f

# Set up MIDI input
midi_in = mido.open_input('LPK25 0')


# Set up audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1  # mono recording
RATE = 44100  # sample rate
RECORD_SECONDS = 5  # recording duration in seconds
WAVE_OUTPUT_FILENAME = 'output2.wav'
total_duration = 0

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
        if msg.type == 'note_on':
            start_time = time.time()

            note = msg.note
            velocity = msg.velocity
            frequency = note_to_freq(note)
            amplitude = velocity / 127.0
        elif msg.type == 'note_off':
            stop_time = time.time()
            duration = stop_time-start_time
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
            total_duration += duration

    # Check if we have recorded the desired number of beats
    if total_duration>=RECORD_SECONDS:  # change this value to record a different number of beats
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