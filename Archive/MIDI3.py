import pyaudio
import mido
import wave
import struct
import numpy as np
import datetime

def note_to_freq(note, concert_A=440.0):
    f = (2.0 ** ((note - 69) / 12.0)) * concert_A
    return f

# Set up MIDI input
#midi_in = mido.open_input('LPK25 0') #Windows
midi_in = mido.open_input('LPK25') #Mac

# Set up audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1  # mono recording
RATE = 44100  # sample rate
RECORD_SECONDS = 60  # recording duration in seconds
WAVE_OUTPUT_FILENAME = 'output2.wav'
total_duration = 0

# Initialize PyAudio object
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

# Start recording and listening for MIDI input
print('Starting recording and MIDI input...')
midi_messages = []

while True:
    # Check for MIDI input
    for msg in midi_in.iter_pending():
        print(msg)
        if msg.type == 'note_on':
             start = datetime.datetime.now()
             print(msg)
        elif msg.type == 'note_off':
            end = datetime.datetime.now()
            end = end.minute*60 + end.second+end.microsecond/1000000
            start = start.minute*60 + start.second+start.microsecond/1000000
            duration = end - start            
            msg.time = duration
            total_duration += duration
        midi_messages.append(msg)

    if total_duration>RECORD_SECONDS:
        break

midi_in.close()   


audio = pyaudio.PyAudio()

print('Starting recording and MIDI input...')
audio_frames = []

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

for msg in midi_messages:
    if msg.type == 'note_on':

        note = msg.note
        velocity = msg.velocity
        frequency = note_to_freq(note)
        amplitude = velocity / 127.0
    elif msg.type == 'note_off':
        duration = msg.time
        t = 0
        while t < duration:
            y = amplitude * np.sin(2 * np.pi * frequency * t)
            data = struct.pack('<h', int(y * 32767))
            audio_frames.append(data)
            stream.write(data)
            t += 1.0 / RATE



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