from scipy.signal import stft
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
from re import L
from random import sample
import os
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import activation,Dense
import math
import zmq
import RTBridge as rtb
from scipy.io import wavfile
import scipy.signal as signal
from keras import backend as K
from scipy.stats import beta
from scipy.fft import fft
import pyaudio
import mido
import wave
import struct
import time
#from sklearn.preprocessing import StandardScaler

def systemID_input_gen_func(duration,fs,steps, min_input, max_input):
    
    samples_factor = int(fs*steps)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values,4))*min_input
    for i in range(4):
        for c in range(0,number_of_values,4):   
            limb_activations[c+i,i] = min_input+((max_input-min_input)*np.random.uniform(0.9,1,1)) 
    activations = np.ones((4,samples_factor*(number_of_values+1)))*min_input
    x = np.linspace(-0.25, 0.25, samples_factor)
    a = 30

    for i in range(4):
        activations_temp = np.ones(samples_factor)*min_input
        for j in range(number_of_values):

            if limb_activations[j,i]>min_input:
                y = beta.pdf(np.linspace(0,1,50),3,5)
                y = (limb_activations[j,i])*(y/max(y))*(max_input-min_input)+0.05
            else:
                y = np.ones(samples_factor)*min_input
            activations_temp = np.concatenate((activations_temp,y))
        activations[i,:] = activations_temp
    return np.transpose(activations)


def systemID_input_gen_func2(duration,fs,steps, min_input, max_input):
    
    samples_factor = int(fs*steps)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values,4))*min_input
    for i in range(4):
        for c in range(0,number_of_values,4):   
            limb_activations[c+i,i] = min_input+((max_input-min_input)*np.random.uniform(0.5,1,1)) 
    activations = np.ones((4,samples_factor*(number_of_values+1)))*min_input

    for i in range(4):
        activations_temp = np.ones(samples_factor)*min_input
        for j in range(number_of_values):
            y = np.ones(samples_factor)*limb_activations[j,i]
            activations_temp = np.concatenate((activations_temp,y))
        activations[i,:] = activations_temp
    return np.transpose(activations)

def systemID_input_gen_func3(duration,fs,steps, min_input, max_input):
    
    n = int(fs*steps/5)
    samples_factor = int(fs*steps)-n
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values,4))*min_input
    for i in range(4):
        for c in range(0,number_of_values,4):   
            limb_activations[c+i,i] = min_input+((max_input-min_input)*np.random.uniform(0.5,1,1)) 
    activations = np.ones((4,(samples_factor+n)*(number_of_values)))*min_input

    for i in range(4):
        activations_temp = []
        for j in range(number_of_values):
            y = np.concatenate((np.ones(samples_factor)*limb_activations[j,i],np.ones(n)*min_input))
            activations_temp = np.concatenate((activations_temp,y))
        activations[i,:] = activations_temp
    return np.transpose(activations)

def spect_preprocessing(logdir,clipping,spect_length,duration=None):

    fs, music = wavfile.read(logdir)
    #x = music[:,0]
    x = music

    if clipping == True:
        N = int((np.shape(x)[0]-duration*fs))
        x= x[0:-N]
    music = x
    samples_number = int(fs*0.1)
    epochs_number = int(np.shape(music)[0]/samples_number)
    music = music[0:int(np.floor(np.shape(music)[0]/samples_number)*samples_number)]

    grouped_matrix_music = music.reshape(-1,int(samples_number))
    N = int(np.floor(np.shape(grouped_matrix_music)[1]/2))
    fft_matrix = [[i+1 for i in range(N)]]
    for k in range(epochs_number):
        array = grouped_matrix_music[k,:]-np.mean(grouped_matrix_music[k,:])
        fft_array = np.abs(fft(array)[0:N])
        fft_array = np.reshape(fft_array,(1,N))
        fft_matrix = np.append(fft_matrix,fft_array,0)


        matrix = fft_matrix[1:,0:400]
        #max_abs_values = [max([abs(element) for element in row]) for row in matrix]
        max_abs_value = max([abs(element) for row in matrix for element in row])

        normalized_matrix = np.zeros_like(matrix)

        for i in range(np.shape(matrix)[0]):
            for j in range(np.shape(matrix)[1]):
                normalized_matrix[i,j] = matrix[i,j]/max_abs_value
            #normalized_matrix = matrix
  
        spect_shape = (spect_length,np.shape(normalized_matrix)[1])
        resampled_array = np.zeros(spect_shape)
        for k in range(np.shape(normalized_matrix)[1]):
            resampled_array[:,k] = np.interp(np.linspace(0, 1, spect_shape[0]), np.linspace(0, 1, np.shape(normalized_matrix)[0]), normalized_matrix[:,k])

            #resampled_array = np.transpose(resampled_array)

    return resampled_array


def train_test_split(x,y,test_size):
    n = x.shape
    n1 = int(np.round(n[0]*test_size))
    x_test = x[0:n1,:]
    y_test = y[0:n1,:]
    x_train = x[n1:-1,:]
    y_train = y[n1:-1,:]
    return x_train, y_train, x_test, y_test

def inverse_mapping_func(music_spect, limb_activations, test_size):# my version of this function
    x_train, y_train, x_test, y_test = train_test_split(music_spect,limb_activations,test_size)
    outputs = np.shape(limb_activations)[-1]
    layers = [
        Dense(units=50, input_shape =(np.shape(x_train[1],)), activation = "sigmoid"),
        Dense(units=4, input_shape=(50,),activation= "softmax"),

    ]
    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy', metrics = ['mse'])
    model.fit(x_train,y_train, epochs=10,validation_data = (x_test,y_test))
    return model 


def inverse_mapping_func2(music_spect, limb_activations, test_size):# my version of this function
    x_train, y_train, x_test, y_test = train_test_split(music_spect,limb_activations,test_size)
    outputs = np.shape(limb_activations)[-1]
    layers = [
        Dense(units=50, input_shape =(np.shape(x_train[1],)), activation = "sigmoid"),
        Dense(units=4, input_shape=(50,),activation= "softmax"),
        tf.keras.layers.Lambda(lambda x: tf.argmax(x, axis=1)),

    ]
    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy', metrics = ['mse'])
    model.fit(x_train,y_train, epochs=10,validation_data = (x_test,y_test))
    return model 


def network_refinement(x1,y1,x2,y2):
    input_updated = np.concatenate((x1,x2))
    output_updated = np.concatenate((y1,y2))
    model = inverse_mapping_func(input_updated, output_updated, 0.2)  
    return model

def gen_activations_from_spect2(music_spect,model,frequency_samples):
    x = music_spect[0:-1,0:frequency_samples]
    limb_activations = model.predict(x)
    return limb_activations

def gen_activations_from_spect(music_spect,model):
    x = music_spect
    limb_activations = model.predict(x)
    return limb_activations

def babbling(babbling_duration, number_of_fingers, dt):
    number_of_babbling_samples = int(np.round(babbling_duration/dt))
    limb_activations = np.empty((number_of_babbling_samples+1,number_of_fingers))
    for c in range(number_of_fingers):
        limb_activations[0:-1,c]=systemID_input_gen_func(number_of_babbling_samples,0.5,0.2,.99)
    # use rt bridge to send the activations to the hand and record the .wav file, then make the spectrogram of it.
    return limb_activations

def degree2Excurs(degrees, diameter):
	return (math.pi*diameter)*(degrees/360)

def act_to_csv(limb_activations,version_number):
    np.savetxt('activations'+str(version_number)+'.csv',limb_activations,delimiter=";")

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def spect_gen(audio):
    spect = stft(audio, window='hann', noverlap= 20, nfft=next_power_of_2(len(audio)))
    return spect

# Higher level codes
def babble(babbling_duration, pxi_ip,fs):
    pxiWin10 = pxi_ip
    pubPort = "5557"
    bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, 20)
    activations = systemID_input_gen_func(babbling_duration, fs/10,0.5,0.5,0.05,1)
    bridge.startConnection()

    for activation_set in activations:
        bridge.sendAndReceive(activation_set)

    return activations

def tarining(inverse_map):
    fs, x = wavfile.read('/Users/hesamazad/Downloads/PhD/Piano Hand Project/Target.wav')
    x2 = signal.decimate(x[:,1],10)
    n = np.shape(x2)[0]
    f, t, Sxx = stft(x2, fs/10,nfft= n)
    bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, 20)
    music_spect_training = spect_normalization(np.abs(Sxx))
    activations_training = gen_activations_from_spect(music_spect_training, inverse_map)
    bridge.startConnection()
    for activation_set in activations_training:
        _ = bridge.sendAndReceive(activation_set)
        _ = bridge.sendAndReceive([0.05]*4, 2)
    return activations

def note_to_freq(note, concert_A=440.0):
    f = (2.0 ** ((note - 69) / 12.0)) * concert_A
    return f

def Record(record_duration,name):
    midi_in = mido.open_input('LPK25 0')

    FORMAT = pyaudio.paInt16
    CHANNELS = 1  
    RATE = 44100  
    RECORD_SECONDS = record_duration 
    WAVE_OUTPUT_FILENAME = name
    total_duration = 0

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    print('Starting recording and MIDI input...')
    audio_frames = []
    while True:

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
                    y = amplitude * np.sin(2 * np.pi * frequency * t)
                    data = struct.pack('<h', int(y * 32767))
                    audio_frames.append(data)
                    stream.write(data)
                    t += 1.0 / RATE
                total_duration += duration

        if total_duration>=RECORD_SECONDS:  
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

	
    
