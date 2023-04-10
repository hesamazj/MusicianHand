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


def systemID_input_gen_func(duration,fs,steps, pass_chance, min_input, max_input):
    
    samples_factor = int(fs*steps/2)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values+1,4))*min_input
    pass_rand = np.random.uniform(0,1,(number_of_values+1,4))

    for i in range(4):
        for c in range(0, number_of_values+1):
            if pass_rand[c,i] > pass_chance:
                limb_activations[c,i] = min_input+((max_input-min_input)*np.random.uniform(0,1,1)) 
    
    activations = np.ones((4,samples_factor*number_of_values))*0.05
    for i in range(4):
        c = 0
        for j in range(number_of_values):
            for j in np.linspace(limb_activations[j,i],  limb_activations[j+1,i], samples_factor):
                activations[i,c] = j
                c = c+1
    return np.transpose(activations)

def systemID_input_gen_func2(duration,fs,steps, pass_chance, min_input, max_input):
    
    samples_factor = int(fs*steps/2)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values+1,4))*min_input
    pass_rand = np.random.uniform(0,1,(number_of_values+1,4))
    limit = np.zeros(4)
    for c in range(0, number_of_values+1):
        for i in range(4):
            if pass_rand[c,i] > pass_chance and limit[i]<21:
                limit[i] = limit[i]+1
                limb_activations[c,i] = min_input+((max_input-min_input)*np.random.uniform(0,1,1)) 
                break
    
    activations = np.ones((4,samples_factor*number_of_values))*0.05
    for i in range(4):
        c = 0
        for j in range(number_of_values):
            for j in np.linspace(limb_activations[j,i],  limb_activations[j+1,i], samples_factor):
                activations[i,c] = j
                c = c+1
    return np.transpose(activations)

def systemID_input_gen_func3(duration,fs,steps, pass_chance, min_input, max_input):
    
    samples_factor = int(fs*steps/2)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values+1,4))*min_input
    pass_rand = np.random.uniform(0,1,(number_of_values+1,4))
    limit = np.zeros(4)
    x = 1
    for c in range(0, number_of_values+1):
        for i in range(4):   
            if pass_rand[c,i] > pass_chance and limit[i]<(number_of_values/5)and x<i:
                x = i
                limit[i] = limit[i]+1
                limb_activations[c,i] = min_input+((max_input-min_input)*np.random.uniform(0,1,1)) 
                break
    
    activations = np.ones((4,samples_factor*number_of_values))*0.05
    for i in range(4):
        c = 0
        for j in range(number_of_values):
            for j in np.linspace(limb_activations[j,i],  limb_activations[j+1,i], samples_factor):
                activations[i,c] = j
                c = c+1
    return np.transpose(activations)

def systemID_input_gen_func4(duration,fs,steps, min_input, max_input):
    
    samples_factor = int(fs*steps/2)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values+1,4))*min_input
    for i in range(4):
        for c in range(0,number_of_values,4):   
            limb_activations[c+i,i] = min_input+((max_input-min_input)*np.random.uniform(0.5,1,1)) 
    activations = np.ones((4,samples_factor*number_of_values))*0.05
    for i in range(4):
        c = 0
        for j in range(number_of_values):
            for j in np.linspace(limb_activations[j,i],  limb_activations[j+1,i], samples_factor):
                activations[i,c] = j
                c = c+1
    return np.transpose(activations)

def activation_function(rise_fall_duration,steps,alpha,fs):
    rise_fall_factor = rise_fall_duration*fs
    non_active_duration = steps-2*rise_fall_duration
    non_active = non_active_duration*fs
    up = np.linspace(0.05,alpha,int(rise_fall_factor))
    down = np.linspace(0.05,0.05-alpha,int(rise_fall_factor))
    flat_part = np.ones(int(non_active)+1)*0.05
    time_series = np.concatenate((up,flat_part,down))
    return time_series

def music_preprocessing(logdir,clipping,duration=None):
    
    file = logdir
    fs, x = wavfile.read(file)
    x2 = signal.decimate(x[:,1],10)
    fs =fs/10
    if clipping == True:
        N = int((np.shape(x2)[0]-duration*fs)/2)
        x2 = x2[N:-N]
    x2 = x2-np.mean(x2)
    preprocessed_music = x2/max(np.abs(x2))
    
    return fs, preprocessed_music

def spect_preprocessing(logdir,clipping,spect_shape,duration=None):
    
    if clipping:
        fs, preprocessed_music = music_preprocessing(logdir,clipping,duration)  
    else:
        fs, preprocessed_music = music_preprocessing(logdir,clipping)  
    f, t, Sxx = stft(preprocessed_music, fs, nfft = 2048)

    matrix = np.abs(Sxx)[25:525]
    col_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / col_norms

    resampled_array = np.zeros(spect_shape)
    for k in range(np.shape(normalized_matrix)[0]):
        resampled_array[k,:] = np.interp(np.linspace(0, 1, spect_shape[1]), np.linspace(0, 1, np.shape(normalized_matrix)[1]), normalized_matrix[k,:])

    return np.transpose(resampled_array)

def spect_normalization(music_spect):
    music_spect = np.array(music_spect)
    music_spect = music_spect - music_spect.min()
    music_spect = music_spect/music_spect.max()
    return music_spect

def train_test_split(x,y,test_size):
    n = x.shape
    n1 = int(np.round(n[0]*test_size))
    x_test = x[0:n1,:]
    y_test = y[0:n1,:]
    x_train = x[n1:-1,:]
    y_train = y[n1:-1,:]
    return x_train, y_train, x_test, y_test

def systemID_input_gen_func_last(duration,fs,steps, min_input, max_input):
    
    samples_factor = int(fs*steps)
    number_of_values = int(duration/steps)
    limb_activations = np.ones((number_of_values+1,4))*min_input
    for i in range(4):
        for c in range(0,number_of_values,4):   
            limb_activations[c+i,i] = min_input+((max_input-min_input)*np.random.uniform(0.8,1,1)) 
    activations = np.ones((4,samples_factor*number_of_values))*0.05
    for i in range(4):
        c = 0
        for j in range(number_of_values):
            for j in np.linspace(limb_activations[j,i],  limb_activations[j+1,i], samples_factor):
                activations[i,c] = j
                c = c+1
    return np.transpose(activations)

def inverse_mapping_func(music_spect, limb_activations, test_size):# my version of this function
    x_train, y_train, x_test, y_test = train_test_split(music_spect,limb_activations,test_size)
    outputs = np.shape(limb_activations)[-1]
    layers = [
        Dense(units=500, input_shape =(np.shape(x_train[1],)), activation = "sigmoid"),
        Dense(units=100, input_shape=(500,),activation= "sigmoid"),
        Dense(units=outputs, input_shape=(100,), activation = "softmax")
    ]
    
    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy', metrics = ['mse'])
    model.fit(x_train,y_train, epochs=30,validation_data = (x_test,y_test))
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
