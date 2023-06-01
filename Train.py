import RTBridge as rtb
import numpy as np
from functions import *

# Init

log_directory = './datalog/'
Experiment_ID = 'v0_0000'
log_directory+'Experiment_'+Experiment_ID+'_activations.txt'

# Initial Network

activations = np.loadtxt(os.path.join(log_directory,'_activations.txt'))
music_spect = spect_preprocessing2(os.path.join(log_directory,'Babbling_v0_05.wav'),14800,6758325)
model = inverse_mapping_func3(np.transpose(music_spect),activations,0.1)
music_spect = spect_preprocessing2(os.path.join(log_directory,'target_for_test.wav'),1200,549045)
predicted_activations = model.predict(np.transpose(music_spect))
np.savetxt(os.path.join(log_directory,'predicted_activations.txt'),predicted_activations)

# RTBridge Setup

pxiWin10 = "169.254.172.223:5555"
pubPort = "5557"
bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, milliTimeStep= int(10))

activations = predicted_activations

# Connection

bridge.startConnection()

start = time.time()

# Send Data

for activation_set in activations:
    _ = bridge.sendAndReceive(activation_set)
_ = bridge.sendAndReceive([0.05]*4, 2)

end = time.time()
time_differnce = end - start

# Massages

print("\n\nTest has completed")
print(time_differnce)
