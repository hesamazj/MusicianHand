import RTBridge as rtb
import numpy as np
from functions import *
import datetime



# Babbling
Experiment_ID = 'v0_02'
babbling_duration = 60
fs = 100
minimum_input = 0.05
maximum_input = 1
steps = 1

# RTBridge Setup

pxiWin10 = "169.254.172.223:5555"
pubPort = "5557"
bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, milliTimeStep= int(10))

#Activations

activations = systemID_input_gen_func2(babbling_duration,fs,steps, minimum_input,maximum_input)
np.savetxt('./Activations/Experiment_'+Experiment_ID+'_activations.txt',activations)
#plt.plot(activations)
#plt.show()  


# Connection
bridge.startConnection()

_ = bridge.sendAndReceive([0.05]*4, 2)
start = time.time()

# Send Data
for activation_set in activations:
    _ = bridge.sendAndReceive(activation_set)
    a = 2
end = time.time()
time_differnce = end - start

print(time_differnce)

_ = bridge.sendAndReceive([0.05]*4, 2)

print("\n\nTest has completed")


