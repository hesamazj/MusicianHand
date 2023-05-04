import RTBridge as rtb
import numpy as np
from functions import *
import datetime




# RTBridge Setup

pxiWin10 = "169.254.172.223:5555"
pubPort = "5557"
bridge = rtb.BridgeSetup(pubPort, pxiWin10, rtb.setups.hand_4_4, milliTimeStep= int(10))

#Activations

activations = np.loadtxt('./train_activations/activation_2.txt')


# Connection
bridge.startConnection()

start = time.time()

# Send Data
for activation_set in activations:
    _ = bridge.sendAndReceive(activation_set)
    a = 2
end = time.time()
time_differnce = end - start
_ = bridge.sendAndReceive([0.05]*4, 2)

print(time_differnce)


print("\n\nTest has completed")


