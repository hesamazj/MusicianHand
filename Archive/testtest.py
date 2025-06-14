import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# write a function that makes a ramp and then repeats it, I want the name to be ramp_repeat
def ramp_repeat(ramp_duration, fs, steps, minimum_input, maximum_input):
    # make a ramp
    ramp = np.arange(minimum_input, maximum_input, steps)
    ramp = np.append(ramp, np.arange(maximum_input, minimum_input, -steps))
    # repeat the ramp
    ramp_repeat = np.tile(ramp, int(ramp_duration*fs/len(ramp)))
    return ramp_repeat
# write a function that makes pulses that go up with ramps and fall with ramps, I want the name to be pulse_ramp
def pulse_ramp(pulse_duration, fs, steps, minimum_input, maximum_input):
    # make a ramp
    ramp = np.arange(minimum_input, maximum_input, steps)
    pulse = np.ones(int(pulse_duration*fs))
    wave = np.concatenate((ramp, pulse, ramp[::-1]))
    # multiply the ramp by the pulse
    return wave
