import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# --- 1. Load the Pre-processed Spike Data ---
# Load the data we saved from the pipeline
try:
    spike_data = np.load('sub-002_spike_data.npy')
except FileNotFoundError:
    print("Error: 'sub-002_spike_data.npy' not found.")
    print("Please run the 'run_pipeline.py' script first to generate this file.")
    exit()

sfreq = 500.0  # The sampling frequency of our data (500 Hz)

# Select a single epoch and a single channel to test
epoch_to_test = 10
channel_to_test = 0
input_spike_train = spike_data[epoch_to_test, channel_to_test, :]

# --- 2. Convert Data into Brian2 Spike Format (Corrected) ---
# Find the time steps where spikes occurred in our data array
spike_timesteps = np.where(input_spike_train == 1)[0]

# Create the array of spike times in seconds
spike_times = spike_timesteps / sfreq * second

# Create the array of neuron indices. Since we have only one input neuron
# (at index 0), this array must be all zeros and the same size as spike_times.
neuron_indices = np.zeros(len(spike_timesteps), dtype=int)

# Create the input neuron group
input_group = SpikeGeneratorGroup(1, neuron_indices, spike_times)

# --- 3. Define the LIF Neuron Model ---
# These are the standard equations for a Leaky Integrate-and-Fire neuron
tau = 20*ms       # Membrane time constant
v_rest = -70*mV   # Resting potential
v_thresh = -50*mV # Spike threshold
v_reset = -75*mV  # Reset potential after a spike

# The differential equation for membrane potential 'v'
eqs = '''
dv/dt = (v_rest - v)/tau : volt (unless refractory)
'''

# Create the output neuron group (a single neuron)
output_neuron = NeuronGroup(1, eqs, threshold='v > v_thresh', reset='v = v_reset',
                            refractory=5*ms, method='exact')
output_neuron.v = v_rest # Initialize membrane potential

# --- 4. Connect the Input to the Neuron (Create Synapses) ---
synapse_weight = 5.5*mV # How much each input spike increases the neuron's voltage

# Create the connection
synapses = Synapses(input_group, output_neuron, on_pre='v_post += synapse_weight')
synapses.connect(i=0, j=0)

# --- 5. Set up Monitors to Record Data ---
# Monitor the membrane voltage of the output neuron
state_monitor = StateMonitor(output_neuron, 'v', record=True)
# Monitor the spikes produced by the output neuron
spike_monitor = SpikeMonitor(output_neuron)

# --- 6. Run the Simulation ---
simulation_duration = len(input_spike_train) / sfreq * second
run(simulation_duration)

# --- 7. Plot the Results ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle('LIF Neuron Simulation with EEG Spike Data', fontsize=16)

# Plot 1: Input Spikes from EEG data
axes[0].vlines(spike_times/ms, ymin=0, ymax=1, color='black', label='Input Spikes')
axes[0].set_ylabel('Input')
axes[0].set_yticks([])
axes[0].legend(loc='upper right')

# Plot 2: Neuron's Membrane Potential
axes[1].plot(state_monitor.t/ms, state_monitor.v[0]/mV, label='Membrane Potential')
axes[1].axhline(v_thresh/mV, ls='--', color='red', label='Threshold')
axes[1].set_ylabel('Voltage (mV)')
axes[1].legend(loc='upper right')

# Plot 3: Output Spikes from our LIF Neuron
axes[2].vlines(spike_monitor.t/ms, ymin=0, ymax=1, color='blue', label='Output Spikes')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Output')
axes[2].set_yticks([])
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()