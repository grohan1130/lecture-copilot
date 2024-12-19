import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Parameters
duration = 10  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate (Hz)

# Automatically select an input device
def get_default_input_device():
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Device has input channels
            return index  # Return the index of the first valid input device
    raise RuntimeError("No input device found with at least one input channel.")

# Find the default input device
try:
    input_device = get_default_input_device()
    print(f"Using input device: {sd.query_devices(input_device)['name']}")
except RuntimeError as e:
    print(e)
    exit(1)

# Ensure the output folder exists
output_folder = "recorded-audio-files"
os.makedirs(output_folder, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_folder, "recorded-output.wav")

# Record audio
print("Recording...")
audio_data = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,  # Use mono for universal compatibility
    dtype=np.int16,
    device=input_device
)
sd.wait()  # Wait until the recording is finished
print("Recording complete. Saving file...")

# Save the audio to the specified file
write(output_file, sample_rate, audio_data)

print(f"File saved as '{output_file}'.")
