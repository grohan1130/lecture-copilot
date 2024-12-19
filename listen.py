import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import transcribe  # Import the transcribe module

def get_default_input_device():
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            return index
    raise RuntimeError("No input device found with at least one input channel.")

def record_audio(duration, sample_rate, input_device_index, output_file):
    print("Recording...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        device=input_device_index
    )
    sd.wait()
    print("Recording complete.")

    # Ensure the folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the recording
    write(output_file, sample_rate, audio_data)
    print(f"Audio file saved to: {output_file}")

def main():
    duration = 3600  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate in Hz
    audio_output_file = "recorded-audio-files/recorded-output.wav"
    transcription_output_file = "transcribed-audio-files/transcription.txt"

    try:
        input_device = get_default_input_device()
        print(f"Using input device: {sd.query_devices(input_device)['name']}")

        # Record the audio
        record_audio(duration, sample_rate, input_device, audio_output_file)

        # Automatically call the transcription function
        print("Starting transcription...")
        transcription = transcribe.whisper_test(audio_output_file)  # Call transcription
        print("\nTranscription:")
        print(transcription)

        # Save the transcription to the specified folder
        transcribe.save_transcription(transcription, transcription_output_file)

    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    main()
