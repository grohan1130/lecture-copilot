import whisper
import warnings
import os

def whisper_test(audio_file):
    """
    Transcribes the audio file using Whisper model.
    Args:
        audio_file (str): Path to the audio file to be transcribed.
    Returns:
        str: Transcribed text from the audio file.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = whisper.load_model("base")
    print(f"Transcribing audio file: {audio_file} ...")
    result = model.transcribe(audio_file)
    return result["text"]

def save_transcription(transcription, output_file):
    """
    Saves the transcription to a text file.
    Args:
        transcription (str): Transcribed text to save.
        output_file (str): Path to the text file to save the transcription.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(transcription)
    print(f"Transcription saved to: {output_file}")

if __name__ == "__main__":
    # If this file is run standalone, it demonstrates functionality.
    audio_file = input("Enter the path to the audio file: ").strip()
    if not os.path.isfile(audio_file):
        print(f"Error: The file '{audio_file}' does not exist.")
    else:
        transcription = whisper_test(audio_file)
        print("\nTranscription:")
        print(transcription)

        output_file = input("Enter the path to save the transcription: ").strip()
        save_transcription(transcription, output_file)
