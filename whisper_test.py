import whisper
import warnings

def whisper_test(audio_file):
    """
    Transcribes the audio file using Whisper model.
    Args:
        audio_file (str): Path to the audio file to be transcribed.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Ignore FutureWarnings (whisper is reliable)
    warnings.filterwarnings("ignore", category=FutureWarning)  

    # Load small "base" Whisper model
    model = whisper.load_model("base")

    # Transcribe audio file
    result = model.transcribe(audio_file)  
    return result["text"]

def main():
    # Use audio file for testing
    audio_file = "audio-files/harvard.wav"
    transcription = whisper_test(audio_file)
    print(transcription)

if __name__ == "__main__":
    main()
