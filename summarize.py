from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OpenAI API key not found. Please set it in the .env file.")

# Set the OpenAI API key
openai.api_key = api_key

def summarize_text(text):
    """
    Summarizes the given text using OpenAI's GPT-4 model.
    Args:
        text (str): The text to summarize.
    Returns:
        str: The summarized text.
    """
    try:
        print("Generating summary...")
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150,  # Adjust the token limit for summary length
            temperature=0.5  # Controls randomness of the output
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred during summarization."

def read_transcription(file_path):
    """
    Reads the transcription text from a file.
    Args:
        file_path (str): Path to the transcription text file.
    Returns:
        str: Content of the transcription file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def save_summary(summary, output_file):
    """
    Saves the summary to a text file.
    Args:
        summary (str): Summary to save.
        output_file (str): Path to the output text file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(summary)
    print(f"Summary saved to: {output_file}")

def main():
    # Input and output file paths
    transcription_file = "transcribed-audio-files/transcription.txt"
    summary_output_file = "transcription-summaries/summary.txt"

    try:
        # Read the transcription file
        transcription = read_transcription(transcription_file)

        # Generate the summary
        summary = summarize_text(transcription)
        print("\nSummary:")
        print(summary)

        # Save the summary
        save_summary(summary, summary_output_file)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
