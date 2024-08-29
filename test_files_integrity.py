import os
import wave

def check_audio_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            try:
                with wave.open(filepath, 'rb') as wf:
                    frames = wf.getnframes()
                    if frames == 0:
                        print(f"Empty audio file: {filepath}")
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

# Define the directory containing your audio files
audio_directory = 'mel/DataSets/LJSpeech-1.1/wavs'

# Run the check
check_audio_files(audio_directory)