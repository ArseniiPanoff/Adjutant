import subprocess
import os

def resample_audio_in_place(file_list_path, target_sampling_rate):
    with open(file_list_path, 'r') as file_list:
        for line in file_list:
            if not line.strip():  # Skip empty lines
                continue

            # Parse file path and transcription
            parts = line.strip().split('|')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            audio_path = parts[0].strip()  # Remove any extra whitespace

            # Check if the audio file exists
            if not os.path.isfile(audio_path):
                print(f"File not found: {audio_path}")
                continue

            # Create a temporary path for resampling
            temp_path = audio_path + ".temp.wav"

            # Debugging output
            print(f"Resampling {audio_path} to {temp_path} with target sample rate {target_sampling_rate}")

            # Resample the audio file
            command = [
                'sox', audio_path,
                '-r', str(target_sampling_rate),
                temp_path
            ]
            try:
                subprocess.run(command, check=True)

                # Check if the temp file was created and is not empty
                if os.path.getsize(temp_path) > 0:
                    # Replace the original file with the resampled file
                    os.replace(temp_path, audio_path)
                    print(f"Resampled and updated {audio_path}")
                else:
                    print(f"Temp file is empty, skipping replacement: {temp_path}")
                    os.remove(temp_path)
            except subprocess.CalledProcessError as e:
                print(f"Error resampling {audio_path}: {e}")
                # Clean up temporary file if an error occurs
                if os.path.isfile(temp_path):
                    os.remove(temp_path)

    print("All files processed.")


def clean_up_temp_files(file_list_path):
    # Scan the file list for all file paths
    with open(file_list_path, 'r') as file_list:
        for line in file_list:
            if not line.strip():  # Skip empty lines
                continue

            # Parse file path
            parts = line.strip().split('|')
            if len(parts) != 2:
                continue

            audio_path = parts[0].strip()
            temp_path = audio_path + ".temp.wav"

            # Check and remove temp files
            if os.path.isfile(temp_path):
                os.remove(temp_path)
                print(f"Removed leftover temp file: {temp_path}")


# Define parameters
file_list_path = 'filelists/combined_filelist.txt'
target_sampling_rate = 22050

# Run the resampling function
resample_audio_in_place(file_list_path, target_sampling_rate)

# Clean up any remaining .temp.wav files
clean_up_temp_files(file_list_path)

