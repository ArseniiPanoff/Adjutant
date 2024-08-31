import os
import csv

def prepare_ljspeech_data(ljspeech_dir):
    wav_dir = os.path.join(ljspeech_dir, 'wavs')
    metadata_path = os.path.join(ljspeech_dir, 'metadata.csv')

    data = []

    # Reading the CSV file with metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            wav_file = row[0] + '.wav'
            transcript = row[1]
            wav_path = os.path.join(wav_dir, wav_file)
            data.append((wav_path, transcript))

    return data

def prepare_vctk_data(vctk_dir):
    wav_dir = os.path.join(vctk_dir, 'wav48')
    txt_dir = os.path.join(vctk_dir, 'txt')

    data = []

    # Looping through each speaker
    for speaker in os.listdir(wav_dir):
        speaker_wav_dir = os.path.join(wav_dir, speaker)
        speaker_txt_dir = os.path.join(txt_dir, speaker)

        # Checking if the corresponding directories for the speaker exist
        if os.path.exists(speaker_wav_dir) and os.path.exists(speaker_txt_dir):
            for wav_file in os.listdir(speaker_wav_dir):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(speaker_wav_dir, wav_file)
                    txt_file = wav_file.replace('.wav', '.txt')
                    txt_path = os.path.join(speaker_txt_dir, txt_file)

                    # Checking if the text file exists for this audio
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                        data.append((wav_path, transcript))

    return data

def combine_datasets(ljspeech_data, vctk_data):
    combined_data = ljspeech_data + vctk_data
    return combined_data

# Example usage:
ljspeech_dir = "mel/DataSets/LJSpeech-1.1"
vctk_dir = "mel/DataSets/archive/VCTK-Corpus/VCTK-Corpus"

ljspeech_data = prepare_ljspeech_data(ljspeech_dir)
vctk_data = prepare_vctk_data(vctk_dir)

# Combining the data
combined_data = combine_datasets(ljspeech_data, vctk_data)

# Now combined_data contains all pairs (audio path, transcription)
# You can save this to a file or use it directly for training

# Saving the combined data to a file
with open('combined_dataset.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['audio_path', 'transcription'])
    writer.writerows(combined_data)

csv_file = 'combined_dataset.csv'
output_file = 'filelists/combined_filelist.txt'

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for row in reader:
            out_f.write(f'{row[0]}|{row[1]}\n')