import os
import librosa
from utils import (pre_processing, digits_segmentation, valid_digits,
                   get_training_samples_signal, recognition,
                   TXT_FILE_NOT_FOUND, TXT_FILE_WRONG_EXTENSION,
                   AUDIO_WAV_EXTENSION, TXT_DIGITS_FOUND, TXT_DIGITS_RECOGNIZED)

# Define constants or import them from elsewhere
DEFAULT_SAMPLE_RATE = 32000

def main():
    file_path = "sample-1.wav"

    # Validate the existence of the file
    if not os.path.exists(file_path):
        print(TXT_FILE_NOT_FOUND.format(file_path))
        return

    # Validate the file extension
    _, extension = os.path.splitext(file_path)
    if extension.lower() not in AUDIO_WAV_EXTENSION:
        print(TXT_FILE_WRONG_EXTENSION.format(file_path))
        return

    # Load the file and process
    signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)
    pre_proceed_signal = pre_processing(signal, os.path.splitext(os.path.basename(file_path))[0])

    print("Finding digits...")

    # Segment digits and recognize them
    # Segment digits and recognize them
    samples = digits_segmentation(pre_proceed_signal)
    digits_array = valid_digits(pre_proceed_signal, samples)
    dataset_training_signals = get_training_samples_signal()

    print(TXT_DIGITS_FOUND.format(len(digits_array)))
    # Prints the list that contains all the words found and separates each word
    # with a ", " excluding the last one.
    print()
    print(TXT_DIGITS_RECOGNIZED)
    print(", ".join([str(i) for i in digits_array]))
if __name__ == "__main__":
    main()
