import os

import librosa
import numpy as np

class DATASET:
    FoR = 1
    ASVSpoof = 2
    ADD = 3

class PreProcesser:
    def __init__(self):
        self.sample_rate = 16000
        # TODO: fix these values to match AST paper
        self.window_length = 128
        self.hop_length = 256


    # Fake or Real Structure
    # Testing
    #   Fake
    #   Real
    # Training
    #   Fake
    #   Real
    # Validation (dont need this)
    #   Fake
    #   Real

    # ASVSpoof 2021
    # flac
    #   All files

    # ADD
    # Idk need to download first


    def convert_to_spectrogram(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        spec = librosa.stft(audio, win_length=self.window_length, hop_length=self.hop_length)
        return librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    def save_spectrogram(self, file_path, save_path, log_file="../logs/preprocess"):
        spec = self.convert_to_spectrogram(file_path)
        np.save((save_path + ".npy"), spec)

        file_name = os.path.basename(save_path) + ".npy"
        with open(log_file, "a") as f:
            f.write(file_name + "\n")

    @staticmethod
    def load_spectrogram(file_path):
        spec = np.load(file_path + '.npy')
        return spec

    @staticmethod
    def load_folder(file_path):
        specs = []
        files = os.listdir(file_path)
        for file in files:
            file_path = os.path.join(file_path, file)
            spec = PreProcesser.load_spectrogram(file_path)
            specs.append(spec)
        return specs

    def load_keys(self, file_path):
        keys = []

        with open(file_path, 'r') as file:
            for line in file:
                keys.append(line.strip())
        return keys

    # group = Testing, Training
    # class_ = Fake, Real
    def fake_or_real_process(self, file_path, output_path, group: str , class_: str, convert_all, samples):
        path = os.path.join(file_path, group)
        subfolder_path = os.path.join(path, class_)
        file_names = os.listdir(subfolder_path)

        if convert_all:
            samples = len(file_names)

        for i in range(samples):
            audio_path = os.path.join(subfolder_path, file_names[i])

            # Removes the folder name where the dataset is
            # Example "../audio_files_250/Testing/etc" -> "/Testing/etc"
            temp_sub_path = os.path.normpath(subfolder_path).split(os.sep)
            subfolder_parts = temp_sub_path[2:]
            clean_subfolder_path = os.path.join(*subfolder_parts)

            save_path = os.path.normpath(os.path.join(output_path, clean_subfolder_path, file_names[i]))
            self.save_spectrogram(audio_path, save_path)


    def asvspoof_process(self, file_path, output_path, class_, convert_all, samples):
        keys = self.load_keys("../keys/" + class_)
        audio_file_path = os.path.join(file_path, "flac")
        audio_file_list = os.listdir(audio_file_path)
        # remove .flac from name
        file_names = [os.path.splitext(file)[0] for file in audio_file_list]

        if convert_all:
            samples = len(file_names)

        for i in range(samples):
            # Extreme performance
            if file_names[i] in keys:
                print("Audio file: " + file_names[i] + "is in list")
                audio_path = os.path.join(audio_file_path, file_names[i]) +".flac"
                save_path = os.path.join(output_path, class_, file_names[i])
                save_path = os.path.normpath(save_path)
                self.save_spectrogram(audio_path, save_path)


    # Assumes being called inside the ./src folder
    # Kinda cooked but w/e
    def preprocess_dataset(self, dataset, file_path, output_path= "../spectrograms",  convert_all=False, samples=20):
        os.makedirs("../logs", exist_ok=True)

        # clear log file
        with open("../logs/preprocess", "w") as f:
            pass

        if dataset == DATASET.FoR:
            os.makedirs(os.path.join(output_path, "FoR", "Training", "Fake"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "FoR", "Training", "Real"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "FoR", "Testing", "Fake"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "FoR", "Testing", "Real"), exist_ok=True)

            self.fake_or_real_process(file_path, output_path + "/FoR", "Testing", "Fake", convert_all, samples)
            self.fake_or_real_process(file_path, output_path + "/FoR", "Testing", "Real", convert_all, samples)
            self.fake_or_real_process(file_path, output_path + "/FoR", "Training", "Fake", convert_all, samples)
            self.fake_or_real_process(file_path, output_path + "/FoR", "Training", "Real", convert_all, samples)

        if dataset == DATASET.ASVSpoof:
            os.makedirs(os.path.join(output_path, "ASVSpoof", "fake"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "ASVSpoof", "bonafide"), exist_ok=True)

            self.asvspoof_process(file_path, output_path + "/ASVSpoof", "fake", convert_all, samples)
            self.asvspoof_process(file_path, output_path + "/ASVSpoof", "bonafide", convert_all, samples)







