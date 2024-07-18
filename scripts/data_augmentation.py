import os
from torchaudio.utils import download_asset
import torch
import torchaudio
import torchaudio.functional as F
from generate_dataset_from_TEDLIUM.generate_dataset import get_clapped_spectorgram, get_timestamps

# returns new waveform after applying low-pass filter
def apply_lowpass(wav_path: str):
    if not wav_path.endswith(".wav"):
        raise ValueError(f"the file: {wav_path} in not a .wav file")
    
    sample_wav = download_asset(wav_path)
    waveform, sample_rate = torchaudio.load(sample_wav, channels_first=False)
    effect = ",".join(
        [
        "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
        # "atempo=0.8",  # reduce the speed
        # "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|decays=0.3"
        # Applying echo gives some dramatic feeling
        ],
    )
    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector(waveform, sample_rate)
    
# add noise to the speech waveform at 10,20 and 3 SNR levels.
def add_noisw_with_snr(wav_path: str):
    if not wav_path.endswith(".wav"):
        raise ValueError(f"the file: {wav_path} in not a .wav file")
    
    noise_path = "./noise.wav"
    sample_noise = download_asset(noise_path)
    noise, _ = torchaudio.load(sample_noise)

    sample_wav = download_asset(wav_path)
    speech, sample_rate = torchaudio.load(sample_wav)
    noise = noise[:, : speech.shape[1]]

    # define SNR values and apply noise
    snr_dbs = torch.tensor([20, 10, 3])
    noisy_speeches = F.add_noise(speech, noise, snr_dbs)

    # separate noisy speeches for each SNR level
    noisy_speech_20db = noisy_speeches[0:1]
    noisy_speech_10db = noisy_speeches[1:2]
    noisy_speech_3db = noisy_speeches[2:3]
    return noisy_speech_20db, noisy_speech_10db, noisy_speech_3db, sample_rate   
