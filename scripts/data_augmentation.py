import os
from torchaudio.utils import download_asset
import torch
import torchaudio
import torchaudio.functional as F

def apply_lowpass(wav_path: str):
    r"""
    Applies a low-pass filter to the input WAV file with a frequency threshold of 300 Hz.

    This function takes the path of a WAV file as input, applies a low-pass filter with a 
    frequency cutoff at 300 Hz using the SoX effect, and returns the filtered waveform.

    Parameters:
    -----------
    wav_path : str
        The file path to the input WAV file. The file must have a `.wav` extension.

    Returns:
    --------
    torch.Tensor
        The waveform after applying the low-pass filter.

    Raises:
    -------
    ValueError
        If the input file does not have a `.wav` extension.
    """
    if not wav_path.endswith(".wav"):
        raise ValueError(f"the file: {wav_path} in not a .wav file")
    
    sample_wav = download_asset(wav_path)
    waveform, sample_rate = torchaudio.load(sample_wav, channels_first=False)
    effect = ",".join(
        [
        "lowpass=frequency=300:poles=1",
        ],
    )
    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)
    

def add_noisw_with_snr(wav_path: str):
    r"""
        Adds noise to the input WAV file at different signal-to-noise ratio (SNR) levels.

        This function takes the path of a WAV file as input and applies noise to the waveform
        at three different SNR levels: 20 dB, 10 dB, and 3 dB. It returns the noisy waveforms
        corresponding to each SNR level along with the sample rate.

        Parameters:
        -----------
        wav_path : str
            The file path to the input WAV file. The file must have a `.wav` extension.

        Returns:
        --------
        tuple:
            - torch.Tensor: The noisy waveform with 20 dB SNR.
            - torch.Tensor: The noisy waveform with 10 dB SNR.
            - torch.Tensor: The noisy waveform with 3 dB SNR.
            - int: The sample rate of the original WAV file.
        
        Raises:
        -------
        ValueError
            If the input file does not have a `.wav` extension.
        FileNotFoundError
            If the noise file does not exist.
        """    
    if not wav_path.endswith(".wav"):
        raise ValueError(f"the file: {wav_path} in not a .wav file")
    
    noise_path = "./noise.wav"
    if not os.path.exists(noise_path):
        raise FileNotFoundError(f"The noise file: {noise_path} does not exist")

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
