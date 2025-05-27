import numpy as np
import torch
import librosa
from model import VGG

def load_model(checkpoint_path: str):
    if torch.cuda.is_available(): # gpu
        checkpoint = torch.load(checkpoint_path)
        device = torch.device('cuda')
    else: # cpu
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        device = torch.device('cpu')

    model = VGG(vgg_name='VGG16')
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)

    # If model was saved using DataParallel, unwrap it if needed
    if isinstance(model, torch.nn.DataParallel):
        model = model.moduleba
    model.eval()
    return model, device    

def spect_loader(path: str):
    window_size=0.02
    window_stride=0.01
    window='hamming'
    normalize=True
    max_len=101

    y, sr = librosa.load(path=path, sr=None)
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)

    spect, phase = librosa.magphase(D)
    spect_dB = librosa.amplitude_to_db(spect, ref=np.max)

    spect = np.log1p(spect)

    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
        spect_dB = np.hstack((spect_dB, pad))

    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
        spect_dB = spect_dB[:, :max_len]
    
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect_dB = np.resize(spect_dB, (1, spect_dB.shape[0], spect_dB.shape[1]))
    spect = torch.FloatTensor(spect)
    spect_dB = torch.FloatTensor(spect_dB)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        mean_dB = spect_dB.mean()
        std = spect.std()
        std_dB = spect_dB.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
        if std_dB != 0:
            spect_dB.add_(-mean_dB)
            spect_dB.div_(std_dB)            

    return spect, spect_dB