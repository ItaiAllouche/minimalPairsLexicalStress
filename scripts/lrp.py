import torch
import torch.nn as nn
import numpy as np
from captum.attr import LRP
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from src.GCommandsPytorch.gcommand_loader import spect_loader
import os


def do_lrp(model, wav_path, pre_phones_time_points):
    r"""
    Performs Layer-wise Relevance Propagation (LRP) on a given model using a spectrogram 
    generated from the input WAV file, and visualizes the resulting attribution heatmap.

    This function loads a spectrogram from the provided WAV file, applies LRP using 
    specified rules on the model layers, and visualizes the spectrogram along with the 
    corresponding heatmap to indicate areas of stress (Initial or Final).

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model on which LRP will be performed.
    wav_path : str
        The file path to the input WAV file. The file must have a `.wav` extension.
    pre_phones_time_points : list
        Precomputed time points for phones, used for aligning the spectrogram with the 
        corresponding heatmap.
    
    Raises:
    -------
    FileNotFoundError
        If the WAV file does not exist.
    """

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"The WAV file: {wav_path} does not exist")
    # load spectogram from wav file
    spect, phones_time_points = spect_loader(path=wav_path, window_size=.02, window_stride=.01, window='hamming', normalize=True, phones_time_points=pre_phones_time_points)

    # add betch dimension 
    input = spect.unsqueeze(1)

    # get the predicted class
    target_class = torch.argmax(model(input)).item()
    print(target_class)

    layers = list(model._modules["features"])
    num_layers = len(layers)

    for idx_layer in range(1, num_layers):
        if isinstance(layers[idx_layer], nn.Conv2d):
            if idx_layer <= 1:
                # apply GammaRule on conv layers near the input to preserve the activations as they are
                setattr(layers[idx_layer], "rule", GammaRule())
                pass
            else:
                # apply Alpha1_Beta0_Rule on conv layes near the output to propagates only positive relevance.
                setattr(layers[idx_layer], "rule", Alpha1_Beta0_Rule())


    # apply EpsilonRule on fully connected layers close to the output
    setattr(model._modules["fc1"], "rule", EpsilonRule(epsilon=1e-6))
    setattr(model._modules["fc2"], "rule", EpsilonRule(epsilon=1e-6))
    
    lrp = LRP(model)

    attributions_lrp = lrp.attribute(input, target=target_class)

    # convert attributions to numpy for visualization
    attributions_np = attributions_lrp.squeeze().cpu().detach().numpy()

    # convert your input spectrogram to numpy for visualization
    input_np = input.squeeze().cpu().detach().numpy()

    # fix dimansions for visualization
    input_np_3d = np.expand_dims(input_np, axis=-1)
    attributions_np_3d = np.expand_dims(attributions_np, axis=-1)

    stress = 'Initial' if target_class == 1 else 'Final'
    title = [f"Spectogram With Stress Location: [{stress}]", "Corresponding Heat Map"]

    # visualize the attribution map
    _ = viz.visualize_image_attr_multiple(attr=attributions_np_3d,
                                phones_time_points=phones_time_points,
                                original_image=input_np_3d,
                                methods=["original_image", "heat_map"],
                                show_colorbar=False,
                                signs=["all", "positive"],
                                outlier_perc=2, titles=title,
                                fig_size=(9,5))