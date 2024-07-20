"""
This file contains the implementation of the do_lrp function, which performs Layer-wise Relevance Propagation (LRP) on a deep learning model using the Captum library.
Captum is a library for model interpretability, providing various algorithms to explain the predictions of PyTorch models. 
LRP is specifically used to understand which parts of the input data contributed most to the model's decision.
"""

import torch
import torch.nn as nn
import numpy as np
from captum.attr import LRP
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from src.GCommandsPytorch.gcommand_loader import spect_loader


def do_lrp(model: nn.Module, wav_path: str):
    # load spectogram from wav file
    spect = spect_loader(path=wav_path, window_size=.02, window_stride=.01, window='hamming', normalize=True)

    # add betch dimension 
    input = spect.unsqueeze(1)

    # get the predicted class
    target_class = torch.argmax(model(input)).item()
    print(target_class)

    layers = list(model._modules["features"])
    num_layers = len(layers)

    # for idx_layer in range(1, num_layers):
    #     if idx_layer <= 16:
    #         setattr(layers[idx_layer], "rule", GammaRule())
    #     else:
    #         setattr(layers[idx_layer], "rule", EpsilonRule())
            
    # setattr(model._modules["fc1"], "rule", EpsilonRule(epsilon=0))
    # setattr(model._modules["fc2"], "rule", EpsilonRule(epsilon=0))
    
    lrp = LRP(model)

    attributions_lrp = lrp.attribute(input, target=target_class)

    # convert attributions to numpy for visualization
    attributions_np = attributions_lrp.squeeze().cpu().detach().numpy()

    # convert your input spectrogram to numpy for visualization
    input_np = input.squeeze().cpu().detach().numpy()

    # fix dimansions for visualization
    input_np_3d = np.expand_dims(input_np, axis=-1)
    attributions_np_3d = np.expand_dims(attributions_np, axis=-1)

    # Visualize the attribution map using visualize_image_attr
    _ = viz.visualize_image_attr_multiple(attr=attributions_np_3d,
                                original_image=input_np_3d,
                                methods=["original_image", "heat_map"],
                                show_colorbar=True,
                                signs=["all", "positive"],
                                outlier_perc=2)