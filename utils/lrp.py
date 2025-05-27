import numpy as np
import torch
from captum.attr import LRP
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule, IdentityRule
import torch.nn as nn

def do_lrp(model, spect, device):
    input = spect.unsqueeze(1)
    input = input.to(device)

    prediction_dict = {0: "FS", 1: "IS"}
    target_class = torch.argmax(model(input.requires_grad_())).item()
    print(f"Predicion: {prediction_dict[target_class]}")
    
    layers = list(model._modules['features'])
    num_layers = len(layers)

    for idx_layer in range(1, num_layers):
        if isinstance(layers[idx_layer], nn.Conv2d):
            if idx_layer <= 2:
                setattr(layers[idx_layer], "rule", IdentityRule())
            else:
                setattr(layers[idx_layer], "rule",  Alpha1_Beta0_Rule())


    setattr(model._modules["fc1"], "rule", EpsilonRule(epsilon=1e-9))
    setattr(model._modules["fc2"], "rule", EpsilonRule(epsilon=1e-9))

    lrp = LRP(model)
    attributions_lrp = lrp.attribute(input, target=target_class)

    # convert attributions to numpy for visualization
    attributions_np = attributions_lrp.squeeze().cpu().detach().numpy()
    attributions_np_3d = np.expand_dims(attributions_np, axis=-1)

    heatmap = viz._normalize_attr(attributions_np_3d, 'positive', 2, reduction_axis=None)

    return heatmap[:, :50]