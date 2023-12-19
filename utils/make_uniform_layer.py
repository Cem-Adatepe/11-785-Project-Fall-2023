import torch

def make_uniform_layer(model, layer):
    torch.nn.init.zeros_(model.layers[layer].self_attn.k_proj.weight)
    torch.nn.init.zeros_(model.layers[layer].self_attn.q_proj.weight)
    torch.nn.init.eye_(model.layers[layer].self_attn.v_proj.weight)
    print('Made layer {} uniform'.format(layer))
    return model