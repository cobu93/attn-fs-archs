import torch.nn as nn




def build_mlp(
        n_input,
        n_output,
        hidden_units=None,
        activation_fn=None
):

    if hidden_units is None: 
        hidden_units = []
    
    sizes = [n_input] + hidden_units + [n_output]

    if activation_fn is None: 
        activation_fn = nn.Identity()

    if not isinstance(activation_fn, list):
        activation_fn = [activation_fn for _ in range(len(sizes) - 1)]

    assert len(activation_fn)==len(sizes)-1, "The number of activation functions is not valid"
        
    layers = [] 

    for sizes_idx in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[sizes_idx], sizes[sizes_idx + 1]))
        layers.append(activation_fn[sizes_idx])   
            
    return nn.Sequential(*layers)






