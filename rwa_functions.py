import torch
from tqdm import tqdm
import numpy as np

def ActRecorder(layer, net, model, centre_loc, dim, batch_num, batch_size, zero_mean):
    '''
    This function records the activations of the units in a specified layer of the network
    using an ensemble of noise patterns generated from a uniform distribution.
    '''  
    
    # Initialisations
    num_units = net.get_submodule(layer).out_channels
    noise = [] # input noise
    activations = [[] for i in range(num_units)] # output activations
    rloc = centre_loc[0] # row location of the centre of the receptive field
    cloc = centre_loc[1] # column location of the centre of the receptive field
    image_dim = dim[0] # width/height of input image
    num_channels = dim[1] # number of channels in input image
    # Initialise hidden states for Jaco LSTM
    if model == 'jaco':
        hx, cx = torch.zeros((batch_size, 128), dtype=torch.float32), torch.zeros((batch_size, 128), dtype=torch.float32)
    
    # Forward pass to record activations
    with tqdm(total = batch_num) as pbar:
        for batch in range(batch_num):
            # Generate noise
            if zero_mean:
                X = (torch.rand(batch_size, num_channels, image_dim, image_dim) - 0.5) * 255 # zero mean uniform noise, [-127.5, 127.5]
            else:
                X = torch.rand(batch_size, num_channels, image_dim, image_dim) * 255 # uniform noise, [0, 255]
            
            # Forward-pass to get activations
            with torch.no_grad():
                if model == 'jaco':
                    x1, x2, (_, _) = net(0, X, (hx,cx), 1)
                else: # MNIST or fetch
                    x1, x2 = net(X)
            for unit in range(num_units): # record value at the centre pixel
                if layer == 'conv1':
                    activations[unit].append(x1[:, unit, rloc, cloc])
                elif layer == 'conv2':
                    activations[unit].append(x2[:, unit, rloc, cloc])
            noise.append(X)
            pbar.update(1)
        print('Activation recording complete.')
        
    # reshape the activation and noise for analysis and visualisation
    with tqdm(total=num_units) as pbar:
        for unit in range(num_units):
            activations[unit] = torch.cat(activations[unit])
            pbar.update(1)
    activations = torch.stack(activations)
    noise = torch.cat(noise)
    nosie = np.transpose(noise, (0, 2, 3, 1)) # transform to visualisable format
    print(f"Shape of activation response list: {activations.shape}")
    print(f"Shape of noise list: {noise.shape}")
        
    return activations, noise