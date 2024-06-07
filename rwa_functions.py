import torch
from tqdm import tqdm
import numpy as np

## Activation Recorder
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
            X = torch.randn(batch_size, num_channels, image_dim, image_dim)
            
            # scale the noise to the desired range
            if zero_mean:
                desired_mean = 0
                desired_std = 127 / 3

                # Scale the data
                X = X * desired_std + desired_mean

                # Ensure the data is within the desired range
                X = torch.clamp(X, -127.5, 127.5)
            else:
                desired_mean = 127.5
                desired_std = 127 / 3

                # Scale the data
                X = X * desired_std + desired_mean

                # Ensure the data is within the desired range
                X = torch.clamp(X, 0, 255)
            
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
            # Reshape X by moving the 2nd dimension to the last
            X = torch.transpose(X, 1, len(X.shape)-1)
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


## Receptive Field Estimations
def RWA(activations, noise, absolute):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    the response-weighted analysis method (modified version of the spike-triggered average method).
    '''
    
    # Initialisations
    num_units = activations.shape[0]
    rf = torch.zeros((num_units, noise.shape[1], noise.shape[2], noise.shape[3]))
    
    # calculate the receptive field using response-weighted average
    with tqdm(total = num_units * noise.shape[0]) as pbar:
        for i in range(num_units):
            for j in range(noise.shape[0]):
                # select mode of calculation (absolute or not)
                # weight input noise by the corresponding output activation
                if absolute:
                    rf[i] += abs(activations[i, j]) * noise[j]
                else:
                    rf[i] += activations[i, j] * noise[j]
                pbar.update(1)
            # normalise by the number of non-zero activations
            if activations[i][activations[i] != 0].shape[0] != 0: # avoid division by zero
                rf[i] /= activations[i][activations[i] != 0].shape[0]
                
    print(f"Shape of receptive field list: {rf.shape}")
    return rf


def CorrRWA(activations, noise):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    pearson correlation - SINGLE CHANNEL only.
    
    Input:
        - activations: activation responses of the units in the specified layer
        - noise: noise patterns used to stimulate the units, single SELECTED channel
    Output:
        - rf: receptive field of each unit in the specified layer
    '''
    
    # Initialisations
    num_units = activations.shape[0]
    rf = torch.zeros((num_units, noise.shape[1], noise.shape[2])) # single channel
    
    # calculate the receptive field using pearson correlation
    with tqdm(total = num_units * noise.shape[1] * noise.shape[2]) as pbar:
        for i in range(num_units):
            # go through each noise pixel
            for j in range(noise.shape[1]): # each row
                for k in range(noise.shape[2]): # each column
                    # compute correlation between activation and noise pixel
                    rf[i, j, k] = np.corrcoef(activations[i].flatten(), noise[:, j, k].flatten())[0, 1] # returns a matrix and so we take the value at [0, 1]
                    pbar.update(1)
    
    return rf


def RWC(act_conv, mu):
    '''
    This function performs response-weighted covariance, returns the covariance matrix of the noise patterns.
    Inputs are CROPPED noise and rf + the activation responses.
    '''
    cov = torch.zeros(mu.shape[0], mu.shape[2], mu.shape[2]) 
    
    # calculate the response-weighted covariance
    with tqdm(total = mu.shape[0] * mu.shape[1]) as pbar:
        for i in range(mu.shape[0]): # each unit
            for j in range(mu.shape[1]): # each noise pattern
                cov[i] += act_conv[i,j] * (mu[i, j].unsqueeze(1) @ mu[i, j].unsqueeze(0))
                pbar.update(1)
            # normalise by the number of non-zero activations
            if act_conv[i][act_conv[i] != 0].shape[0] != 0: # avoid division by zero
                cov[i] /= act_conv[i][act_conv[i] != 0].shape[0]

    return cov


## Receptive field cropping
def CorrMask(noise, act_conv, threshold=0.01):
    '''
    This function ACCUMULATES the Pearson correlation between each noise pixel and the activation
    in order to locate the actual receptive field for neurons in a given layer.
    '''
    
    # initialisations
    correlation = torch.zeros((noise.shape[1], noise.shape[2]))
    
    # calculate pearson correlation coefficient between each noise pixel and activation
    with tqdm(total = act_conv.shape[0] * noise.shape[1] * noise.shape[2]) as pbar:
        for i in range(act_conv.shape[0]): # each neuron
            for j in range(noise.shape[1]):
                for k in range(noise.shape[2]):
                    correlation[j,k] += abs(np.corrcoef(act_conv[i].flatten(), noise[:, j, k].flatten())[0, 1])
                    pbar.update(1)
                    
    # get the average correlation value over the number of units
    correlation = correlation / act_conv.shape[0]
    
    return correlation


def RfCrop(correlation, rf, threshold=0.01):
    '''
    This function crops the receptive field using the correlation map to locate the actual
    receptive field for neurons in a given layer.
    '''
    
    # create binary mask
    mask = correlation.numpy() > threshold
    
    # get the coordinates of the bounding box
    coords = np.argwhere(mask)
    
    # calculate the width and height of the bounding box for coords
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    
    # create a torch tensor to store the cropped receptive fields
    rf_cropped = torch.zeros((rf.shape[0], width, height))
    
    # rf_cropped is the cropped version of rf using the bounding box
    for i in range(rf.shape[0]):
        rf_cropped[i] = rf[i, x0:x1+1, y0:y1+1]
    
    return rf_cropped, mask