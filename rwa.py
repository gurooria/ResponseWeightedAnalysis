## this file contains all the functions for response-weighted analysis
import torch
from tqdm import tqdm
import numpy as np


## activation recorders

def JacoActRecorder(layer, net, batch_num, batch_size, input_x=64, input_y=64, zero_mean=True):
    '''
    This function records the activations of the units in a specified layer of the network
    using an ensemble of noise patterns generated from a uniform distribution.
    '''  

    # initialisations
    num_units = net.get_submodule(layer).out_channels
    hx, cx = torch.zeros((batch_size, 128), dtype=torch.float32), torch.zeros((batch_size, 128), dtype=torch.float32)
    noise = [] # input noise patterns
    act_conv = [[] for i in range(num_units)] # responses of conv units to be recorded
    
    # get the centre locations of activation map
    net.eval()
    X = (torch.rand(batch_size, 3, input_x, input_y) - 0.5) * 255 # colored noise pattern ensemble, zero mean
    x1,x2,(_,_) = net(0, X, (hx,cx), 1) # forward pass to get the activation map
    if layer == 'conv1':
        nR,nC = x1[0,0,:,:].shape
    elif layer == 'conv2':
        nR,nC = x2[0,0,:,:].shape
    rloc, cloc = int(round(float(nR)/2.0)), int(round(float(nC)/2.0))
    
    # forward pass to record activations
    with tqdm(total = batch_num) as pbar:
        for i in range(batch_num):
            # set noise pattern distribution
            if zero_mean:
                X = (torch.rand(batch_size, 3, input_x, input_y) - 0.5) * 255 # zero mean noise, [-127.5, 127.5]
            else:
                X = torch.rand(batch_size, 3, input_x, input_y) * 255 # non-zero mean noise, [0, 255]
            # forward pass to get activation maps
            with torch.no_grad():
                x1,x2,(_,_) = net(0, X, (hx,cx), 1)
            # record the activations for each unit using the same noise ensemble
            for j in range(num_units):
                if layer == 'conv1':
                    act_conv[j].append(x1[:, j, rloc, cloc]) # records centre value of activation map
                elif layer == 'conv2':
                    act_conv[j].append(x2[:, j, rloc, cloc])
            noise.append(X) # records noise pattern
            pbar.update(1)
    print("Activation recording completed")
    
    # reshape the activation and noise for analysis and visualisation
    with tqdm(total = num_units) as pbar:
        for i in range(num_units):
            act_conv[i] = torch.cat(act_conv[i])
            pbar.update(1)
    act_conv = torch.stack(act_conv)
    noise = torch.cat(noise)
    noise = np.transpose(noise, (0, 2, 3, 1)) # transform to visualisation format in rgb state
    print(f"Shape of activation response list: {act_conv.shape}")
    print(f"Shape of noise list: {noise.shape}")
    
    return act_conv, noise

def MnistActRecorder(layer, net, batch_num, batch_size, input_x=28, input_y=28, zero_mean=True):
    '''
    This function records the activations of the units in a specified layer of the network
    using an ensemble of noise patterns generated from a uniform distribution.
    '''   
    
    # initialisations
    num_units = net.get_submodule(layer).out_channels
    noise = [] # input noise patterns
    act_conv = [[] for i in range(num_units)] # responses of conv units to be recorded
    
    # get the centre locations of activation map
    net.eval()
    X = (torch.rand(batch_size, 1, input_x, input_y) - 0.5) * 255 # noise pattern ensemble, zero mean
    x1, x2 = net(X) # forward pass to get the activation map
    if layer == 'conv1':
        nR,nC = x1[0,0,:,:].shape
    elif layer == 'conv2':
        nR,nC = x2[0,0,:,:].shape
    rloc, cloc = int(round(float(nR)/2.0)), int(round(float(nC)/2.0))
    
    # forward pass to record activations
    with tqdm(total = batch_num) as pbar:
        for i in range(batch_num):
            # set noise pattern distribution
            if zero_mean:
                X = (torch.rand(batch_size, 1, input_x, input_y) - 0.5) * 255
            else:
                X = torch.rand(batch_size, 1, input_x, input_y) * 255 # non-zero mean noise, [0, 255]
            # forward pass
            with torch.no_grad():
                x1, x2 = net(X)
            # record the activations for each unit using the same noise ensemble
            for j in range(num_units):
                if layer == 'conv1':
                    act_conv[j].append(x1[:, j, rloc, cloc])
                elif layer == 'conv2':
                    act_conv[j].append(x2[:, j, rloc, cloc])
            noise.append(X)
            pbar.update(1)
    print("Activation recording completed")
            
    # reshape the activation and noise for analysis and visualisation
    with tqdm(total=num_units) as pbar:
        for i in range(num_units):
            act_conv[i] = torch.cat(act_conv[i])
            pbar.update(1)
    act_conv = torch.stack(act_conv)
    noise = torch.cat(noise)
    noise = np.transpose(noise, (0, 2, 3, 1)) # transform to visualisation format in rgb state
    print(f"Shape of activation response list: {act_conv.shape}")
    print(f"Shape of noise list: {noise.shape}")
            
    return act_conv, noise


## response-weighted analysis

def RWA(act_conv, noise, absolute=False):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    the response-weighted analysis method (modified version of the spike-triggered average method).
    '''
    
    # initialisations
    num_units = act_conv.shape[0]
    rf = torch.zeros((num_units, noise.shape[1], noise.shape[2], noise.shape[3]))
    
    # calculate the receptive field using response-weighted average
    with tqdm(total = num_units * noise.shape[0]) as pbar:
        for i in range(num_units):
            count = 0 # count the number of non-zero activations
            for j in range(noise.shape[0]):
                # select mode of calculation (absolute or not)
                if absolute:
                    rf[i] += abs(act_conv[i, j]) * noise[j]
                else:
                    rf[i] += act_conv[i, j] * noise[j]
                # check if the activation is non-zero to record the count
                if act_conv[i, j] != 0:
                    count += 1
                pbar.update(1)
            # normalise by the number of non-zero activations
            if count != 0:
                rf[i] /= count

    print(f"Shape of receptive field list: {rf.shape}")
    
    return rf

def RWC(act_conv, noise, rf, zero_mean=True):
    '''
    This function performs response-weighted covariance.
    '''
    
    # initialisations
    num_units = act_conv.shape[0]
    cov = torch.zeros(num_units, noise.shape[1]*noise.shape[2], noise.shape[1]*noise.shape[2])

    # reformat rf data to fit with noise
    if zero_mean:
        rf1 = ((rf - rf.min()) / (rf.max() - rf.min()) - 0.5) * 255 # normalise to [-127.5, 127.5]
    else:
        rf1 = (rf - rf.min()) / (rf.max() - rf.min()) * 255 # normalise to [0, 255]
    # reshape the rf and noise for analysis
    rf1 = rf1.reshape(num_units, -1)
    noise1 = noise.reshape(noise.shape[0], -1)
    
    # calculate the response-weighted covariance
    with tqdm(total = num_units * noise.shape[0]) as pbar:
        for i in range(num_units):
            mu = noise1 - rf1[i]
            for j in range(noise.shape[0]):
                cov[i] += act_conv[i, j] * (mu[j].unsqueeze(1) @ mu[j].unsqueeze(0))
                pbar.update(1)
            cov[i] /= act_conv[i][act_conv[i] != 0].shape[0]
        
    return cov

def CorrRWA(act_conv, noise):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    the correlation-based response-weighted average method - for single channel only.
    '''

    # initialisations
    num_units = act_conv.shape[0]
    rf = torch.zeros((num_units, noise.shape[1], noise.shape[2])) # single channel
    
    # calculate the receptive field using correlation-based response-weighted average
    with tqdm(total = num_units * noise.shape[1] * noise.shape[2]) as pbar:
        for i in range(num_units):
            count = 0 # count the number of non-zero activations
            for j in range(noise.shape[1]):
                for k in range(noise.shape[2]):
                    rf[i, j, k] = np.corrcoef(act_conv[i].flatten(), noise[:, j, k].flatten())[0, 1]
                    # check if the activation is non-zero to record the count
                    count += 1
                    pbar.update(1)
            # normalise by the number of non-zero activations
            if count != 0:
                rf[i] /= count
    
    return rf

def MutualRWA(act_Conv, noise):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    mutual information - for single channel only.
    '''
    
    # intialisations
    num_units = act_conv.shape[0]
    rf = torch.zeros((num_units, noise.shape[1], noise.shape[2])) # single channel
    
    
    return


# Receptive Field Cropping

def CorrLoc(noise, act_conv):
    '''
    This function calculates the Pearson correlation between each noise pixel and the activation
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
    
    # convert correlation to numpy array
    correlation = correlation.numpy()
    
    # create binary mask
    threshold = threshold
    mask = correlation > threshold
    
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
            