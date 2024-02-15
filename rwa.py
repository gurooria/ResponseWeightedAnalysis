# this file contains all the functions for response-weighted analysis
import torch
from tqdm import tqdm
import numpy as np


def ActRecorder(layer, net, NBatches=100, BSize=1000, inputX=64, inputY=64, zeroMean=False):
    '''
    This function records the activations of the units in a specified layer of the network
    using an ensemble of noise patterns generated from a uniform distribution.
    '''    
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    hx, cx = torch.zeros((BSize, 128), dtype=torch.float32), torch.zeros((BSize, 128), dtype=torch.float32)
    noise = [] # input noise patterns
    act_conv = [[] for i in range(NUnits)] # responses of conv units to be recorded

    # get the centre locations of activation map
    X = (torch.rand(BSize, 3, inputX, inputY) - 0.5) * 255 # colored noise pattern ensemble
    x1,x2,(_,_) = net(0, X, (hx,cx), 1) # forward pass to get the activation map
    if layer == 'conv1':
        nR,nC = x1[0,0,:,:].shape
    elif layer == 'conv2':
        nR,nC = x2[0,0,:,:].shape
    rloc, cloc = int(round(float(nR)/2.0)), int(round(float(nC)/2.0))
    
    # forward pass to record activations
    net.eval()
    with tqdm(total=NBatches) as pbar:
        for i in range(NBatches):
            # set noise pattern distribution
            if zeroMean:
                X = (torch.rand(BSize, 3, inputX, inputY) - 0.5) * 255 # zero mean noise, [-127.5, 127.5]
            else:
                X = torch.rand(BSize, 3, inputX, inputY) * 255 # non-zero mean noise, [0, 255]
            # forward pass
            with torch.no_grad():
                x1,x2,(_,_) = net(0, X, (hx,cx), 1)
            # record the activations for each unit using the same noise ensemble
            for j in range(NUnits):
                if layer == 'conv1':
                    act_conv[j].append(x1[:, j, rloc, cloc])
                elif layer == 'conv2':
                    act_conv[j].append(x2[:, j, rloc, cloc])
            noise.append(X)
            pbar.update(1)
            
    # reshape the activation and noise for analysis and visualisation
    with tqdm(total=NUnits) as pbar:
        for i in range(NUnits):
            act_conv[i] = torch.cat(act_conv[i])
            pbar.update(1)
    act_conv = torch.stack(act_conv)
    noise = torch.cat(noise)
    noise = np.transpose(noise, (0, 2, 3, 1)) # transform to visualisation format in rgb state
    
    print(f"Shape of activation response list: {act_conv.shape}")
    print(f"Shape of noise list: {noise.shape}")
    
    return act_conv, noise

def ActRecorder_mnist(layer, net, NBatches=100, BSize=1000, inputX=28, inputY=28, zeroMean=False):
    '''
    This function records the activations of the units in a specified layer of the network
    using an ensemble of noise patterns generated from a uniform distribution.
    '''    
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    noise = [] # input noise patterns
    act_conv = [[] for i in range(NUnits)] # responses of conv units to be recorded
    
    # get the centre locations of activation map
    X = (torch.rand(BSize, 1, inputX, inputY) - 0.5) * 255 # colored noise pattern ensemble
    x1,x2 = net(X) # forward pass to get the activation map
    if layer == 'conv1':
        nR,nC = x1[0,0,:,:].shape
    elif layer == 'conv2':
        nR,nC = x2[0,0,:,:].shape
    rloc, cloc = int(round(float(nR)/2.0)), int(round(float(nC)/2.0))

    # forward pass to record activations
    net.eval()
    with tqdm(total=NBatches) as pbar:
        for i in range(NBatches):
            # set noise pattern distribution
            if zeroMean:
                X = (torch.rand(BSize, 1, inputX, inputY) - 0.5) * 255 # zero mean noise, [-127.5, 127.5]
            else:
                X = torch.rand(BSize, 1, inputX, inputY) * 255 # non-zero mean noise, [0, 255]
            # forward pass
            with torch.no_grad():
                x1,x2 = net(X)
            # record the activations for each unit using the same noise ensemble
            for j in range(NUnits):
                if layer == 'conv1':
                    act_conv[j].append(x1[:, j, rloc, cloc])
                elif layer == 'conv2':
                    act_conv[j].append(x2[:, j, rloc, cloc])
            noise.append(X)
            pbar.update(1)
            
    # reshape the activation and noise for analysis and visualisation
    with tqdm(total=NUnits) as pbar:
        for i in range(NUnits):
            act_conv[i] = torch.cat(act_conv[i])
            pbar.update(1)
    act_conv = torch.stack(act_conv)
    noise = torch.cat(noise)
    noise = np.transpose(noise, (0, 2, 3, 1)) # transform to visualisation format in rgb state
    
    print(f"Shape of activation response list: {act_conv.shape}")
    print(f"Shape of noise list: {noise.shape}")

    return act_conv, noise


def RWA(layer, net, act_conv, noise, NBatches=100, BSize=1000, inputX=64, inputY=64, absolute=False):
    '''
    This function estimates the receptive field of each unit in the specified layer using
    the response-weighted analysis method (modified version of the spike-triggered average method).
    '''
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    rf = torch.zeros((NUnits, inputX, inputY, 3)) # receptive field
    
    # response-weighted average
    with tqdm(total=NUnits*NBatches*BSize) as pbar:
        for i in range(NUnits):
            for j in range(NBatches*BSize):
                if absolute: # use absolute value of the activation
                    rf[i] += abs(act_conv[i, j]) * noise[j]
                else:
                    rf[i] += act_conv[i, j] * noise[j]
                pbar.update(1)
            rf[i] /= (act_conv[i] != 0).sum() # normalise by the number of non-zero activations
    
    print(f"Shape of receptive field list: {rf.shape}")
    
    return rf


def RWC(layer, net, act_conv, noise, rf, NBatches=100, BSize=1000, inputX=64, inputY=64, zeroMean=False):
    '''
    This function performs response-weighted covariance analysis to estimate the autocorrelation matrix
    of the receptive field of each unit in the specified layer.
    '''
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    cov = torch.zeros((NUnits, inputX*inputY, inputX*inputY, 3)) # covariance matrix
    
    # reshape rf and noise for RWC operations
    rf1 = rf.reshape(rf.shape[0], -1, 3) # reshaped into (NUnits, inputX*inputY, 3)
    for i in range(NUnits): # normalise rf1 to to be the same scale as noise
        if rf1[i].any() != 0:
            if zeroMean:
                rf1[i] = ((rf1[i] - rf1[i].min()) / (rf1[i].max() - rf1[i].min()) - 0.5) * 255 # [-127.5, 127.5]
            else:
                rf1[i] = (rf1[i] - rf1[i].min()) / (rf1[i].max() - rf1[i].min()) * 255.0 # [0, 255]
    noise1 = noise.reshape(noise.shape[0], -1, 3) # reshaped into (NBatches*BSize, inputX*inputY, 3)

    print(f"Shape of reshaped receptive field list: {rf1.shape}")
    print(f"Shape of reshaped noise list: {noise1.shape}")
    
    # response-weighted covariance            
    with tqdm(total=NBatches*NUnits*BSize) as pbar:
        for i in range(NUnits):
            mu = rf1[i]
            for j in range(NBatches*BSize):
                tmp = noise1[j] - mu
                cov[i, :, :, 0] += act_conv[i, j] * (tmp.unsqueeze(1)[:, :, 0] @ tmp.unsqueeze(0)[:, :, 0])
                cov[i, :, :, 1] += act_conv[i, j] * (tmp.unsqueeze(1)[:, :, 1] @ tmp.unsqueeze(0)[:, :, 1])
                cov[i, :, :, 2] += act_conv[i, j] * (tmp.unsqueeze(1)[:, :, 2] @ tmp.unsqueeze(0)[:, :, 2])
                pbar.update(1)
            cov[i] /= (act_conv[i] != 0).sum() # divide cov[i] by the number of non-zero values in act_conv[i]

    print(f"Shape of covariance matrix list: {cov.shape}")
    
    return cov

def RWC_mnist(layer, net, act_conv, noise, rf, NBatches=100, BSize=1000, inputX=28, inputY=28, zeroMean=False):
    '''
    This function performs response-weighted covariance analysis to estimate the autocorrelation matrix
    of the receptive field of each unit in the specified layer.
    '''
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    cov = torch.zeros((NUnits, inputX*inputY, inputX*inputY, 1)) # covariance matrix
    
    # reshape rf and noise for RWC operations
    rf1 = rf.reshape(rf.shape[0], -1, 1) # reshaped into (NUnits, inputX*inputY, 1)
    for i in range(NUnits): # normalise rf1 to to be the same scale as noise
        if rf1[i].any() != 0:
            if zeroMean:
                rf1[i] = ((rf1[i] - rf1[i].min()) / (rf1[i].max() - rf1[i].min()) - 0.5) * 255 # [-127.5, 127.5]
            else:
                rf1[i] = (rf1[i] - rf1[i].min()) / (rf1[i].max() - rf1[i].min()) * 255.0 # [0, 255]
    noise1 = noise.reshape(noise.shape[0], -1, 1) # reshaped into (NBatches*BSize, inputX*inputY, 3)

    print(f"Shape of reshaped receptive field list: {rf1.shape}")
    print(f"Shape of reshaped noise list: {noise1.shape}")
    
    # response-weighted covariance            
    with tqdm(total=NBatches*NUnits*BSize) as pbar:
        for i in range(NUnits):
            mu = rf1[i]
            for j in range(NBatches*BSize):
                tmp = noise1[j] - mu
                cov[i,:,:,0] += act_conv[i, j] * (tmp @ tmp.T)
                pbar.update(1)
            cov[i] /= (act_conv[i] != 0).sum() # divide cov[i] by the number of non-zero values in act_conv[i]

    print(f"Shape of covariance matrix list: {cov.shape}")
    
    return cov


def eigenAnalysis(cov, unit):
    '''
    This function performs eigenanalysis on the covariance matrix of the receptive field.
    '''
    eigen = []
    for i in range(3):
        eigen.append(torch.linalg.eigh(cov[unit, :, :, i]))
    return eigen


def PWA(layer, net, act_conv, noise, window, NBatches=100, BSize=1000, inputX=64, inputY=64):
    '''
    This function performs phase-weighted average analysis to visualise a specific window of responses of each unit.
    '''
    # initialisations
    NUnits = net.get_submodule(layer).out_channels
    rf = torch.zeros((NUnits, inputX, inputY, 3)) # receptive field
    
    # response-weighted average
    with tqdm(total=NUnits*NBatches*BSize) as pbar:
        for i in range(NUnits):
            count = 0
            for j in range(NBatches*BSize):
                if (act_conv[i, j] >= window[0]) and (act_conv[i, j] <= window[1]):
                    rf[i] += abs(act_conv[i, j]) * noise[j]
                    count += 1
                pbar.update(1)
            rf[i] /= count # normalise by the number of non-zero activations
    
    print(f"Shape of receptive field list: {rf.shape}")
    
    return rf