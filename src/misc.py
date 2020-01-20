# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.signal import firwin, lfilter
import csv
from sprocket.util import HDF5, extfrm, static_delta


def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Parameters
    ---------
    x : array, shape(`samples`)
        Waveform sequence
    fs: array, int
        Sampling frequency
    cutoff : float, optional
        Cutoff frequency of low cut filter
        Default set to 70 [Hz]

    Returns
    ---------
    lcf_x : array, shape(`samples`)
        Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def read_feats(listf, h5dir, ext='mcep'):
    """HDF5 handler
    Create list consisting of arrays listed in the list

    Parameters
    ---------
    listf : str,
        Path of list file
    h5dir : str,
        Path of hdf5 directory
    ext : str,
        `mcep` : mel-cepstrum
        `f0` : F0

    Returns
    ---------
    datalist : list of arrays

    """

    datalist = []
    with open(listf, 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join(h5dir, f + '.h5')
            h5 = HDF5(h5f, mode='r')
            datalist.append(h5.read(ext))
            h5.close()
    return datalist

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def read_ppg_feats(s_listf,t_listf,h5dir,ext='mcep'):
    s_datalist = []
    t_datalist = []
    with open(s_listf,'r') as fp:
        for line in fp:
            s_mcep = []
            t_mcep = []
            f = line.rstrip()
            h5f = os.path.join(h5dir, f + '.h5')
            h5 = HDF5(h5f, mode='r')
            s_mcep.append(h5.read(ext))
            h5.close()
            source = "BDL_1.27"
            target = "RRBI_16k"
            line = line.replace(source,target)
            f = line.rstrip()
            h5f = os.path.join(h5dir, f + '.h5')
            h5 = HDF5(h5f, mode='r')    
            t_mcep.append(h5.read(ext))
            h5.close()
            source_post = "/home/anurag/kaldi/egs/librispeech/s5/post_source/"
            target_post = "/home/anurag/kaldi/egs/librispeech/s5/post_target/"
            f = "post." + f[-12:] + ".ark"
            print(f)
            s_post = os.path.join(source_post, f)
            t_post = os.path.join(target_post, f)
            s_post = np.loadtxt(s_post)
            t_post = np.loadtxt(t_post)
            for i in range(len(s_post)):
                score = []
                for j in range(len(t_post)):
                    score.append(KL(s_post[i],t_post[j]))
                t_mcep.append(t_post[score.index(min(score))])    
            s_datalist.append(s_mcep)
            t_datalist.append(t_mcep)
            print(len(s_datalist))
            print(len(t_datalist))
        return s_mcep, t_mcep


def extsddata(data, npow, power_threshold=-20):
    """Get power extract static and delta feature vector

    Paramters
    ---------
    data : array, shape (`T`, `dim`)
        Acoustic feature vector
    npow : array, shape (`T`)
        Normalized power vector
    power_threshold : float, optional,
        Power threshold
        Default set to -20

    Returns
    -------
    extsddata : array, shape (`T_new` `dim * 2`)
        Silence remove static and delta feature vector

    """
    if len(data) != len(npow):
        npow = npow[:len(data)]
    extsddata = extfrm(static_delta(data), npow,
                       power_threshold=power_threshold)
    return extsddata

def exts_post_data(data,npow,power_threshold=-20):
    if len(data) != len(npow):
        npow = npow[:len(data)]
    extsddata = extfrm(data, npow,
                       power_threshold=power_threshold)
    return extsddata    

def exts_post_FA(data,FA,delta=False):
	phone = []
	time = []
	with open(FA,'r') as f:
		for i in csv.reader(f,dialect='excel',delimiter='\t'):
			phone.append(i[1])
			time.append(i[0])

	sil_index = [i for i,x in enumerate(phone) if x=='sil']
	reject_frames = []
	for i in sil_index:
	    [start_time, end_time] = time[i].split(' ')
	    for frame in range(int(float(start_time) * 100) + 1, int(float(end_time) * 100) + 1):
	        reject_frames.append(frame)

	total_frames = list(set(range(0,int(float(time[-1].split(' ')[1]) * 100))) - set(reject_frames))			
	if delta == False:
		return(np.array([data[i] for i in total_frames]))
	else:
		new_data = []
		new_data = np.array([data[i] for i in total_frames])
		return(static_delta(new_data))
def transform_jnt(array_list):
    num_files = len(array_list)
    for i in range(num_files):
        if i == 0:
            jnt = array_list[i]
        else:
            jnt = np.r_[jnt, array_list[i]]
    return jnt
