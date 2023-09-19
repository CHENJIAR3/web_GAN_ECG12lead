from ecgpoint_detector import signalDelineation
import  os
import  numpy as np
from config import  myconfig
import  tensorflow as tf
from metric_chen import *
from utils_datat import *
import matplotlib.pyplot as plt
import neurokit2 as nk
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #指定程序在显卡0、1、2中运行
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
from loaddata import read_tfrecords,signal_filter,decode_tfrecords_label,read_tfrecords_label,toonehot,ecgsplit_R2

def ecg_re2(R, ecg_split,a,lenG=1024):
    ecg_r = []
    length = R.shape[0]
    for i in range(length):
        if i != (length-1):
            if ((R[i+1]-R[i])%2==1):
                ecg_r.append(ecg_split[i, 512:512 + int((R[i+1] - R[i])/2), :])
                ecg_r.append(ecg_split[i + 1, 512 - int((R[i+1] - R[i])/2)-1:512,:])
            else:
                ecg_r.append(ecg_split[i, 512:512 +int((R[i + 1] - R[i])/2), :])
                ecg_r.append(ecg_split[i + 1, 512 - int((R[i + 1] - R[i])/2):512, :])
    ecg_c = tf.concat((ecg_split[0,512-R[0]:512], ecg_r[0]), axis=0)
    for i in range(1,len(ecg_r)):
        ecg_c = tf.concat((ecg_c, ecg_r[i]), axis=0)
    ecg_c = tf.concat((ecg_c,ecg_split[length-1,512:512+a-R[length-1],:]),axis=0)
    return ecg_c

def ecgsplit_R3(ecg, ecglen=1024):
    ecg2 = []
    processed_ecg = nk.ecg_process(ecg[:], 500)
    R = processed_ecg[1]['ECG_R_Peaks']
    length = R.shape[0]
    if R[0] > ecglen/2:
        R0_index=np.argmax(ecg[0:512])
        a = tf.zeros(512 - R0_index)
        a = tf.concat((a, ecg[0:R0_index + 512]), axis=0)
        ecg2.append(a)
        ecg2.append(ecg[R[0] - 512:R[0] + 512])
        R0_index = np.asarray([R0_index])
        R_new = np.concatenate((R0_index,R))
    else:
        a = tf.zeros(512 - R[0])
        a = tf.concat((a, ecg[0:R[0] + 512]), axis=0)
        ecg2.append(a)
        R_new = R
    for i in range(1, length):
        if R[i] >= ecglen / 2 and (ecg.shape[0] - R[i]) > ecglen / 2:
            if i == (length - 1):
                ecg2.append(ecg[R[i] - 512:R[i] + 512])
                R_end=R[i]+np.argmax(ecg[R[i]:len(ecg[:,0])])
                a = tf.zeros(len(ecg[:,0]) - R_end)
                a = tf.concat((ecg[R_end-512:len(ecg[:,0])],a), axis=0)
                ecg2.append(a)
                R_end = np.asarray([R_end])
                R_new = np.concatenate((R_new,R_end))
            else:
                ecg2.append(ecg[R[i] - 512:R[i] + 512])
        elif R[i] < ecglen/2:
            a = tf.zeros(512 - R[i])
            a = tf.concat((a, ecg[0:R[i] + 512]), axis=0)
            ecg2.append(a)
        elif R[i] > ecglen / 2 and (ecg.shape[0] - R[i]) < ecglen / 2:
            b = tf.zeros(R[i] + 512 - ecg.shape[0])
            b = tf.concat((ecg[R[i] - 512:ecg.shape[0]], b), axis=0)
            ecg2.append(b)
    return np.asarray(ecg2),R_new,R

def ecg_12lead_r(ecg):
    ecgx,R,R0=ecgsplit_R3(ecg,1024)
    ecgm = generator(ecgx)
    a = len(ecg)
    ecgm_12lead = ecg_re2(R,ecgm,a)
    return ecgm_12lead


if __name__ == '__main__':
    ecglen=1024
    test = read_tfrecords('CPSC_test').batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    generator = tf.keras.models.load_model('GeneratorMeg_resnet_CPSC_train40.9', compile=False)
    for step, inputs in enumerate(test):
        if step ==1:
            ecg,label,=inputs
            break
    ecgx = ecg[0,0,:]
    ecgm = ecg_12lead_r(ecgx)