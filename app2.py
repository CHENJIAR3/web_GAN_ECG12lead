# coding=utf-8
# copyright：CJ3
# 2023.04.04

import json
import  tensorflow as tf
from tensorflow.keras.models import model_from_json,load_model,Sequential
from tensorflow.keras.layers import Input,Conv1D

import tornado.ioloop as ioloop
import tornado.web as web

import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import time
# TensorFlow and tf.keras


import scipy.signal
import scipy.interpolate as si
# import scipy.io
import scipy.io as sio
import neurokit2 as nk
global result


# 心电信号分类模型
# # 模型参数
# Classifier1=tf.saved_model.load(r"./models/Classifier_1")
# Classifier1=load_model(r"./models/classifier_1")
Classifier12=load_model(r"./models/classifier_12")
Generator = load_model(r"./models/Generator")
# Classifier1.summary()
port1=5001

ecglen=10240
dataa=[]
ecgfs=500
# 账户名和密码
nameall=["wwqUser","cjr",'UZZH123']
pwdall=["wwqPWD",'cjr123','PZZH123']

classname = ['Normal','AF','I-AVB','LBBB', 'RBBB',
                                 'PAC','PVC','STD','STE']

lead = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# 是否登录进去
loginflag=0

def ecgtosamelen(ecg,ecglen):
    ecg=np.asarray(ecg)
    if ecg.shape[0]<ecglen:
        ecg=tf.concat([ecg,tf.zeros((ecglen-ecg.shape[0],ecg.shape[1]))],axis=0)
    ecg2=ecg[:ecglen,:]
    return ecg2
# 模型预测
def model_predict(x, model):
    #心电信号预处理
    x=tf.convert_to_tensor(x)
    x=tf.expand_dims(x,axis=0)
    # 模型预测
    preds = model.predict(x)
    # 按照顺序，第一个为lead1 最后一个为max
    predsmax=preds[-1]
    predsmax=predsmax/sum(predsmax[0])
    return preds[0],predsmax

def ecgsplit_R(ecg, ecglen=1024):
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
    return np.asarray(ecg2),R_new

def ecg_re(R, ecg_split,lena,lenG=1024):
    ecg_r = []
    length = R.shape[0]
    for i in range(length):
        if i != (length-1):
            ecg_r.append(ecg_split[i, int(lenG/2):int(lenG/2) + int((R[i + 1] - R[i])/ 2), :])
            ecg_r.append(ecg_split[i + 1,int(lenG/2)-int((R[i+1] - R[i])/2)-int((R[i+1]-R[i])%2):int(lenG/2),:])
            # if ((R[i+1]-R[i])%2==1):
            # else:
                # ecg_r.append(ecg_split[i, lenG/2:lenG/2 +int((R[i + 1] - R[i])/2), :])
                # ecg_r.append(ecg_split[i + 1, lenG/2- int((R[i + 1] - R[i])/2):lenG/2, :])
    ecg_c = tf.concat((ecg_split[0,int(lenG/2)-R[0]:int(lenG/2)], ecg_r[0]), axis=0)
    for i in range(1,len(ecg_r)):
        ecg_c = tf.concat((ecg_c, ecg_r[i]), axis=0)
    ecg_c = tf.concat((ecg_c,ecg_split[length-1,int(lenG/2):int(lenG/2)+lena-R[length-1],:]),axis=0)
    return ecg_c

def generator_predict(x,model,lenG=1024):
    # x=tf.convert_to_tensor(x)
    x=np.squeeze(x)
    ecgx,R_new = ecgsplit_R(x, lenG)
    ecgx=tf.convert_to_tensor(ecgx)
    ecgm=model.predict(ecgx)
    ecg11=ecg_re(R_new,ecgm,len(x))
    ecg12=tf.concat([tf.cast(tf.expand_dims(x,axis=1),dtype=float),tf.cast(tf.squeeze(ecg11),dtype=float)],axis=1)
    # ecg12=ecgtosamelen(ecg12,ecglen)
    return ecg12

def getdata_chen(body):
    a=[]
    radix = 0
    sym = 1
    num = 0
    i=0
    dataflag=0
    body+=b','
    passflag=1
    for j in body:
        if chr(j) ==',' or chr(j)==' ' or chr(j)=='\t' or chr(j)=='\r' or chr(j)=='\n':
            if dataflag==1:
                num *= (2 * sym - 1)
                a.append(num)
                i = i + 1
                dataflag = 0
                num = 0
                radix = 0
                sym = 1
            continue
        if chr(j) == '.':
            radix = 1
            continue
        if chr(j) == '-':
            sym = 0
            continue
        if radix:
                dataflag=1
                num += (j - 48) / pow(10, radix)
                radix = radix + 1
        else:
                dataflag=1
                num = 10 * num + (j - 48)
    return a,i

def genImage(y,fs=500):
    t=np.arange(np.array(y).shape[0])
    t=t/fs
    plt.figure(figsize=(20,10))
    y=np.squeeze(y)
    if y.ndim==1:
        plt.plot(t, y,'b',linewidth=2)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude [mv]')
        dpinum=150
    else:
        # plt.subplots(3,4,figsize=(2.5,2))
        for i in range(y.shape[1]):
            plt.subplot(3, 4, i + 1)
            plt.plot(t[2*fs:4*fs], y[2*fs:4*fs, i], 'b', linewidth=2)
            plt.xticks([])
            plt.yticks([])
            # 永远的神，子图不冲突
            plt.tight_layout()
            plt.title('Generated lead ' + lead[i])
            if i==0:
                plt.title('Real lead ' + lead[i])
            plt.grid(True)

            dpinum=1200
    memdata = io.BytesIO()
    plt.grid(True)
    plt.savefig(memdata, format='png',dpi=dpinum)
    image = memdata.getvalue()
    return image


class UserloginHandler(web.RequestHandler):
    def get(self,*args,**kwargs):
        self.render("login.html")
    def post(self,*args,**kwargs):
        global loginflag
        if self.get_argument("login-num") in nameall and self.get_argument("login-pwd") in pwdall:
            if nameall.index(self.get_argument("login-num"))==pwdall.index(self.get_argument("login-pwd")):
                # self.render("upload.html")
                self.redirect(r"/upload")
                loginflag=1
        if loginflag==0:
            self.render("backhome.html")



class UploadHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        if loginflag:
            output = {}
            for i in range(len(classname)):
                output[classname[i]] = ''
            self.render("upload.html",heartbeat_class=output,heartbeat_class2=output)
        else:
            self.render("login.html")

    def post(self,*args,**kwargs):
        # 读取文件信息
        if self.request.files:
            files = self.request.files['file']
            # 这里面支持txt格式
            if files[0]['content_type']=='text/plain':
                global loginflag
                loginflag=1
                # print(files)
                for file in files:
                    body = file['body']
                # 读取其中的信息
                a, count = getdata_chen(body)
                # 获取采样频率的值
                fs=self.get_argument("fs",500)
                # 判断采样频率是否为数字
                if fs.isdigit():
                    fs=int(fs)
                    global gfs
                    gfs=fs
                    a=np.asarray(a)
                    global dataa

                    # 如果信号为1*n，先进行转置
                    if a.shape[0] == 1:
                        a = np.transpose(a)
                    if a.ndim==1:
                        a = np.expand_dims(a,axis=1)

                    # 信号上采样成360点
                    if fs!=ecgfs:
                        cub=si.interp1d(np.linspace(0,1,a.shape[0]),a,kind='cubic')
                        a=cub(np.linspace(0,1,int(ecgfs/fs)*a.shape[0]))
                    # 数据预处理
                    a=a/max(max(a),abs(min(a)))
                    dataa = a

                    # a=ecgtosamelen(a,ecglen)

                    global ecg12
                    ecg12 = generator_predict(dataa,Generator)
                    print(ecg12.shape)
                    ecg12=ecgtosamelen(ecg12,ecglen)
                    preds,preds12 = np.array(tf.squeeze(model_predict(ecg12,Classifier12))).round(3)
                    # result = classname[np.argmax(preds)]
                    output={}
                    output12={}
                    for i in range(len(preds)):
                        output[classname[i]]=preds[i]
                        output12[classname[i]]=preds12[i]
                    self.render("upload.html", heartbeat_class=output,heartbeat_class2=output12)
                else:
                    response_letter="Error: Non-standard sample frequency"
                    self.render("back.html", response_letter=response_letter)
            else:
                response_letter = "Error: NO Txt"
                self.render("back.html",response_letter =response_letter)
        else:
            response_letter="Error: No file"
            self.render("back.html",response_letter =response_letter)
class img12leadHandler(web.RequestHandler):
    global ecg12
    def get(self, *args, **kwargs):
        if len(dataa) < 1:
            image = genImage([])
        else:

            image = genImage(ecg12[:dataa.shape[0]], gfs)
        self.write(image)
application = web.Application([
    (r"/",UserloginHandler),
    (r"/upload", UploadHandler),
    (r"/img12lead",img12leadHandler),
],
static_path = os.path.join(os.getcwd(), "static"),
debug = True
)

if __name__ == "__main__":
    print('Starting server')
    print('Done. Check http://192.168.1.111:'+str(port1)+'/')
    # nohup挂后台
    application.listen(port1)
    # 持续监听端口
    ioloop.IOLoop.instance().start()

