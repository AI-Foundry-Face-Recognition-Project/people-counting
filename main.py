#%% import
import os
import sys
import cv2
import time
import torch
import numpy as np
from image_widget import ImageWidget
from common_utils import get_api_from_model
import threading
import qdarkstyle
import json
import importlib
from datetime import datetime

if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
try:
    import queue
except ImportError:
    import Queue as queue

#%% def
def del_all_model_zoo_modules():
    alg_names = []
    for dir in os.listdir('./model_zoo'):
        if os.path.isdir(os.path.join('./model_zoo',dir)):
            alg_names.append(str(dir))
    # del path in sys.path
    del_p = []
    for p in sys.path:
        for alg_name in alg_names:
            if alg_name in p:
                del_p.append(p)
                break
    for p in del_p:
        sys.path.remove(p)
    # del modeules
    old_alg_names = []
    all_keys = sys.modules.keys()
    time.sleep(0.2)
    for alg_name in all_keys:
        if 'from' in str(sys.modules[alg_name]) and \
            'model_zoo' in str(sys.modules[alg_name]) and \
            'YoloAll' in str(sys.modules[alg_name]):
            old_alg_names.append(alg_name)
        if 'namespace' in str(sys.modules[alg_name]) and hasattr(sys.modules[alg_name], '__path__'):
            module_path = str(sys.modules[alg_name].__path__)
            if 'model_zoo' in module_path and \
               'YoloAll' in module_path:
               old_alg_names.append(alg_name)       
    for alg_name in old_alg_names:
        del sys.modules[alg_name] 

def add_one_model_path(alg_name):
    sub_dir = os.path.join(alg_name)
    sys.path.append(sub_dir)

def get_api_from_model(alg_name):

    api = None
    del_all_model_zoo_modules()
    add_one_model_path(alg_name)
    
    try:
        api = importlib.import_module(alg_name)
        print('create api from', alg_name, 'success')
    except ImportError as e:
        print('create api from', alg_name, 'failed')
        print('error:', str(e))   
        api = None
        
    return api

def avg(a,b):
    return (a+b)/2

#%% init
#source video
cap=cv2.VideoCapture('B1_Cam1_1.mp4')
#detected log
f=open('location','w')
#save video
outname='out.mp4'
#save video type
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#running device
device = "cuda" if torch.cuda.is_available() else "cpu"
#model select
model_select = 'yolov5_s'
#model zoo path
model_zoo_path = 'YoloV5'
#api init
alg_name = 'model_zoo.' + model_zoo_path + '.alg'
api = importlib.import_module(alg_name)
alg = api.Alg()
alg.load_cfg()
alg.create_model(model_select, device)
#api init end
#%% class
class Detection():
    def __init__(self):
        self.lst=np.array([])
        self.cmd=''
        self.result=np.array([])
        self.null_max=15
        self.null_count=0
        self.i=0
        self.is_start=False
    def in_conv(self,cmd,x1,y1,x2,y2):
        self.cmd=cmd
        self.is_start=True
        tmp=np.array([[y1,y2]])
        if len(self.lst)==0:
            self.lst=tmp
        else:
            np_tmp=self.lst
            self.lst=np.append(np_tmp,tmp,axis=0)
        if x1==0 and y1==0 and x2==0 and y2==0:
            self.null_count+=1
            if self.null_count>=self.null_max:
                self.cmd='end'
        else:
            self.null_count=0
        if self.cmd=='end':
            for i in range(self.null_count):
                self.lst=np.delete(self.lst,-1,axis=0)
            self.derived()
        return self.cmd
    def clear_deviation(self,data ,Sigma=1):
        mean = np.mean(data)
        std = np.std(data)
        std_min=mean - Sigma * std
        std_max=mean + Sigma * std
        for i,j in enumerate(data):
            if not std_min<j<std_max:
                if i==0:
                    data[i]=data[i+1]
                elif i==len(data)-1:
                    data[i]=data[i-1]
                else:
                    data[i]=avg(data[i-1],data[i+1])
        return data
    def derived(self,kernel_size=3):
        global f
        currentDateAndTime = datetime.now()
        now_time=currentDateAndTime.strftime("%H:%M:%S")
        print('###start###')
        print(now_time)
        f.write('###start###\n'+now_time+'\n')
        print(self.lst)
        f.write(str(self.lst)+'\n')
        data=self.lst.transpose()
        data[0]=self.clear_deviation(data[0])
        data[1]=self.clear_deviation(data[1])
        print('---')
        print(data)
        f.write('---\n'+str(data)+'\n')
        clean_data=np.diff(avg(data[0],data[1]))
        print('---')
        print(clean_data)
        f.write('---\n'+str(clean_data)+'\n###end###\n')
        print('###end###')
        clean_data/=100

        self.result=sum(clean_data)
        #print('###')
        #print(clean_data)
        #kernel=np.array([.30,.33,.37])
        #kernel_2=np.array([.49,.51])
        #while len(clean_data)>=kernel_size:
        #    clean_data=np.convolve(clean_data, kernel, mode='valid')
        #print(sum(clean_data))
        #print('###')
        ##while len(clean_data)>1:
        ##    clean_data=np.convolve(clean_data, kernel_2, mode='valid')
        #self.result=clean_data
        print(self.result)
        f.write(str(self.result)+'\n')
#%%
ready , frame =cap.read()
frame=frame[0:1080,400:1520,:]
width,  height=frame.shape[1], frame.shape[0]
out = cv2.VideoWriter(outname, fourcc, 15.0, (width,  height))
detect = Detection()
act='init'
while (ready):
    ret=alg.inference(frame)
    frames=ret['result']
    valid_pred=alg.valid_pred
    boxes = valid_pred[:,0:4]
    boxes = boxes.numpy()
    scores = valid_pred[:, 4]
    scores = scores.numpy()
    guess=valid_pred[:,5]
    dels=[]
    for i,j in enumerate(guess): 
        if j!=0:
                dels.append(i)
    dels.reverse()
    for i in dels:
        boxes=np.delete(boxes,i,axis=0)
        scores=np.delete(scores,i,axis=0)
    ###########
    if len(boxes)==0:
        if detect.is_start:
            act=detect.in_conv('add',0,0,0,0)
    else:
        x1,y1,x2,y2=boxes[0]
        act=detect.in_conv('add',x1,y1,x2,y2)
    if act=='end':
        print('end')
        f.write('end\n')
        del detect
        act='init'
        detect = Detection()
    ###########
    #f.write(str(boxes)+','+str(scores)+'\n')
    cv2.imshow('frame',frames)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ready , frame =cap.read()
    if ready:
        frame=frame[0:1080,400:1520,:]
f.close()
#detect.in_conv('end',0,0,0,0)
out.release()
cv2.destroyAllWindows()

#%% test
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
ret=alg.inference(img)
valid_pred=alg.valid_pred
boxes = valid_pred[:,0:4]
from common_utils import vis
img_array=img
cls = valid_pred[:, 5]
scores = valid_pred[:, 4]
vis(img_array, boxes, scores, cls, conf=0.0)
map_result = {'type':'img'}
map_result['result'] = img_array
plt.imshow(ret['result'])


# %%
