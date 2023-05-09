import sys
import os
import time
import torch
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from image_widget import ImageWidget
from common_utils import get_api_from_model
import threading
import qdarkstyle
import json
try:
    import queue
except ImportError:
    import Queue as queue
print(torch.cuda.is_available())
import importlib
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
a=get_api_from_model('model_zoo.YoloV3.alg')
print(a)