import torch
import os 

def get_device(use_cpu_only,gpu_no):
    
    if use_cpu_only == False:
        device = "cuda:"+str(gpu_no) if torch.cuda.is_available() else "cpu"
        print(f'Total No. of GPUs: {torch.cuda.device_count()}')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Using CPU ONLY")
        device = 'cpu'
    
    print(f'Device : {device}')
    os.environ['device'] = device