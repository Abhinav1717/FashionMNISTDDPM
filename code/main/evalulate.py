import torch
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
import os 
from torch.utils.tensorboard import SummaryWriter
import sys
from scipy.spatial.distance import cdist
from chamferdist import ChamferDistance
from pyemd import emd
from tqdm import tqdm
import numpy as np

sys.path.append('../')

from utils.utils import get_device
from dataset.datasets import FashionMNISTDataset
from transforms.transforms import toTensor, Resize, Normalize
import models.diffusion_vanilla as vanilla_diffusion
import models.diffusion as ensemble_diffusion
device = None

def get_emd(d1, d2):
    d_comb = np.concatenate((d1, d2), axis=0)
    dist = np.linalg.norm((d_comb), axis=1).reshape((-1,1))
    d1 = np.concatenate((np.zeros((d1.shape[0], 1)), d1), axis=1)
    d2 = np.concatenate((np.ones((d2.shape[0], 1)), d2), axis=1)
    d_comb = np.concatenate((d1, d2), axis=0)
    app = np.concatenate((dist, d_comb), axis=1)
    app = app[app[:, 0].argsort()]
    d1_sig, d2_sig = 1 - app[:, 1], app[:, 1]
    dist_sorted = app[:, 2:]
    dist = cdist(dist_sorted, dist_sorted)
    d1_sig = d1_sig.copy(order='C')
    d2_sig = d2_sig.copy(order='C')
    dist = dist.copy(order='C')
    return emd(d1_sig, d2_sig, dist)

def main():
    
    # For Reproducibility
    torch.manual_seed(24)
    
    # Parsing the arguements
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size",default=100,help="Size of Batch")
    parser.add_argument("--learning_rate",default=0.001,help="Learning Rate to train the model")
    parser.add_argument("--n_iterations",default=200,help="No of Epochs")
    parser.add_argument("--images_directory_path",default="../../datasets/FashionMNIST/train",help="Relative Path to the images directory")
    parser.add_argument("--gpu_no",default="1",help="GPU Id of the gpu you want to use for training")
    parser.add_argument("--use_cpu_only",default=False,help="Toggle if you want to use cpu for training",type=bool)
    parser.add_argument("--log_dir",default="../../evaluate_logs/",help="Log Directory Path")
    parser.add_argument("--vanilla_model_dir",default="../../models_vanilla_300/",help="Vanilla Model Directory Path")
    parser.add_argument("--ensemble_model_dir",default="../../models_ensemble_300/",help="Ensemble Model Directory Path")
    parser.add_argument("--levels", default=4, type=int, help='Represents the number of levels')
    parser.add_argument("--n_samples", default=1000, type=int, help='Represents the number of Samples to be generated during inference')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    images_directory_path = args.images_directory_path
    gpu_no = args.gpu_no
    use_cpu_only = args.use_cpu_only
    log_directory_path = args.log_dir
    vanilla_model_directory_path = args.vanilla_model_dir
    ensemble_model_directory_path = args.ensemble_model_dir
    levels = args.levels
    n_samples = args.n_samples
        
    # Checking for CUDA SUPPORT
    get_device(use_cpu_only,gpu_no)
    
    global device
    device = torch.device(os.getenv('device'))
    
    transform = Compose(
        [
            toTensor(), 
            # Resize((128,128)),
            Normalize()
        ]
    )
    
    
    #Loading the Dataset
    
    dataset = FashionMNISTDataset(images_directory_path,transform)

    train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    
    #Initializing the Model   
    image_shape = (1,28,28)
    vanilla_fashion_generator = vanilla_diffusion.DiffusionModel(image_shape,1000,1)
    vanilla_model_path = os.path.join(vanilla_model_directory_path,"fashion_generator.pt")
    vanilla_fashion_generator = vanilla_fashion_generator.to(device)
    vanilla_fashion_generator.load_state_dict(torch.load(vanilla_model_path,map_location=device))
    
    ensemble_fashion_generator = ensemble_diffusion.DiffusionModel(image_shape,1000,levels)
    ensemble_model_path = os.path.join(ensemble_model_directory_path,"fashion_generator.pt")
    ensemble_fashion_generator = ensemble_fashion_generator.to(device)
    ensemble_fashion_generator.load_state_dict(torch.load(ensemble_model_path,map_location=device))
    

    
    vanilla_writer = SummaryWriter(log_dir = os.path.join(log_directory_path,"Fashion_MNIST_Diffusion/vanilla"))
    ensemble_writer = SummaryWriter(log_dir = os.path.join(log_directory_path,"Fashion_MNIST_Diffusion/ensemble"))
    
    vanilla_generated_images = vanilla_fashion_generator.inference(batch_size,n_samples,1,vanilla_writer)
    ensemble_generated_images = ensemble_fashion_generator.inference(batch_size,n_samples,levels,ensemble_writer)
    
    cd = ChamferDistance()
    vanilla_emd = 0.0
    ensemble_emd = 0.0
    
    vanilla_chamfer = 0.0
    ensemble_chamfer = 0.0
    for vanilla_images,ensemble_images in tqdm(zip(vanilla_generated_images,ensemble_generated_images)):
        original_images = next(iter(train_loader))
        original_images = original_images.to(device)
        vanilla_images = vanilla_images.reshape((vanilla_images.shape[0],-1))
        original_images = original_images.reshape((original_images.shape[0],-1))
        ensemble_images = ensemble_images.reshape((ensemble_images.shape[0],-1))
        
        vanilla_emd+=get_emd(original_images.cpu().numpy(),vanilla_images.cpu().numpy())
        ensemble_emd+=get_emd(original_images.cpu().numpy(),ensemble_images.cpu().numpy())
        
        vanilla_chamfer+=cd(original_images.unsqueeze(0).float(),vanilla_images.unsqueeze(0).float()).item()
        ensemble_chamfer+=cd(original_images.unsqueeze(0).float(),ensemble_images.unsqueeze(0).float()).item()
        
    avg_vanilla_emd = vanilla_emd/len(vanilla_generated_images)
    avg_ensemble_emd = ensemble_emd/len(ensemble_generated_images)
    
    avg_vanilla_chamfer = vanilla_chamfer/len(vanilla_generated_images)
    avg_ensemble_chamfer = ensemble_chamfer/len(ensemble_generated_images)
    
    print(f'Average EMD Score for Vanilla DDPM {avg_vanilla_emd}')
    print(f'Average EMD Score for Ensemble DDPM {avg_ensemble_emd}')
    
    print(f'Average Chamfer Distance for Vanilla DDPM {avg_vanilla_chamfer}')
    print(f'Average Chamfer Distance for Ensemble DDPM {avg_ensemble_chamfer}')
    
    vanilla_writer.flush()
    vanilla_writer.close()

    ensemble_writer.flush()
    ensemble_writer.close()
    
if __name__ == "__main__":
    main()

