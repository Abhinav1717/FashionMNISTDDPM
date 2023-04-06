import torch
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
import os 
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('../')

from utils.utils import get_device
from dataset.datasets import FashionMNISTDataset
from transforms.transforms import toTensor, Resize,Normalize
from models.diffusion import DiffusionModel
device = None

def main():
    
    # For Reproducibility
    torch.manual_seed(24)
    
    # Parsing the arguements
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size",default=128,help="Size of Batch")
    parser.add_argument("--learning_rate",default=0.001,help="Learning Rate to train the model")
    parser.add_argument("--n_iterations",default=100,help="No of Epochs")
    parser.add_argument("--images_directory_path",default="../../datasets/FashionMNIST/train",help="Relative Path to the images directory")
    parser.add_argument("--gpu_no",default="1",help="GPU Id of the gpu you want to use for training")
    parser.add_argument("--use_cpu_only",default=False,help="Toggle if you want to use cpu for training",type=bool)
    parser.add_argument("--log_dir",default="../../logs/",help="Log Directory Path")
    parser.add_argument("--model_dir",default="../../models/",help="Model Directory Path")
    parser.add_argument("--levels", default=4, type=int, help='Represents the number of levels')
    args = parser.parse_args()
    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_iterations = args.n_iterations
    images_directory_path = args.images_directory_path
    gpu_no = args.gpu_no
    use_cpu_only = args.use_cpu_only
    log_directory_path = args.log_dir
    model_directory_path = args.model_dir
    levels = args.levels
    
    os.makedirs(model_directory_path,exist_ok=True)
    
    # Checking for CUDA SUPPORT
    get_device(use_cpu_only,gpu_no)
    
    global device
    device = torch.device(os.getenv('device'))
    
    transform = Compose(
        [
            toTensor(), 
            # Resize((128,128)),
            # Normalize()
        ]
    )
    
    
    #Loading the Dataset
    
    dataset = FashionMNISTDataset(images_directory_path,transform)

    train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=4)
   
    images = next(iter(train_loader))
    print(images.shape)
    print(torch.max(images))
    print(torch.min(images))
    #Initializing the Model     
    fashion_generator = DiffusionModel(images.shape[1:],1000, levels)
    fashion_generator = fashion_generator.to(device)
    t = torch.tensor([1], dtype=torch.int64)
    model_path = os.path.join(model_directory_path,"fashion_generator.pt")
    
    writer = SummaryWriter(log_dir = os.path.join(log_directory_path,"Fashion_MNIST_Diffusion"))
    # writer.add_graph(fashion_generator,(images.to(device),t))
    
    fashion_generator.train_model(train_loader,n_iterations,learning_rate,writer,model_path, levels)
    
    torch.save(fashion_generator.state_dict(),model_path)
    
    writer.flush()
    writer.close()

    
if __name__ == "__main__":
    main()

