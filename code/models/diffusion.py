import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append('./')

from models.unet_new import MyUNet
from tqdm import tqdm
from torch.optim import Adam

device = None
class DiffusionModel(nn.Module):
    def __init__(self, input_shape=(1,28,28), timesteps=200, lbeta=1e-4, ubeta=2e-2):
        super().__init__()
            
        global device
        device = torch.device(os.getenv('device'))
        self.models = [MyUNet() for _ in range(3)]
        self.models = [model.to(device) for model in self.models]
        self.timesteps = timesteps
        self.input_shape = input_shape

        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t, idx):

        # if not isinstance(t, torch.Tensor):
        #     t = torch.LongTensor([t]).expand(x.size(0))
        
        # t_embed = self.time_embed[t]
        return self.models[idx](x,t)

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        
        ## Linear Schedule 
        
        # self.betas = torch.linspace(lbeta,ubeta,self.n_steps)
        # self.alphas = 1-self.betas
        # self.alpha_bars = torch.cumprod(self.alphas,dim=0)
        
        # Cosine Clamp Scheduler 
        
        s = 0.008
        timesteps = (
            torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps + s
        )
        alphas_bars = timesteps / (1 + s) * math.pi / 2
        alphas_bars = torch.cos(alphas_bars).pow(2)
        alphas_bars = alphas_bars / alphas_bars[0]
        betas = 1 - alphas_bars[1:] / alphas_bars[:-1]
        
        betas = betas.clamp(min=lbeta,max=ubeta)
        
        # Cosine Beta Schedule
        
        # def cosine(t):
        #     return math.sin((t/self.n_steps)*math.pi/2 - math.pi/2) + 1 
        
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(cosine(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        # Quadratic Schedule
        
        # def quadratic(t):
        #     return (t/self.n_steps)**2
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(quadratic(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        #Square Root Schedule
        
        # def square_root(t):
        #     return torch.sqrt(t/self.n_steps)
        # t_range = torch.arange(1,self.n_steps+1)
        # betas_unsqueezed = [float(square_root(x)) for x in t_range]
        # betas = [x*((ubeta-lbeta)) + lbeta for x in betas_unsqueezed]
        # betas = torch.Tensor(betas)
        
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas,dim=0)
        

    def q_sample(self, x, t,epsilon):
        """
        Sample from q given x_t.
        """
        
        xt = torch.sqrt(self.alpha_bars[t]).reshape(t.shape[0],1,1,1)*x + torch.sqrt(1-self.alpha_bars[t].reshape(t.shape[0],1,1,1))*epsilon
        return xt

    def p_sample(self, x, t, idx):
        """
        Sample from p given x_t.
        """
        n_samples = x.shape[0]
        z = torch.zeros_like(x)
            
        if t[0]>0:
            for i in range(n_samples):
                z[i] = torch.randn((self.input_shape[0],self.input_shape[1],self.input_shape[2]))
                
        with torch.no_grad():
            x = x.type(torch.float32)
            epsilon_predicted = self.forward(x,t, idx)
            ep_coefficient = self.betas[t]/torch.sqrt(1-self.alpha_bars[t])
            
            x = x - ep_coefficient.reshape(ep_coefficient.shape[0],1,1,1)*epsilon_predicted
            
            x = x/(torch.sqrt(self.alphas[t]).reshape(self.alphas[t].shape[0],1,1,1))
            
            sigma_t = torch.sqrt(self.betas[t])
            
            x = x + sigma_t.reshape(sigma_t.shape[0],1,1,1)*z
            
        return x

    def train_model(self,train_loader,n_iterations,learning_rate,writer,model_path, levels=4):    
        
        optimizers = [Adam(self.models[i].parameters(),learning_rate) for i in range(3)]
        for l in tqdm(range(levels)):
            curr_bound = self.timesteps // 2**l
            self._range = None
            if l == 0:
                self._range = [(0, curr_bound), 
                        (0, 0), 
                        (0, 0)]
            else:
                self._range = [(0, curr_bound), 
                        (curr_bound, self.timesteps - curr_bound), 
                        (self.timesteps - curr_bound, self.timesteps)]
            range_count = [max(0, cb[1] - cb[0]) for cb in self._range]
            iteration_no = 0
            epochs = n_iterations // 2**l
            for epoch in tqdm(range(epochs),leave=False):
                
                for batch in tqdm(train_loader):
                    writer.add_image('input_image', batch[0], iteration_no)
                    batch = batch.to(device)
                    batch_size = batch.shape[0]
                    
                    sampled_idx = torch.multinomial(torch.tensor(range_count, dtype=torch.float32), num_samples=1)
                    # print(sampled_idx,"dskgh")
                    t = torch.distributions.uniform.Uniform(self._range[sampled_idx][0], self._range[sampled_idx][1]).sample(torch.Size((batch_size,))).long()
                    epsilon = torch.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
                
                    for i in range(batch_size):
                        epsilon[i] = torch.randn(self.input_shape)
                        
                    epsilon = epsilon.to(device)
                    epsilon = epsilon.type(torch.float32)
                    t = t.to(device)
                    xt = self.q_sample(batch,t,epsilon)
                    
                    xt = xt.to(device)
                    xt = xt.type(torch.float32)
                    
                    for optim in optimizers:
                        optim.zero_grad()
                    
                    epsilon_predicted = self.forward(xt, t, sampled_idx)
                    
                    loss_fn = torch.nn.MSELoss()
                    loss = loss_fn(epsilon_predicted,epsilon)   
                    loss.backward()
                    optimizers[sampled_idx].step()
                    
                    writer.add_scalar("Loss",loss.item(),iteration_no)
                
                    iteration_no+=1    
                
                if epoch%10 == 0:
                    generated_images = self.sample(10)
                    writer.add_image("Generated_Images",generated_images[0],iteration_no)
                    writer.add_image("Generated_Images_2",generated_images[1],iteration_no)
                    writer.add_image("Generated_Images_3",generated_images[2],iteration_no)
                    writer.add_image("Generated_Images_4",generated_images[3],iteration_no)
                    writer.add_image("Generated_Images_5",generated_images[4],iteration_no)
                    writer.add_image("Generated_Images_6",generated_images[5],iteration_no)
                    writer.add_image("Generated_Images_7",generated_images[6],iteration_no)
                    writer.add_image("Generated_Images_8",generated_images[7],iteration_no)
                    writer.add_image("Generated_Images_9",generated_images[8],iteration_no)
                    writer.add_image("Generated_Images_10",generated_images[9],iteration_no)
                if epoch%100 == 0:
                    torch.save(self.state_dict(),model_path)
            if l == 0:
                for i in range(1,3):
                    self.models[i].load_state_dict(self.models[0].state_dict())
                

    def sample(self, n_samples, progress=False, return_intermediate=False):
        
        intermediate_results = []
        xt = torch.zeros(n_samples,self.input_shape[0],self.input_shape[1],self.input_shape[2])
        for i in range(n_samples):
            xt[i] = torch.randn((self.input_shape[1],self.input_shape[2]))
        
        xt = xt.to(device)
        xt = xt.type(torch.float32)
        
        for t in reversed(range(self.timesteps)):
            t = torch.Tensor([t]).repeat(n_samples).to(device).type(torch.int64)
            idx = None    
            for i, tup in enumerate(self._range):
                if t[0] in range(tup[0], tup[1]):
                    idx = i
                    break
            # print(idx,t[0])
            xt = self.p_sample(xt,t, idx)
            intermediate_results.append(xt)
            
        if return_intermediate:
            return xt,intermediate_results
        else:
            return xt

    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * torch.div(i,2,rounding_mode='floor')) / torch.tensor(d_model, dtype=torch.float32))
        return pos * angle_rates

    def positional_encoding(self,position, d_model):
        # INPUT n_steps and n_dims to the function.
        # OUTPUT 2D tensor you can query to get a time embedding of (index on t-1 as this starts from 0 to n_steps-1)
        angle_rads = self.get_angles(torch.arange(position).unsqueeze(1),
                                torch.arange(d_model).unsqueeze(0),
                                d_model)

        # apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        # Get a n_steps * n_dims tensor for positional encoding
        return angle_rads.type(torch.float32)