import numpy as np
from operator import mul
from functools import reduce
import time
import tqdm
import torch
from torch import nn
from frameworks import Framework
from environments import SeedBase


class G_Model(Framework):
    def __init__(self, args, SeedSpace, Execute, algo = None, model_path = None):
        super().__init__(args, SeedSpace, Execute, algo, model_path)
        self.normal_space = SeedBase(low=np.zeros(self.seed_space.dim), high=np.ones(self.seed_space.dim), shape=self.seed_space.shape, dim=self.seed_space.dim)
        normal_cases = self.normal_space.random_generate(number=1000)
        self.diffusion_model = Diffusion(data_size=self.seed_space.dim)
        self.diffusion_model.setup()
        self.diffusion_model.train(normal_cases, epochs=100)

    def test(self):
        start_time = time.time()
        while not self.terminate(time.time()-start_time):
            normal_case_list = []
            for _ in range(50):
                normal_case = self.diffusion_model.generate()
                test_case = normal_case.reshape(self.seed_space.shape) * (self.seed_space.high - self.seed_space.low) + self.seed_space.low
                episode_reward, failure, info = self.executer(test_case)
                if 'success' in info and not info['success']:
                    continue
                self.all_test_cases.append(test_case)
                normal_case_list.append(normal_case)
                if failure:
                    self.result.append(test_case)
                    self.failure_time.append((time.time()-start_time)/3600)
                if self.pbar is not None:
                    self.pbar.update(1)
                    self.pbar.set_postfix({'Found': len(self.result)})
            for _ in range(100):
                normal_case_list.append(self.normal_space.random_generate())
            self.diffusion_model.train(normal_case_list)
        self.efficiency = len(self.all_test_cases) / (time.time()-start_time)
        if self.pbar is not None:
            self.pbar.close()


def position_encoding_init(n_position, d_pos_vec):
    ''' 
    Init the sinusoid position encoding table 
    n_position in num_timesteps and d_pos_vec is the embedding dimension
    '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).to(torch.float32)

class Denoising(torch.nn.Module):

    def __init__(self, x_dim, num_diffusion_timesteps):
        super(Denoising, self).__init__()

        self.linear1 = torch.nn.Linear(x_dim, 256)
        self.emb = position_encoding_init(num_diffusion_timesteps,x_dim)
        self.linear2 = torch.nn.Linear(256, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, x_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x_input, t):
        emb_t = self.emb[t]
        x = self.linear1(x_input+emb_t)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x


class Diffusion:
    def __init__(self, 
    batch_size = 1, 
    epoch = 1, 
    data_size = 128, 
    training_step_per_epoch = 50, 
    num_diffusion_step = 50
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_size = data_size
        self.training_step_per_epoch = training_step_per_epoch
        self.num_diffusion_step = num_diffusion_step
        self.setup()

    def setup(self):
        self.beta_start = .0004
        self.beta_end = .02

        self.device = torch.device("cpu")

        self.betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_diffusion_step) ** 2
        self.alphas = 1 - self.betas

        # send parameters to device
        self.betas = torch.tensor(self.betas).to(torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas).to(torch.float32).to(self.device)

        # alpha_bar_t is the product of all alpha_ts from 0 to t
        self.list_bar_alphas = [self.alphas[0]]
        for t in range(1,self.num_diffusion_step):
            self.list_bar_alphas.append(reduce(mul,self.alphas[:t]))
            
        self.list_bar_alphas = torch.cumprod(self.alphas, axis=0).to(torch.float32).to(self.device)

        self.criterion = nn.MSELoss()
        self.denoising_model = Denoising(self.data_size, self.num_diffusion_step).to(self.device)
        # disgusting hack to put embedding layer on 'device' as well, as it is not a pytorch module!
        self.denoising_model.emb = self.denoising_model.emb.to(self.device)
        self.optimizer = torch.optim.AdamW(self.denoising_model.parameters(), lr=3e-4)

    def train(self, training_data, epochs=None):
        if epochs is None:
            epochs = self.epoch
        indices = list(range(len(training_data)))
        for epoch in range(epochs):  # loop over the dataset multiple times
            # sample a bunch of timesteps
            Ts = np.random.randint(1,self.num_diffusion_step, size=self.training_step_per_epoch)
            for _, t in enumerate(Ts):
                # produce corrupted sample
                index = np.random.choice(indices)
                x_init = training_data[index]
                x_init = torch.from_numpy(x_init).to(torch.float32).to(self.device)
                q_t = self.q_sample(x_init, t, self.list_bar_alphas, self.device)
                        
                # calculate the mean and variance of the posterior forward distribution q(x_t-1 | x_t,x_0)
                mu_t, cov_t = self.posterior_q(x_init, q_t, t, self.alphas, self.list_bar_alphas, self.device)
                # get just first element from diagonal of covariance since they are all equal
                sigma_t = cov_t[0][0]
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                mu_theta = self.denoising_model(q_t , t)
                loss = self.criterion(mu_t, mu_theta)
                loss.backward()

                self.optimizer.step()
                
            # self.pbar.set_description('Epoch: {} Loss: {}'.format(epoch, running_loss/self.training_step_per_epoch))
            # print('running_loss:\t',running_loss)


    def q_sample(self, x_start, t, list_bar_alphas, device):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        alpha_bar_t = list_bar_alphas[t]
        
        mean = alpha_bar_t*x_start
        cov = torch.eye(x_start.shape[0]).to(device)
        cov = cov*(1-alpha_bar_t)
        return torch.distributions.MultivariateNormal(loc=mean,covariance_matrix=cov).sample().to(device)


    def denoise_with_mu(self, denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
        """
        Denoising function considering the denoising models tries to model the posterior mean
        """
        alpha_t = list_alpha[t]
        beta_t = 1 - alpha_t
        alpha_bar_t = list_alpha_bar[t]
        
        mu_theta = denoise_model(x_t,t)
        
        x_t_before = torch.distributions.MultivariateNormal(loc=mu_theta,covariance_matrix=torch.diag(beta_t.repeat(self.data_size))).sample().to(device)
            
        return x_t_before


    def posterior_q(self, x_start, x_t, t, list_alpha, list_alpha_bar, device):
        """
        calculate the parameters of the posterior distribution of q
        """
        beta_t = 1 - list_alpha[t]
        alpha_t = list_alpha[t]
        alpha_bar_t = list_alpha_bar[t]
        # alpha_bar_{t-1}
        alpha_bar_t_before = list_alpha_bar[t-1]
        
        # calculate mu_tilde
        first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)
        second_term = x_t * torch.sqrt(alpha_t)*(1- alpha_bar_t_before)/ (1 - alpha_bar_t)
        mu_tilde = first_term + second_term
        
        # beta_t_tilde
        beta_t_tilde = beta_t*(1 - alpha_bar_t_before)/(1 - alpha_bar_t)
        
        cov = torch.eye(x_start.shape[0]).to(device)*(1-alpha_bar_t)
        
        return mu_tilde, cov


    def generate(self):
        data = torch.distributions.MultivariateNormal(loc=torch.zeros(self.data_size),covariance_matrix=torch.eye(self.data_size)).sample().to(self.device)
        for t in range(self.num_diffusion_step):
            data = self.denoise_with_mu(self.denoising_model,data,self.num_diffusion_step-t-1, self.alphas, self.list_bar_alphas, self.data_size, self.device)

        case = data.cpu().numpy()
        case = np.clip(case,0,1)
        return case