import numpy as np
import torch
from torch import nn, optim
from environments import ExecuteBase
from . import GuidanceBase, Fuzz


class CureFuzz(Fuzz):
    def __init__(self, args, SeedSpace, Execute, algo:str = None, model_path:str = None):
        super().__init__(args, SeedSpace, Execute, algo, model_path)
        self.guidance = Guidance(self.executer)
        for seed, execute_data in self.init_seeds:
            samplePr, add = self.guidance.analyze(execute_data)
            self.fuzzer.further_mutation((seed, execute_data.current_reward, execute_data.last_final_state), samplePr)
        del self.init_seeds

class Guidance(GuidanceBase):
    ALPHA = 5
    BETA = 0.01
    INTRINSIC_THRESHOLD = 10
    ENTROPY_TRHESHOLD = 10
    class RND(nn.Module):
        def __init__(self, input_size=24, hidden_size=256, output_size=256):
            super().__init__()
            self.target_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
            self.predictor_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

            # Initialize the target network with random weights
            for m in self.target_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1)
                    nn.init.constant_(m.bias, 0)

            # Make target network's parameters not trainable
            for param in self.target_net.parameters():
                param.requires_grad = False
                
        def forward(self, x):
            target_out = self.target_net(x)
            predictor_out = self.predictor_net(x)
            return target_out, predictor_out
        
    def __init__(self, executer: ExecuteBase):
        self.reward_low_threshold = executer.reward_low_threshold
        self.reward_high_threshold = executer.reward_high_threshold

        self.rnd = None
        self.device = torch.device('cpu')
        self.learning_rate=1e-4
        self.criterion = nn.MSELoss()
    
    def train_rnd(self, states, l2_reg_coeff=1e-4):
        if self.rnd is None:
            self.dim = np.array(states[0]).reshape(-1).shape[0]
            if self.dim > 512:
                self.rnd = self.RND(np.array(states[0]).reshape(-1).shape[0], hidden_size=1, output_size=1).to(self.device)
            else:
                self.rnd = self.RND(np.array(states[0]).reshape(-1).shape[0], hidden_size=64, output_size=64).to(self.device)
            self.optimizer = optim.Adam(list(self.rnd.predictor_net.parameters()), lr=self.learning_rate)
        number = len(states)
        state_tensor = torch.FloatTensor(np.array(states)).view(number, -1).to(self.device)
        target_out, predictor_out = self.rnd(state_tensor)
         # Compute the mean squared error between the predictor and target network outputs
        mse_loss = self.criterion(predictor_out, target_out)
        intrinsic_reward = mse_loss.item()

        # Add L2 regularization to the loss
        l2_reg = 0
        for param in self.rnd.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss = mse_loss + l2_reg_coeff * l2_reg

        self.optimizer.zero_grad()
        loss.backward()
        np.clip
        self.optimizer.step()
    
        return intrinsic_reward
    
    def analyze(self, data):
        intrinsic_reward = self.train_rnd(data.sequence)
        entropy = np.linalg.norm(np.asarray(data.sequence[-2]) - np.asarray(data.last_final_state))
        normalized_reward = 2 * (data.current_reward - self.reward_low_threshold) / (self.reward_high_threshold - self.reward_low_threshold) - 1
        if intrinsic_reward*self.BETA > 100:
            samplePr = np.exp(100)+np.exp(-normalized_reward*self.ALPHA)+entropy
        else:
            samplePr = np.exp(intrinsic_reward*self.BETA)+np.exp(-normalized_reward*self.ALPHA)+entropy
        add = bool(
            intrinsic_reward > self.INTRINSIC_THRESHOLD or 
            data.current_reward < data.last_reward or 
            entropy > self.ENTROPY_TRHESHOLD
        )
        return samplePr, add