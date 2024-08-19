import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs, batch_size):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        self.batch_size = batch_size  # 保存 batch_size

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))
    

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # 使用 VQGAN 的編碼器生成潛在向量
        z, z_indices, q_loss = self.vqgan.encode(x)
        return z, z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear_gamma(ratio):
            return 1 - ratio
        
        def cosine_gamma(ratio):
            return np.cos(np.pi * ratio / 2)
        
        def square_gamma(ratio):
            return 1 - ratio ** 2
        
        if mode == "linear":
            return linear_gamma
        elif mode == "cosine":
            return cosine_gamma
        elif mode == "square":
            return square_gamma
        else:
            raise NotImplementedError(f"Gamma function mode '{mode}' is not implemented.")

##TODO2 step1-3:            
    def forward(self, x):
        # Ground truth: encode the input image to z_indices
        z, z_indices = self.encode_to_z(x)
        
        # 重塑 z_indices 为 (batch_size, num_image_tokens)
        z_indices = z_indices.view(self.batch_size, -1)
        
        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        # 加上隨機生成的 mask
        a_indices = mask * z_indices + (~mask) * masked_indices
        
        # Get logits from the transformer
        logits = self.transformer(a_indices)

        # print(f"logits shape: {logits.shape}")
        # print(f"z_indices shape: {z_indices.shape}")
        
        return logits, z_indices
    
    ##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, masked_z_indices, mask_bc, step, total_iter, mask_num, gamma_type):
        
        masked_z_indices[mask_bc] = 1024
        
        # Step 1: Obtain logits from the transformer
        logits = self.transformer(masked_z_indices)
        # print(logits.size())

        # Step 2: Apply softmax to convert logits into a probability distribution across the last dimension.
        probs = torch.softmax(logits, dim=-1)

        # Step 3: Find the maximum probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)
        z_indices_predict_prob[~mask_bc] = float('inf')
        # Step 4: Calculate the current ratio
        # ratio = self.gamma((step + 1) / total_iter)
        ratio_func = self.gamma_func(mode=gamma_type)
        ratio = ratio_func((step + 1) / total_iter)

        # Step 5: Add temperature annealing gumbel noise as confidence
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))
        # gumbel_noise, _ = torch.max(gumbel_noise, dim=-1)
        # print(f"z_indices_predict_prob shape: {z_indices_predict_prob.shape}")
        # print(f"gumbel_noise shape: {gumbel_noise.shape}")
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * gumbel_noise

        # print(confidence)
        
        # Step 6: Sort the confidence for ranking
        sorted_confidence, sorted_indices = torch.sort(confidence)
        
        # print(sorted_confidence, sorted_indices)

        # Step 7: Define how many tokens to unmask based on the mask scheduling
        # num_to_unmask = int(self.gamma(ratio) * mask_bc.size(-1))
        # print(f"Step {step}: unmasked tokens: {num_to_unmask}")


        # Step 8: Unmask the selected tokens
        # unmask_indices = sorted_indices[:, :num_to_unmask]
        
        # print(unmask_indices.size())
        # print(mask_bc.size())
        # print(f"Step {step}: sorted_indices: {sorted_indices}")
        # print(f"Step {step}: unmask_indices: {unmask_indices}")
        # print(f"Step {step}: mask_bc before scatter_: {mask_bc}")
        # mask_bc.scatter_(1, unmask_indices, False)
        # print(f"Step {step}: mask_bc after scatter_: {mask_bc}")
        
        z_indices_predict[~mask_bc] = masked_z_indices[~mask_bc]
        # print(ratio*mask_num)
        mask_bc[:, sorted_indices[:, math.floor(ratio*mask_num):]] = False

        # Return the updated predictions and mask
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
