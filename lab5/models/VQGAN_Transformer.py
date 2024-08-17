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
            return ratio
        
        def cosine_gamma(ratio):
            return 0.5 * (1 + np.cos(np.pi * ratio))
        
        def square_gamma(ratio):
            return ratio ** 2
        
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

        # Get logits from the transformer
        logits = self.transformer(z_indices)

        # print(f"logits shape: {logits.shape}")
        # print(f"z_indices shape: {z_indices.shape}")
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, x, num_steps=12):
        # Step 1: Encode input image to latent space
        z, z_indices = self.encode_to_z(x)
        
        # Initialize mask: start with all tokens masked
        mask = torch.ones_like(z_indices, dtype=torch.bool)
        mask_bc = mask.clone()
        
        for step in range(num_steps):
            # Step 2: Masked tokens are replaced by a special mask token
            masked_z_indices = z_indices.clone()
            masked_z_indices[mask] = self.mask_token_id
            
            # Step 3: Get logits from transformer
            logits = self.transformer(masked_z_indices)
            
            # Apply softmax to convert logits into probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Step 4: Find the maximum probability and the corresponding token predictions
            z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)
            
            # Calculate confidence with Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs)))
            confidence = z_indices_predict_prob + self.choice_temperature * gumbel_noise
            
            # Sort tokens by confidence
            _, sorted_indices = torch.sort(confidence, dim=-1, descending=True)
            
            # Update mask based on the gamma function and sorted confidence
            num_to_unmask = int(self.gamma(step / num_steps) * mask.size(-1))
            unmask_indices = sorted_indices[:, :num_to_unmask]
            
            # Unmask selected tokens
            mask.scatter_(1, unmask_indices, False)
            mask_bc = torch.where(mask_bc, mask, mask_bc)
        
        # Step 5: At the end of decoding, add back the original unmasked token values
        z_indices_final = torch.where(mask_bc, z_indices_predict, z_indices)
        
        return z_indices_final, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
