# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler, # 1: scheduler
    model_output: torch.FloatTensor, # 2: noise_pred, [16, 16, 64, 64], t时刻给定输入sample/latents的时候的self.transformer的预测出来的volecity，不是具体的Noisy image!!! NOTE 
    timestep: Union[float, torch.FloatTensor], # 3: one time point! not a list
    sample: torch.FloatTensor, # 4: latents, [16, 16, 64, 64], t时刻的self.transformer的输入；
    noise_level: float = 0.7, # 5: 0.7
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output (velocity field tensor) from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    #import ipdb; ipdb.set_trace()
    model_output=model_output.float() # 来自现在timestep时刻的flow matching model的预测结果，noisy image
    sample=sample.float() # NOTE TODO ? 输入, sample + t --> self.transformer --> model_output/velocity
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep] # tensor([1000.], device='cuda:0') -> t=1000 -> step_index=[0]
    prev_step_index = [step+1 for step in step_index] # [1]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1))) # 关于噪声强度的, from self=scheduler, scheduler.sigmas = tensor([1.0000, 0.9913, 0.9824, 0.9731, 0.9634, 0.9534, 0.9430, 0.9323, 0.9211, ...         0.2062, 0.1465, 0.0810, 0.0089, 0.0000], device='cuda:0') sigma.shape=[1, 1, 1, 1] and sigma=tensor([[[[1.]]]], device='cuda:0'), noise schedule!
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1))) # tensor([[[[0.9913]]]], device='cuda:0')
    sigma_max = self.sigmas[1].item() # TODO 这个是固定的吗？0.9913
    dt = sigma_prev - sigma # NOTE < 0, tensor([[[[-0.0087]]]], device='cuda:0'), sigma_{t-1} - sigma_t
    #import ipdb; ipdb.set_trace()
    if sde_type == 'sde': # NOTE here
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level # https://arxiv.org/pdf/2505.05470 flow-grpo paper, page 5, 公式(9)的正下方 

        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt # NOTE 就是flow-grpo page 5的公式(9)的没有epsilon项的一个变形！！！
        # 当前样本 + velocity 输出 → 推算前一个状态的平均值； NOTE 

        if prev_sample is None: # None, in
            variance_noise = randn_tensor(
                model_output.shape, # torch.Size([16, 16, 64, 64])
                generator=generator, # None
                device=model_output.device, # device(type='cuda', index=0)
                dtype=model_output.dtype, # torch.float32
            ) # variance_noise.shape=[8, 16, 64, 64] = epsilon
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise # NOTE 
            # x_{t+dt}  = x_t +               g(x_t, t) *     sqrt(dt)     * epsilon 

        # NOTE Line 69's sigma = std_dev_t * sqrt(-dt) ? 是的，use Eq (9) second part, coeff of epsilon = sigma_t * sqrt(-dt) = standard derivation of the new prob!
        #import ipdb; ipdb.set_trace()
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        ) # log_prob.shape=[8, 16, 64, 64]
        # p(x_{t+dt} | x_t) = N(x_t + f*dt, g^2*dt)
        # logp = -(x-mu)^2/(2*sigma^2) - log(sigma) - 1/2*log(2*pi), here, mu=prev_sample_mean, sigma=std_dev_t * sqrt(-dt) NOTE
        # NOTE std_dev_t=0, so log_prob=nan, TODO shall we do something???
    elif sde_type == 'cps': # consistency policy sampling (cps)
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
    #import ipdb; ipdb.set_trace()
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim))) # torch.Size([16]), batch_size=16, tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], ... since std_dev_t=0 ... 
    
    if return_sqrt_dt: # False
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
    # prev_sample.shape=[16, 16, 64, 64] 这个就出来了, x_t-1时刻的noisy audio tensor；
    # log_prob.shape=[16] all nan, not-a-number
    # prev_sample_mean.shape=[16, 16, 64, 64], the mean of 'prev_sample'，x_t-1的分布的mean
    # std_dev_t=tensor([[[[0.]]]], device='cuda:0'), shape=[1,1,1,1]
