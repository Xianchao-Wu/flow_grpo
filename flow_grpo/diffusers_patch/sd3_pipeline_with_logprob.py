# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None, # 512
    width: Optional[int] = None,  # 512
    num_inference_steps: int = 28, # 40
    sigmas: Optional[List[float]] = None, # None
    guidance_scale: float = 7.0, # 4.5
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None, # [16, 205, 4096] 无条件cond + 有条件text，经过三个tokenizers之后，得到的 NOTE
    negative_prompt_embeds: Optional[torch.FloatTensor] = None, # [16, 205, 4096]
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # [16, 2048]
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # [16, 4096]
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    return_prev_sample_mean: bool = False
):
    height = height or self.default_sample_size * self.vae_scale_factor # 512
    width = width or self.default_sample_size * self.vae_scale_factor # 512
    import ipdb; ipdb.set_trace()
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds, # [8, 205, 4096]
        negative_prompt_embeds=negative_prompt_embeds, # [8, 205, 4096]
        pooled_prompt_embeds=pooled_prompt_embeds, # [8, 2048]
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, # [8, 2048]
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs, # ['latents']
        max_sequence_length=max_sequence_length, # 256
    )

    self._guidance_scale = guidance_scale # 4.5
    self._skip_layer_guidance_scale = skip_layer_guidance_scale # 2.8
    self._clip_skip = clip_skip # None
    self._joint_attention_kwargs = joint_attention_kwargs # None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0] # NOTE here; 8

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    ) # None
    (
        prompt_embeds, # [8, 205, 4096]
        negative_prompt_embeds, # [8, 205, 4096]
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance: # True
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) # NOTE 前面是null prompt，后面是真正的prompt --> prompt_embeds.shape=[16, 205, 4096]
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0) # --> [16, 2048]

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels # 16
    import ipdb; ipdb.set_trace()
    latents = self.prepare_latents( # NOTE TODO
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float() # tensor, latents.shape=[8, 16, 64, 64]

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    ) # timesteps=tensor([1000.0000,  960.1293,  913.3490,  857.6923,  790.3683,  707.2785, ...          602.1506,  464.8760,  278.0488,    8.9286], device='cuda:0'); num_inference_steps=10
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []
    all_prev_latents_mean = []

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt: # False
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents # [8, 16, 64, 64] -> [16, 16, 64, 64]
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0]) # 这是一个时间点，扩充到16个，8个cfg，八个condition
            noise_pred = self.transformer(
                hidden_states=latent_model_input, # NOTE
                timestep=timestep, # [16]
                encoder_hidden_states=prompt_embeds, # [16, 205, 4096]
                pooled_projections=pooled_prompt_embeds, # [16, 2048]
                joint_attention_kwargs=self.joint_attention_kwargs, # None
                return_dict=False,
            )[0] # <class 'peft.peft_model.PeftModel'> output noise_pred.shape=[16, 16, 64, 64]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance: # True
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # NOTE, self.guidance_scale=4.5
            latents_dtype = latents.dtype # torch.float32
            import ipdb; ipdb.set_trace()
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, # NOTE  1: FlowMatchEulerDiscreteScheduler, 
                noise_pred.float(), # 2: [16, 16, 64, 64]
                t.unsqueeze(0), # 3: tensor(1000., device='cuda:0') 注意，这是一个具体的时间点！
                latents.float(), # 4: [16, 16, 64, 64]
                noise_level=noise_level, # 5: 0.7
            ) # NOTE TODO sde step
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_prev_latents_mean.append(prev_latents_mean)
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if return_prev_sample_mean:
        return image, all_latents, all_log_probs, all_prev_latents_mean
    return image, all_latents, all_log_probs
