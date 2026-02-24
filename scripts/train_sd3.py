from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

#import ipdb; ipdb.set_trace()
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device): # max_sequence_length=128, device='cuda'
    #import ipdb; ipdb.set_trace()
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        ) # [1, 205, 4096] and [1, 2048]
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators
# sample.keys() = dict_keys(['prompt_embeds', 'pooled_prompt_embeds', 'timesteps', 'latents', 'next_latents', 'log_probs', 'advantages']); j=0; embeds.shape=[16, 205, 4096]; pooled_embeds.shape=[16, 2048]; config: <class 'ml_collections.config_dict.config_dict.ConfigDict'> 完整的配置文件信息
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    #import ipdb; ipdb.set_trace()
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2), # noisy input
            timestep=torch.cat([sample["timesteps"][:, j]] * 2), # timestep
            encoder_hidden_states=embeds, # textual prompt's embedding
            pooled_projections=pooled_embeds, # textual prompt's pooled embedding
            return_dict=False,
        )[0] # noise_pred = predicted velocity field
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        ) # velocity field mixture of between conditional and un-conditional NOTE
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    #import ipdb; ipdb.set_trace() 
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(), # predicted velocity field
        sample["timesteps"][:, j], # time
        sample["latents"][:, j].float(), # noisy audio input of time t
        prev_sample=sample["next_latents"][:, j].float(), # predicted noisy audio 'output' of time t-1
        noise_level=config.sample.noise_level, # 0.7
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    import ipdb; ipdb.set_trace()
    if config.train.ema: # True
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1) # [1, 205, 4096] to [16, 205, 4096] and config.sample.test_batch_size=16
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1) # [1, 2048] to [16, 2048]

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader, # len(test_dataloader)=64
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        #import ipdb; ipdb.set_trace()
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, # a list with 16 textual sequences, real world text in english
            text_encoders, # a list of 3 text encoders
            tokenizers, # a list of 3 tokenizers
            max_sequence_length=128, 
            device=accelerator.device # 'cuda'
        ) # prompt_embeds.shape=[16, 205, 4096], pooled_prompt_embeds.shape=[16, 2048]
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds): # not in!
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                import ipdb; ipdb.set_trace()
                images, _, _ = pipeline_with_logprob( # NOTE TODO
                    pipeline,
                    prompt_embeds=prompt_embeds, # [16, 205, 4096]
                    pooled_prompt_embeds=pooled_prompt_embeds, # [16, 2048]
                    negative_prompt_embeds=sample_neg_prompt_embeds, # ['']=neg prompt, pure empty -> [16, 205, 4096]
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds, # ['']=neg prompt, pure empty -> [16, 2048]
                    num_inference_steps=config.sample.eval_num_steps, # 40
                    guidance_scale=config.sample.guidance_scale, # 4.5
                    output_type="pt",
                    height=config.resolution, # 512
                    width=config.resolution, # 512
                    noise_level=0,
                ) # 这个文生图的方法，是40步迭代。output images.shape=[16, 3, 512=height, 512=width]
        #import ipdb; ipdb.set_trace()
        # TODO 学习代码的时候，是直接用下面的，可以看清楚reward_fn， ocr，的逻辑细节。' ' -> ''，然后用edit distance来计算图片中识别出来的ocr的文字，和prompt中的“”双引号里面的文本内容进行的对比
        #rewards = reward_fn(images, prompts, prompt_metadata, only_strict=False)

        #import ipdb; ipdb.set_trace()
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False) # images.shape=[16, 3, 512, 512] after ode and sde! final denoised images with a batch size=16
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
        break # NOTE TODO for debug only and we only look at one batch
    import ipdb; ipdb.set_trace() 
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy() # [16, 3, 512, 512]
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device) # tensor([[49406,   320,  1400,  ..., 49407, 49407, 49407], where '<|endoftext|>'=49407, and '<|startoftext|>'=49406; last_batch_prompt_ids.shape=[16, 256] since max_length=256
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy() # [16, 256]
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    ) # 'a high - fashion runway with a sleek, modern backdrop displaying " spring collection 2 0 2 4 ". models walk confidently on the catwalk, showcasing vibrant, floral prints and pastel tones, under soft, ambient lighting that enhances the fresh, spring vibe.' ...
    last_batch_rewards_gather = {}
    for key, value in rewards.items(): # {'ocr': [0.44999999999999996, 0.9375, 0.9166666666666666, 0.6, 1.0, 0.15000000000000002, 0.0, 0.16666666666666663, 0.9411764705882353, 0.9411764705882353, 1.0, 0.9545454545454546, 1.0, 0.4285714285714286, 0.8666666666666667, 0.9583333333333334], 'avg': [0.44999999999999996, 0.9375, 0.9166666666666666, 0.6, 1.0, 0.15000000000000002, 0.0, 0.16666666666666663, 0.9411764705882353, 0.9411764705882353, 1.0, 0.9545454545454546, 1.0, 0.4285714285714286, 0.8666666666666667, 0.9583333333333334]}
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process: # True
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather)) # TODO why min(15, 16) = num_samples=15 ????
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index] # [16, 3, 512, 512]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                ) # [3, 512, 512]
                pil = pil.resize((config.resolution, config.resolution)) # resize((512, 512))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg")) # '/tmp/tmpkm0ygw3h', idx=0
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices] # a list with 15 elements, one element is alike: {'ocr': 0.45, 'avg': 0.45}
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema: # True
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    #import ipdb; ipdb.set_trace()
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    ) # ProjectConfiguration(project_dir='logs/2026.02.20_01.17.47', logging_dir='logs/2026.02.20_01.17.47', automatic_checkpoint_naming=True, total_limit=5, iteration=0, save_on_each_node=False)

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    import ipdb; ipdb.set_trace()
    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        #config.pretrained.model # 'stabilityai/stable-diffusion-3.5-medium' # TODO NOTE
        '/workspace/asr/flow_grpo/ckpts/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80',
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False) # 83,819,683 = 83.8M parameters
    pipeline.text_encoder.requires_grad_(False) # 123,650,304 = 123.6M parameters
    pipeline.text_encoder_2.requires_grad_(False) # 694,659,840 = 694.6M parameters
    pipeline.text_encoder_3.requires_grad_(False) # 4,762,310,656 = 4.7B parameters, which is big
    pipeline.transformer.requires_grad_(not config.use_lora) # 2,243,171,520 = 2.2B parameters, which is alike middle size! NOTE 当使用lora的时候，transformer的参数不需要梯度；如果不使用lora，则transformer的参数是需要梯度的，也就是需要更新transformer中的参数来做grpo！所以说，现在的grpo，就是为了训练下面的这些lora中的low ranker adapter weight matrices的这些参数！ 后续需要知道他们的大小是多少
    import ipdb; ipdb.set_trace()
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32) # 'cuda:7' okay good interesting, 
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora: # True NOTE TODO this is peft usage of lora = low-ranking adapter!
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        ) # LoraConfig(task_type=None, peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, peft_version='0.18.1', base_model_name_or_path=None, revision=None, inference_mode=False, r=32, target_modules={'attn.to_v', 'attn.add_k_proj', 'attn.to_q', 'attn.to_k', 'attn.add_q_proj', 'attn.to_add_out', 'attn.to_out.0', 'attn.add_v_proj'}, exclude_modules=None, lora_alpha=64, lora_dropout=0.0, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights='gaussian', layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, alora_invocation_tokens=None, use_qalora=False, qalora_group_size=16, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, target_parameters=None, arrow_config=None, ensure_weight_tying=False) NOTE
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config) # NOTE okay, add lora adapters to current pipeline.transformer architecture!  18,776,064=18M parameters that are trainable, which is 0.0083 = 0.83% less than 1% of parameters of current pipeline.transformer model! NOTE TODO
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) # 382 elements, all with a size of 49152, so the number of trainable parameters is 382 * 49152 = 18,776,064 = 18.7M
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32: # True, TF = tensor float!
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam: # False
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW # here NOTE

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate, # TODO fixed? 0.0003
        betas=(config.train.adam_beta1, config.train.adam_beta2), # (0.9, 0.999)
        weight_decay=config.train.adam_weight_decay, # 0.0001
        eps=config.train.adam_epsilon, # 1e-08
    )
    import ipdb; ipdb.set_trace()
    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn) # ocr: 1.0 <function multi_score.<locals>._fn at 0x7f96d7445f30>
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn) # ocr: 1.0

    if config.prompt_fn == "general_ocr": # True
        train_dataset = TextPromptDataset(config.dataset, 'train') # config.dataset='/workspace/asr/flow_grpo/dataset/ocr'; <__main__.TextPromptDataset object at 0x7f9ba5127520>
        test_dataset = TextPromptDataset(config.dataset, 'test') # <__main__.TextPromptDataset object at 0x7f9ba515a7a0>

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size, # 8
            k=config.sample.num_image_per_prompt, # 8
            num_replicas=accelerator.num_processes, # 1
            rank=accelerator.process_index, # 0
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=1, #8, TODO for debug only, set num_workers=1
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=1, #8, TODO for debug only, set num_workers=1  
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    import ipdb; ipdb.set_trace()
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device) # NOTE neg_prompt_embed.shape=[1, 205, 4096], neg_pooled_prompt_embed.shape=[1, 2048]

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1) # [8, 205, 4096]
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1) # [8, 205, 4096]
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1) # [8, 2048]
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1) # [8, 2048]

    if config.sample.num_image_per_prompt == 1: # 8, not 1
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking: # True
        stat_tracker = PerPromptStatTracker(config.sample.global_std) # TODO what for and why state tracker?

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast # config.use_lora = True; autocast = <class 'contextlib.nullcontext'>
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size # 8
        * accelerator.num_processes # 1
        * config.sample.num_batches_per_epoch # 8
    ) # 64, too small... TODO why only 64 batches in one epoch???
    total_train_batch_size = (
        config.train.batch_size # 8
        * accelerator.num_processes # 1
        * config.train.gradient_accumulation_steps # 4, TODO need to change this, maybe 1 for debugging is better?
    ) # 32 = total_train_batch_size

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        import ipdb; ipdb.set_trace()
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0: # 0 % 60 == 0 yes
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config) # config.save_dir='logs/ocr/sd3.5-M'
        import ipdb; ipdb.set_trace()
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch), # 8
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter) # prompts=a list of 8 textual sequences; prompt_metadata=[{}, {}, {}, {}, {}, {}, {}, {}]
            import ipdb; ipdb.set_trace()
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, # 3 encoders
                tokenizers, # 3 tokenizers
                max_sequence_length=128, 
                device=accelerator.device
            ) # prompt_embeds.shape=[8, 205, 4096], pooled_prompt_embeds.shape=[8, 2048] 这是用了三个tokenizers
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device) # [8, 256] 这是只用了最初一个tokenizer

            # sample
            if config.sample.same_latent: # False
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    import ipdb; ipdb.set_trace()
                    images, latents, log_probs = pipeline_with_logprob( # NOTE TODO
                        pipeline,
                        prompt_embeds=prompt_embeds, # [8, 205, 4096]
                        pooled_prompt_embeds=pooled_prompt_embeds, # [8, 2048]
                        negative_prompt_embeds=sample_neg_prompt_embeds, # [8, 205, 4096]
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds, # [8, 2048]
                        num_inference_steps=config.sample.num_steps, # 10
                        guidance_scale=config.sample.guidance_scale, # 4.5
                        output_type="pt",
                        height=config.resolution, # 512
                        width=config.resolution, 
                        noise_level=config.sample.noise_level, # 0.7
                        generator=generator # None
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96), e.g., [8, 11, 16, 64, 64]
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps), [8, 10]

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps), e.g., [8, 10]

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids, # tensor([[49406,   320, 23187,  ..., 49407, 49407, 49407], [8, 256]
                    "prompt_embeds": prompt_embeds, # [8, 205, 4096]
                    "pooled_prompt_embeds": pooled_prompt_embeds, # [8, 2048]
                    "timesteps": timesteps, # [8, 10]
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )
        import ipdb; ipdb.set_trace()
        # wait for all rewards to be computed
        for sample in tqdm(
            samples, # 8 batches in total, each batch is with 8 sequences. NOTE
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result() # NOTE 这里就有取值了; {'ocr': [0.050000000000000044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30000000000000004], 'avg': [0.050000000000000044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30000000000000004]}; reward_metadata={}
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            } # 普通的list，转成了tensor

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        } # NOTE 绝了, samples['prompt_ids'].shape=[64, 256]

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i] # [3, 512, 512]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引; tmpdir='/tmp/tmph100l8rw' NOTE 这是把生成的图片保存到硬盘了

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"] # shape=[64], 'avg' is same with 'ocr'
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps) # num_train_timesteps = 9, so samples['rewards']['avg'].shape=[64, 9]
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking: # True
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy() # [64, 256]
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg']) # [64, 9]
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts)) # 64
                print("len unique prompts", len(set(prompts))) # 5

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages) # [64, 9], 9步都是一样的取值结果
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0) # [64] all True
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()} # dict_keys(['prompt_embeds', 'pooled_prompt_embeds', 'timesteps', 'latents', 'next_latents', 'log_probs', 'advantages'])

        total_batch_size, num_timesteps = samples["timesteps"].shape # 64, 10; for example         [1000.0000,  960.1293,  913.3490,  857.6923,  790.3683,  707.2785,           602.1506,  464.8760,  278.0488,    8.9286]], device='cuda:0')
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs): # config.train.num_inner_epochs=1
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device) # Returns a random permutation of integers from 0 to n - 1 随机生成一个排列 0 to 63
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                import ipdb; ipdb.set_trace()
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    ) # embeds.shape=[16, 205, 4096]
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    ) # pooled_embeds.shape=[16, 2048]
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps, # [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast(): # 在这个代码块里面，PyTorch 自动帮你选择用半精度还是单精度计算。自动判断，更稳定.
                            import ipdb; ipdb.set_trace()
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config) # transformer=<class 'peft.peft_model.PeftModel'>, diffusion.model.pipeline, NOTE 
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    # NOTE TODO
                                    import ipdb; ipdb.set_trace()
                                    # old: with transformer.module.disable_adapter():
                                    base_model = transformer.module if hasattr(transformer, "module") else transformer
                                    with base_model.disable_adapter(): # NOTE 不用lora adapters是区分的关键了！！！ 这个非常重要
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config) # NOTE TODO why same inputs as line 894??? --> 因为这里禁止了adapter的使用了，从而transformer/pipeline都回到了最初的样子了!!! TODO 没有看到disable_adapter的效果啊，因为prev_sample_mean 和prev_sample_mean_ref是一样的取值... why? prev_sample_mean == prev_sample_mean_ref is true... ipdb> alist = list(transformer.base_model.model.transformer_blocks[0].attn.to_q.lora_A.parameters()) 有意思，这里是有取值的!!! NOTE

                        import ipdb; ipdb.set_trace()
                        '''
                        ipdb> sample['advantages'] output of OCR edit distance, same for all the 9 steps...?
                        tensor([[-0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461,
                                 -0.2461],
                                [-0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461,
                                 -0.2461],
                                [-0.5828, -0.5828, -0.5828, -0.5828, -0.5828, -0.5828, -0.5828, -0.5828,
                                 -0.5828],
                                [-0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461, -0.2461,
                                 -0.2461],
                                [ 0.1151,  0.1151,  0.1151,  0.1151,  0.1151,  0.1151,  0.1151,  0.1151,
                                  0.1151],
                                [ 0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,
                                  0.1480],
                                [ 0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,  0.1480,
                                  0.1480],
                                [ 1.8994,  1.8994,  1.8994,  1.8994,  1.8994,  1.8994,  1.8994,  1.8994,
                                  1.8994]], device='cuda:0', dtype=torch.float64)
                        '''
                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max, # -5
                            config.train.adv_clip_max, # 5
                        ) # tensor([-0.2461, -0.2461, -0.5828, -0.2461,  0.1151,  0.1480,  0.1480,  1.8994], = advantages
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j]) # tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0', ratio=prob.new / prob.ref = p_theta/p_theta_old, NOTE flow-grpo paper's page 4's r_t^i(\theta)'s definition
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range, # 1.0 - 0.0001
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0: # 0.04， KL的系数 beta
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss # NOTE TODO 这个很重要, tensor(-0.1237, device='cuda:0', dtype=torch.float64, grad_fn=<AddBackward0>)
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

