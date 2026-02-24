#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import torch


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length, # 128 NOTE
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids # </s>=1, <pad>=0, for ''=prompt input, text_input_ids=[1, 0...0]=[</s>, <pad>]
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0] # NOTE 24 layers of T5Block, and even same token of <pad>, this text_encoder will change the output prompt embedding vectors; out.shape=[1, 128, 4096]

    dtype = text_encoder.dtype # torch.float16
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1) # [1, 128, 4096]
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds # [1, 128, 4096]

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    #import ipdb; ipdb.set_trace()
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77, # NOTE this is important! '' -> padded to a length of 77 now
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0] # NOTE TODO <CLS>对应的vector
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape # seq_len=77 = max_length in Line 74
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds
    # [1, 77, 768], [1, 768]; 2nd tokenizer, add '<|startoftext|>'=49406 and '<|endoftext|>'=49407 to current prompt text! NOTE; for the 2nd tokenizer, the output is prompt_embeds.shape=[1, 77, 1280], pooled_prompt_embeds.shape=[1, 1280]

def encode_prompt(
    text_encoders, # 3 text encoders
    tokenizers, # 3 tokenizers for the textual prompt embedding
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2] # use the first two tokenizers
    clip_text_encoders = text_encoders[:2] # use the first two text encoders

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt, # ['']
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt, # 1
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None, # text_input_ids_list=None
        )
        clip_prompt_embeds_list.append(prompt_embeds) # [1, 77, 768]
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds) # [1, 768]

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1) # [1, 77, 768] and [1, 77, 1280] -> [1, 77, 2048]
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1) # [1, 77, 768] and [1, 77, 1280] -> [1, 77, 2048] 这是把两个tokenizer的结果，在feature dimension拼接起来

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    ) # t5_prompt_embed.shape=[1, 128, 4096]

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    ) # 因为t5 tokenizer的feature dim是4096，而前面两个tokenizers的feature dimension是2048，所以需要把2048补全pad到长度4096。[1, 77, 2048] -> [1, 77, 4096]
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2) # [1, 77, 4096] and [1, 128, 4096] -> [1, 205, 4096] NOTE 有意思，前面两个tokenizer是在feature dim拼接的，然后长度维持77。而前两个tokenizer的结果，和第三个t5 tokenizer合并的时候，是在sequence length上面拼接的，即77+128 -> 205是新的长度了。

    return prompt_embeds, pooled_prompt_embeds
    # [1, 205, 4096] and [1, 2048]

