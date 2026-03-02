import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std # 是否使用整个group的std，或者是只用当前prompt的多个outputs的std
        self.stats = {}
        self.history_prompts = set()
    def update(self, prompts, rewards, type='grpo'): # prompts=a list with 64 textual sequences; rewards.shape=(64,9)
        #import ipdb; ipdb.set_trace()
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts) # 5 unique prompts
        advantages = np.empty_like(rewards)*0.0 # (64,9) all 0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt] # NOTE (8,9) 这是只抽取当前的prompt相关的8个rewards
            if prompt not in self.stats:
                self.stats[prompt] = []
            else:
                self.stats[prompt] = self.stats[prompt].tolist()
            self.stats[prompt].extend(prompt_rewards) # TODO a bug was here
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True) # (8, 9) -> mean -> (1, 9)
            if self.global_std: # True
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards; rewards.shape=(64,9) -> std -> (1,9) NOTE 这是用了64个样本的std了，有意思
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo': # NOTE here
                advantages[prompts == prompt] = (prompt_rewards - mean) / std # NOTE TODO 这个就是严格按照定义走的!!! shape=(64,9) 只有prompt相关的8个位置上有取值
            elif type=='rwr': # just use the original reward scores! rwr = reward
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
                # 'a', 1 3 6, then advantages = [0, 0, 0, 0, 0, 1###], 
                # 'b', 2, 5, then advantates = [0, 0, 0, 0, 1###, 1], 
                # 'c', 4, then advantages = [0, 0, 0, 1###, 1, 1] 一个prompt，如果有多个candidate and rewards，那么只选择一个reward最大的！！！其他的都是0，即ignore NOTE
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    #max_idx = 1 # TODO if there is only sample??? what to do?
                    max_idx = 1 if len(prompt_advantages) > 1 else 0

                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0 # 得分reward最大的，作为1.0，positive example
                result[min_idx] = -1.0 # 得分reward最小的，作为-1.0，negative example
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages
        # when 'sft', return is advantages=array([0., 0., 0., 1., 1., 1.])

        # when 'dpo', for 'a' with 1, 3, 6; advantages=array([-1.,  0.,  0.,  0.,  0.,  1.])
        # finally is: Advantages: [-1. -1.  0. -1.  1.  1.], since 'c' has only one candidate, so no way to distingush '1' and '-1'! NOTE

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0 # NOTE 计算平均一个prompt里面有几个output candidates，即平均的Group的大小；对于下面的测试的例子来说，a, b, c三个prompts，然后a有3个候选，b有2个,c有1个，这样就是：(3+2+1)/3=2.0=average group size。
        history_prompts = len(self.history_prompts) # 目前为止的unique textual prompt的数量
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a'] # 3 'a', 2 'b', and 1 'c'
    rewards = [1, 2, 3, 4, 5, 6]
    for atype in ['grpo', 'rwr', 'sft', 'dpo']:
        print('----{}----'.format(atype))
        advantages = tracker.update(prompts, rewards, type=atype)
        print("Advantages:", advantages)
        avg_group_size, history_prompts = tracker.get_stats()
        print("Average Group Size:", avg_group_size)
        print("History Prompts:", history_prompts)
        tracker.clear()
        print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()
