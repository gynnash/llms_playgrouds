import os
import math
import time
import tiktoken
import inspect
import torch.distributed as dist
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_embed = config.n_embed
        self.n_head = config.n_head

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # att = (q @ k.transpose(-2, -1)) / (1 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T]==0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # 参数共享
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # x 是输入的文本转换成的 token id 序列
        B, T = x.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过 {self.config.block_size} 的限制"
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)  # position embedding (T, n_embed)
        tok_emb = self.transformer.wte(x)    # token embedding (B, T, n_embed)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        # 最后一层 layernorm
        x = self.transformer.ln_f(x)
        # 输出层
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # targets: (B, T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: ", model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768), # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),  # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),   # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257  # gpt模型都是这个数
        config_args['block_size'] = 1024   # gpt模型都是这个数
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 加载已训练的GPT2模型参数
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # 从 tf gpt2 带来的一些需要转置的参数，硬编码在这里手动转置
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 挑选出需要计算梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 区分需要 decay 和不需要 decay 的参数
        # 一般 2D 参数都需要 decay，比如矩阵 weight 参数，embedding等
        # 一般 1D 参数不需要 decay，比如 layernorm 里的参数, bias 参数等
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensor: {len(decay_params)}, with {num_decay_params} params")
        print(f"num nodecay parameter tensor: {len(nodecay_params)}, with {num_nodecay_params} params")
        # fused AdamW 支持一次性将多个参数读入到一个 GPU Kernel 里统一执行更新
        # 普通 AdamW 类似于循环读入多个参数到多个 GPU Kernel 里执行更新
        # 因此 fused AdamW 节省了数据传输、Kernel 启动的开销，在 Cuda 上执行比普通 AdamW 快很多
        # fused 模式目前默认是 False，因为是最新提出的特性，在很多老设备上不支持。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# 简单的数据加载器，用于生成每个 batch 的数据
class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        data_root = 'data/edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards for split {split} in dir {data_root}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

        # with open("data/input.txt", "r") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# 单卡运行脚本： python train_gpt2.py
# 多卡使用 torchrun 运行脚本: torchrun --standalone --nproc_per_node={num_gpus} train_gpt2.py
# torchrun 会设置环境变量：RANK、LOCAL_RANK、WORLD_SIZE
from torch.distributed import init_process_group, destroy_process_group

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    # 每个进程的唯一编号, 不同进程在同一时间运行相同代码, ddp_rank 保证不同进程运行在不同的数据上
    ddp_rank = int(os.environ['RANK'])
    # 同一node(物理机或容器)上如果有多个GPU，会有多个进程，local_rank 表示同一个 node 上的进程相对编号
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # world size 决定启动的进程数
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # 设置主进程，会做一些打印日志、保存 checkpoint 的工作
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # mps 是苹果设备芯片中自带的GPU设备，可以支持 pytorch，比 cpu 快一些
    elif hasattr(torch.backends, 'mps'):
        device = 'mps'
    print(f"using device: {device}")

# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
# print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#total_batch_size = 524288  # GPT2 论文中 125M 版本使用的是 2**19 ～ 0.5M
total_batch_size = 65536  # 因为资源问题，total_batch_size 设置小一点性能会好一点
B, T = 4, 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"希望使用的 batch_size: {total_batch_size}")
    print(f" => 每次计算梯度进行累积的步数: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
t0 = time.time()
model = torch.compile(model)
t1 = time.time()
dt = t1 - t0
print(f"torch.compile time cost: {dt:.2f}s")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
#warmup_steps = 5722 # ~ 375e6/(2**16)，375e6 是 GPT3 论文给的，前 375M token 是 warmup
warmup_steps = 100  # GPT2 124M 和 GPT3 175B 参数量差了1000+倍，所以 warmup 准备设置小一点
max_steps = 152587  # ~ 10e9/(2**16)
def get_lr(it):
    # warmup 截断, lr 线性增长
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 最后阶段，lr 保持最低运行
    if it > max_steps:
        return min_lr
    # 中间阶段，lr 基于 cosine 函数逐渐下降
    # decay_ratio 从 0 到 1
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # coeff 从 1 到 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

import tiktoken
enc = tiktoken.get_encoding('gpt2')

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
    
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 5
        max_length = 30
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
        tokens = tokens.unsqueeze(0)
        tokens = tokens.repeat(num_return_sequences, 1)  # (5, 8)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)  # (B, T, vocab_size)
                # 取 logits 里 T 维度里最后一个位置的 logit
                logits = logits[:, -1, :]  # (B, vocab_size)
                # 将 logits 转成概率的形式
                probs = F.softmax(logits, dim=-1)
                # 采样 top50 
                # topk_probs: (5, 50), topk_indices: (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 从 topk 里选择一个
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (5, 1)
                # 获取选择的 token 在词表中的索引
                xcol = torch.gather(topk_indices, -1, ix)  # (5, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # 一次大 batch_size 更新参数后，将梯度置为 0
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
            #import code; code.interact(local=locals())
        # 因为每次计算的 loss 是对 batch 内样本 loss 求平均，而 grad_accum_steps 每步之间是梯度直接相加
        # 因此需要执行下面这步，保证梯度累积的值，与整个 total_batch_size 一次性求出来的梯度是相同的。
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # 持续计算梯度并累积
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    token_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    token_per_sec = token_processed / dt;
    if master_process and step % 1 == 0:
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time_cost: {dt:.2f}s | tok/sec: {token_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
# 当你不像训练模型，只是使用时，就加一句这个代码
model.eval()
# 有 GPU 的话让模型在 GPU 上跑
model.to('cuda')

# 处理 prompt，把文本转成模型输入形式
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
print(tokens)
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0)
print(tokens)
tokens = tokens.repeat(num_return_sequences, 1)  # (5, 8)
print(tokens)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # 取 logits 里 T 维度里最后一个位置的 logit
        logits = logits[:, -1, :]  # (B, vocab_size)
        # 将 logits 转成概率的形式
        probs = F.softmax(logits, dim=-1)
        # 采样 top50 
        # topk_probs: (5, 50), topk_indices: (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从 topk 里选择一个
        ix = torch.multinomial(topk_probs, 1)  # (5, 1)
        # 获取选择的 token 在词表中的索引
        xcol = torch.gather(topk_indices, -1, ix)  # (5, 1)
        x = torch.cat((x, xcol), dim=1)
print(x)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)