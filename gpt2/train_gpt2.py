import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

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
        att = (q @ k.transpose(-2, -1)) / (1 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

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
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

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

    def forward(self, x):
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
        logits = self.lm_head(x)
        return logits

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