"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50_304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int= 768
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optianal bias. PyTorch does not support simply bias=False in LayerNorm """
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        #regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        #flash attention make GPU go brrrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding size (n_embd)

        # calculate query, key, values ofr all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_size]
        
        # causal self-attention; self-attend: [B, n_head, T, head_size] x [B, n_head, head_size, T] -> [B, n_head, T, T]
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # [B, n_head, T, T] x [B, n_head, T, head_size] -> [B, n_head, T, head_size]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.act = new_gelu
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
    
    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)    
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self):
        pass
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of lengh {t}, block size is {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape [1, t]

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape [b, t, n_embd]
        pos_emb = self.transformer.wpe(pos)  # positional embeddings of shape [1, t, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)  # [b, t, n_embd]
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # [b, t, n_embd]
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # [b, t, vocab_size]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # [b, 1, vocab_size]  note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    def crop_block_size(self):
        pass
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        pass
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        
        # separate out all parameters to those that will and won't experience regularizing weight decay
        pass
    
    def estimate_mfu(self):
        pass
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequencecontext is growing too long, we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)  # [b, min(idx_cond.size(1), block_size), vocab_size]
            # pluck the logits at the final step and scale by desired terperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution to get the next index
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx