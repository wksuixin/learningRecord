from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer
import torch
from math import sqrt
import torch.nn.functional as F

# model_path = "/home/wk/code/model/Qwen2.5-0.5B/"
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# text = "这是一个测试transformer的程序"
# inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# print(inputs)

# config = AutoConfig.from_pretrained(model_path)
# token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
# print(token_emb)

# inputs_embs = token_emb(inputs.input_ids)
# print(inputs_embs)

# Q = K = V = inputs_embs
# dim_k = V.size(-1)
# scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
# print(scores)

# weights = F.softmax(scores, -1)
# print(weights.sum(dim=-1))
# weights.shape
# attn_outputs = torch.bmm(weights, V)
# print(attn_outputs.shape)

def scaled_dot_product_atten(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask==0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

def preprocess(tokenizer, embedding_layer, texts):
    inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False)
    inputs_embs = embedding_layer(inputs.input_ids)
    return inputs_embs

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)        
    
    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_atten(self.q(query), self.k(key), self.v(value),\
            query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for i in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([h(query, key, value, query_mask, key_mask, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

if __name__ == "__main__":
    model_path = "/home/wk/code/model/Qwen2.5-0.5B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    
    texts = "这是一个测试transformer的程序"
    inputs_embs = preprocess(tokenizer, token_emb, texts)
    Q  = inputs_embs
    texts = "这是我自己写的程序，是我学习transformer用到的程序"
    inputs_embs = preprocess(tokenizer, token_emb, texts)
    K = V = inputs_embs
    
    # res = scaled_dot_product_atten(Q, K, V)
    # print(res.shape)
    
    # attention_head = AttentionHead(896, 64)
    # outputs = attention_head(Q, K, V)
    # print(outputs.shape)
    
    multi_head = MultiHeadAttention(config)
    multi_head(Q, K, V)
    
    