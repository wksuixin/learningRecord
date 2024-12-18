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
        x_list = [h(query, key, value, query_mask, key_mask, mask) for h in self.heads]
        x = torch.cat(x_list, dim=-1)
        x = self.output_linear(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3) #config.hidden_dropout_prob)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, mask=None):
        hidden_state = self.layer_norm_1(x)
        x = x+self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        x = self.layer_norm_2(x)
        x = x+self.feed_forward(x)
        return x

class MyEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps = 1e-12)
        self.dropout = nn.Dropout()
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings 
    




class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(0.5)#config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.drop_out(x)
        return x



if __name__ == "__main__":
    model_path = "/home/wk/code/model/Qwen2.5-0.5B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    feed_forward = FeedForward(config)
    
    texts = "这是一个测试transformer的程序"
    inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False)
    inputs_embs = preprocess(tokenizer, token_emb, texts)
    Q  = inputs_embs
    texts = "这是我自己写的程序，是我学习transformer用到的程序"
    inputs_embs = preprocess(tokenizer, token_emb, texts)
    K = V = inputs_embs
    
    
    multi_head = MultiHeadAttention(config)
    atten_output = head_output = multi_head(Q, K, V)
    

    feed_dorward = FeedForward(config)
    feed_dorward(atten_output)
    out = feed_forward(head_output)
    print(out.shape)
    
    
    encode_layer = TransformerEncoderLayer(config)
    encode_layer(Q)
    
    embedding_layer = MyEmbeddings(config)
    embedding_layer(inputs.input_ids)

    