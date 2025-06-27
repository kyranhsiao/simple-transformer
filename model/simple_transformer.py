import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from math import sqrt



def scaled_dot_product_attention(query, key, value,
                                 query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None: 
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)

    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(in_features=embed_dim, out_features=head_dim)
        self.k = nn.Linear(in_features=embed_dim, out_features=head_dim)
        self.v = nn.Linear(in_features=embed_dim, out_features=head_dim)

    def forward(self, query, key, value,
                query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(self.q(query), self.k(key), self.v(value),
                                                    query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of attention heads!"
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    
    def forward(self, query, key, value,
                query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.linear_2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeds = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        # learnable position embedding
        self.pos_embeds = nn.Embedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        token_embeds = self.token_embeds(input_ids)
        pos_embeds = self.pos_embeds(pos_ids).to(input_ids.device)

        embeds = token_embeds + pos_embeds
        embeds = self.layer_norm(embeds)
        embeds = self.dropout(embeds)

        return embeds

"""
#########################################
The method of abosulte position embedding
#########################################
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeds = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        # abosulte position embedding
        self.register_buffer("pos_embeds", self._build_sinusoidal_position_embedding(config.max_position_embeddings, config.hidden_size))
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def _build_sinusoidal_position_embedding(self, max_len, dim):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)

        token_embeds = self.token_embeds(input_ids)
        pos_embeds = self.pos_embeds[:, :seq_len, :].to(input_ids.device)
        
        embeds = token_embeds + pos_embeds
        embeds = self.layer_norm(embeds)
        embeds = self.dropout(embeds)

        return embeds
"""
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mask=None):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attn(hidden_state, hidden_state, hidden_state,
                               mask=mask)
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size)
        self.masked_attn = MultiHeadAttention(config) # self attention layer with causal mask
        self.attn = MultiHeadAttention(config) # cross attention layer
        self.feed_forward = FeedForward(config)
    
    def forward(self, x,  enc_output, tgt_mask=None, mem_mask=None):
        hidden_state_1 = self.layer_norm_1(x)
        # decoder can only attend to previous tokens
        x = x + self.masked_attn(hidden_state_1, hidden_state_1, hidden_state_1,
                                 mask=tgt_mask)
        hidden_state_2 = self.layer_norm_2(x)
        x = x + self.attn(hidden_state_2, enc_output, enc_output,
                          mask=mem_mask)
        x = x + self.feed_forward(self.layer_norm_3(x))

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, enc_output, tgt_mask=None, mem_mask=None):
        batch_size, seq_len, _ = x.size()
        # tgt_mask must be a lower triangular matrix
        if tgt_mask is None:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
            tgt_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, mem_mask)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeds = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src_ids, tgt_ids, 
                src_mask=None, tgt_mask=None, mem_mask=None,
                return_logits=False):
        src_embeds = self.embeds(src_ids)
        tgt_embeds = self.embeds(tgt_ids)

        mem = self.encoder(src_embeds, mask=src_mask)

        output = self.decoder(tgt_embeds, mem, tgt_mask=tgt_mask, mem_mask=mem_mask)

        logits = self.linear(output)

        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=-1)

if __name__ == "__main__":
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    src_txt = "time flies like an arrow"
    tgt_txt = "fast birds fly"

    src_inputs = tokenizer(src_txt, return_tensors="pt", add_special_tokens=False)
    tgt_inputs = tokenizer(tgt_txt, return_tensors="pt", add_special_tokens=False)

    print(f"{'src input_ids:':<20}{src_inputs.input_ids}")
    print(f"{'tgt input_ids:':<20}{tgt_inputs.input_ids}")

    config = AutoConfig.from_pretrained(model_ckpt)
    print(f"{'config vocab size: ':<20}{config.vocab_size}")

    transformer = Transformer(config)
    
    logits = transformer(src_inputs.input_ids, tgt_inputs.input_ids, 
                         return_logits=True)
    print(f"{'logits shape: ':<20}{logits.shape}")

    probs = transformer(src_inputs.input_ids, tgt_inputs.input_ids)
    print(f"{'probs shape: ':<20}{probs.shape}")

