import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model Hyperparameters ---
n_embd = 64  # Embedding dimension
n_head = 4  # Number of attention heads
n_layer = 4  # Number of transformer blocks
dropout = 0.0  
block_size = 32  # Maximum context length for predictions
num_kv_groups = 2



def precompute_rope_embeddings(dim, seq_len, device, theta=10000.0):
    if not (dim % 2 == 0):
        raise ValueError("Dimension must be even for RoPE")
    # (dim / 2)
    #\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2]\}.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device) / dim))
    # (seq_len)
    m = torch.arange(seq_len, device=device)
    # (seq_len, dim / 2)
    # theta = m * Theta
    angle = torch.outer(m, freqs)
    #cos(theta) + i*sin(theta)
    # (max_seq_len, dim // 2) 
    freqs_cis = torch.polar(torch.ones_like(angle), angle)
    return freqs_cis

def apply_rope(x, freqs_cis):
    # x:(Batch, NumHeads, SeqLen, HeadDim)
    # freqs_cis: precomputed cos/sin pairs (SeqLen, HeadDim // 2)
    # x_grouped : (Batch, NumHeads, SeqLen, HeadDim // 2, 2)
    x_grouped = x.float().reshape(*x.shape[:-1], -1, 2)
    # x_complex: (Batch, NumHeads, SeqLen, HeadDim // 2)
    x_complex = torch.view_as_complex(x_grouped)
    # freqs_cis :(SeqLen, HeadDim // 2)
    # Reshape for broadcasting: (1, 1, SeqLen, HeadDim // 2)
    freqs_cis_b = freqs_cis.unsqueeze(0).unsqueeze(0)
    # (x'_even,x'_odd) = (x_even + i*x_odd) * (cos(Theta) + i*sin(Theta))
    # (Batch, NumHeads, SeqLen, HeadDim // 2)
    x_rotated_complex = x_complex * freqs_cis_b
    # (B, H, T, D/2, 2)
    x_rotated = torch.view_as_real(x_rotated_complex) 
    # (B, H, T, D)
    x_out = x_rotated.flatten(start_dim=-2)
    return x_out.type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x / torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.gamma

class GroupQueryAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, n_embd, num_q_heads,num_kv_groups, head_size, block_size, device):
        super().__init__()
        assert num_q_heads % num_q_heads == 0
        assert n_embd % num_q_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_groups = num_kv_groups
        self.head_size = head_size
        self.n_embd = n_embd #  (num_heads * head_size)

        #out feature_size (nums_head * head_dim)
        self.q_proj = nn.Linear(self.n_embd, self.num_q_heads * self.head_size, bias=False)
        #k v out shape(nums_key_value_head * head_dim)
        self.k_proj = nn.Linear(self.n_embd, self.num_kv_groups * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.num_kv_groups * self.head_size, bias=False)
        self.out_proj = nn.Linear(self.num_q_heads * self.head_size, self.n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "freqs_cis", 
            # (block_size, self.head_size // 2)
            precompute_rope_embeddings(self.head_size, block_size, device),
            #fixed value
            persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape # Batch, Block_size, n_embd
 
        # (B, T, n_embd)
        q = self.q_proj(x) 
        k = self.k_proj(x) 
        v = self.v_proj(x) 

        q = q.view(B, T, self.num_q_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_kv_groups, self.head_size).transpose(1, 2) 
        v = v.view(B, T, self.num_kv_groups, self.head_size).transpose(1, 2) 

        # (seq_len, head_size // 2)
        current_freqs_cis = self.freqs_cis[:T]
        
        q_rope = apply_rope(q, current_freqs_cis)
        k_rope = apply_rope(k, current_freqs_cis)

        #Repeat K,V heads
        num_repeats = self.num_q_heads // self.num_kv_groups
        # (B, num_q_heads, T, head_size)
        k_rope_repeated = k_rope.repeat_interleave(num_repeats,dim=1)
        # (B, num_q_heads, T, head_size)
        v_repeated = v.repeat_interleave(num_repeats,dim=1)
        
        wei = q_rope @ k_rope_repeated.transpose(-2, -1) * (self.head_size ** -0.5)
        
        #mask
        tril_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        wei = wei.masked_fill(tril_mask == 0, float('-inf'))
        
        att_probs = F.softmax(wei, dim=-1)
        att_probs = self.dropout(att_probs)
        
        # (B, num_q_heads, T, head_size)
        out = att_probs @ v_repeated 
        # (B, T, n_embd_q_heads_dim)
        out = out.transpose(1, 2).reshape(B, T, self.num_q_heads * self.head_size)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd_param):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_param, 4 * n_embd_param),
            nn.ReLU(),
            nn.Linear(4 * n_embd_param, n_embd_param),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head,num_kv_group, device): 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = GroupQueryAttention(n_embd,n_head,num_kv_group, head_size, block_size, device) 
        self.ffwd = FeedForward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, current_device):
        super().__init__()
        self.device = current_device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head,num_kv_groups,self.device) for _ in range(n_layer)] # Pass self.device to Block
        )
        self.ln_f = RMSNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C_embd)
        # x = tok_emb + pos_emb 
        x = tok_emb 

        x = self.blocks(x)  # (B,T,C_embd)
        x = self.ln_f(x)  # (B,T,C_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C_vocab = logits.shape
            logits = logits.view(B * T, C_vocab)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
         # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] 
            logits, _ = self(idx_cond) 
             # focus only on the last time step
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx 