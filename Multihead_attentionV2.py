import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) #C Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # .view implicitly split the matrix by adding a `num_heads` dimension. Then unroll last dim:
                                                                        # (b,num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)                                                                        
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)#D
        keys = keys.transpose(1, 2) #E Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens,head_dim)
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I
        #J
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec

## To illustrate this batched matrix multiplication, ####################
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], #The shape of this tensor is (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
[0.8993, 0.0390, 0.9268, 0.7388],
[0.7179, 0.7058, 0.9156, 0.4340]],
[[0.0772, 0.3565, 0.1479, 0.5331],
[0.4066, 0.2318, 0.4545, 0.9737],
[0.4606, 0.5159, 0.4220, 0.5786]]]]) 

## a batched matrix multiplication between the tensor itself and a view of the tensor where we transposed the last two dimensions, num_tokens and head_dim
print(a @ a.transpose(2, 3)) 

## the result below is the same as matrix multiplication ####################
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)  