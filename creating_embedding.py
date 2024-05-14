import torch
from torch.utils.data import Dataset, DataLoader
##Suppose we have the following four input tokens with IDs 2, 3, 5, and 1###
input_ids = torch.tensor([2, 3, 5, 1])

## suppose we have a small vocabulary of only 6 words#############
## and we want to create embeddings of size 3 
## in embedding weight there is one row for each of token. here the output is a 6*3 matrix

vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
## tensor([[ 0.3374, -0.1778, -0.1690],
        # [ 0.9178,  1.5810,  1.3010],
        # [ 1.2753, -0.2010, -0.1606],
        # [-0.4015,  0.9666, -1.1481],
        # [-1.1589,  0.3255, -0.6315],
        # [-2.8400, -0.7849, -1.4096]], requires_grad=True

# After we instantiated embedding layer, let's now apply it to a token ID to obtain embedding vector
print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015, 0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>) this is line 4 of above matrix

print(embedding_layer(input_ids))
# tensor([[ 1.2753, -0.2010, -0.1606],
# [-0.4015, 0.9666, -1.1481],
# [-2.8400, -0.7849, -1.4096],
# [ 0.9178, 1.5810, 1.3010]], grad_fn=<EmbeddingBackward0>)
## more realistic embedding dimension ########################
# 256-dimensional vector representation
import GPTDatasetV1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
 
output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = GPTDatasetV1.create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

## use the embedding layer to embed these token IDs into 256-dimensional vectors#######################
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
# torch.Size([8, 4, 256]) => each token ID is now embedded as a 256-dimensional vector

## GPT model's absolute embedding approach = create another embedding layer that has the same dimension astoken_embedding #############
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

## input embedding ############################################
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)