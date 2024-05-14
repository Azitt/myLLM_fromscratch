import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #Tokenize the entire text
        
        for i in range(0, len(token_ids) - max_length, stride): #Use a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): # Return the total number of rows in the dataset
            return len(self.input_ids)
    def __getitem__(self, idx): # Return a single row from the dataset
            return self.input_ids[idx], self.target_ids[idx]
        
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #Initialize the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #Create dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) # drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
    return dataloader 

with open("the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
first_batch = next(data_iter)
print(first_batch)       