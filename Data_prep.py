import re
from SimpleTokenizer import SimpleTokenizerV1
from SimpleTokenizerV2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total number of character:",len(raw_text))
print(raw_text[:99]) 


text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)   
# separating punctuation#####################
result = re.split(r'([,.]|\s)', text)
print(result)

# removing whitespace characters##############
result = [item for item in result if item.strip()]
print(result)

# to handle other types of punctuation########
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# basic tokenizer for the entire input story###############
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("number of tokens in this text(without whitespaces):",len(preprocessed))

# the first 30 tokens#####################################
print(preprocessed[:30])

## create a list of unique tokens from input text#################################
all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print(vocab_size)
## create vocabulary ############################################################
vocab = {token:integer for integer,token in enumerate(all_words)}
for i,v in enumerate(vocab.items()):
    print(v)
    if i > 50:
        break
## calling class########################
tokenizer = SimpleTokenizerV1(vocab) 
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
pride."""
ids = tokenizer.encode(text)
print(ids)   
print(tokenizer.decode(ids))
## modified vocab to have speciall tokens########################################
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
 print(item)
 
### tokenizerv2 that handle unknown word##########################################         
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))