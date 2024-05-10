import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# remove first 50 tokens########################
enc_sample = enc_text[50:]

# create input-target pair##########################
context_size = 4 # context size determines how many tokens are included in the input

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]  # targets= inputs that are shifted by one position
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size+1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(context, "---->", desired)

# covert token IDS to text ##############################
for i in range(1, context_size+1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(tokenizer.decode(context), "---->", tokenizer.decode([desired])) 
 