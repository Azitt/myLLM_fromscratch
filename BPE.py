from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")

# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
text = "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# convert the token IDs back into text####
strings = tokenizer.decode(integers)
print(strings)
