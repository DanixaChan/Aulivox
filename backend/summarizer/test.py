from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
print(tokenizer)