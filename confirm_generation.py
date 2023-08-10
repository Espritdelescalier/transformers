from transformers import CURForCausalLM, AutoTokenizer
import torch

torch_device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = CURForCausalLM.from_pretrained(
    "./checkpoints_wiki/").to(torch_device)


model_inputs = tokenizer(
    'My cat is sick so I will take him to', return_tensors='pt').to(torch_device)

# generate 40 new tokens
greedy_output = model.generate(**model_inputs,
                               max_new_tokens=40,
                               no_repeat_ngram_size=2,
                               early_stopping=True)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
