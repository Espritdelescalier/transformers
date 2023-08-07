from transformers import CURForCausalLM, AutoTokenizer
import torch

torch_device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = CURForCausalLM.from_pretrained(
    "./checkpoints_wiki/").to(torch_device)


model_inputs = tokenizer('As a note, the particular relationship of our model to the model of Krotov & Hopfield (2020), is what they refer to as a ‘type B’ model. These are models with contrastive normalisation on the memory neurons (via a softmax in our case), as opposed to ‘type B’ models which have a power activation function on the memory neurons. TEM (left hand side of Equation 11) corresponds to a linear activation.', return_tensors='pt').to(torch_device)

# generate 40 new tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
