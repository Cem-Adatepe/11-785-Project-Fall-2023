import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import time as tm

def get_gpt_layer_representations(seq_len, text_array, remove_chars, uniform_layer):
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    GPT = {}
    for i in range(-1, 12):
        GPT[i] = []

    for i in range(len(text_array)):
        if i % 100 == 0:
            print(i)
        if i < seq_len:
            words = text_array[: seq_len]
        else:
            words = text_array[i - seq_len + 1: i + 1]
        tokens = []
        for j in range(len(words)):
            token = (tokenizer(words[j], return_tensors="pt")['input_ids']).tolist()
            tokens.extend(token[0])


        inputs = torch.tensor([tokens])



        outputs = model(inputs, output_hidden_states = True)
        #print(outputs.hidden_states[0].shape)
        states = outputs.hidden_states
        for i in range(-1, 12):
            layer = np.mean(states[i+1].detach().numpy(), 1)
            GPT[i].append(layer)
    return GPT

    