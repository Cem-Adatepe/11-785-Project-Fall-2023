import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import time as tm

def get_gpt_layer_representations(seq_len, text_array, remove_chars):
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    print(text_array.shape)
    GPT = {}
    for i in range(-1, 12):
        GPT[i] = []
    for i, word in enumerate(text_array):
        if i % 100 == 0:
            print(i)
        inputs = tokenizer(word, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states = True)
        #print(outputs.hidden_states[0].shape)
        states = outputs.hidden_states
        for i in range(-1, 12):
            layer = np.mean(states[i+1].detach().numpy(), 1)
            GPT[i].append(layer)
    