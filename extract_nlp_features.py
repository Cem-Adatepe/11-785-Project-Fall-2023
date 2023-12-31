from utils.gpt_utils import get_gpt_layer_representations
from utils.llama_utils import get_llama_layer_representations
import time as tm
import numpy as np
import torch
import os
import argparse

                
def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):             
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

                
model_options = ['gpt', 'llama']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert', choices=model_options)                
    parser.add_argument("--sequence_length", type=int, default=1, help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--output_dir", required=True, help='directory to save extracted representations to')
    parser.add_argument("--uniform_layer", type=int, default=None, help="layer to make uniform (default: None)")

    args = parser.parse_args()
    print(args)
    
    text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
    remove_chars = [",","\"","@"]

    if args.nlp_model == 'gpt':
        nlp_features = get_gpt_layer_representations(args.sequence_length, text_array, remove_chars, args.uniform_layer)
    elif args.nlp_model == 'llama':
        nlp_features = get_llama_layer_representations(args.sequence_length, text_array, remove_chars, args.uniform_layer)
    else:
        print('Unrecognized model name {}'.format(args.nlp_model))
        
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)          
              
    save_layer_representations(nlp_features, args.nlp_model, args.sequence_length, args.output_dir)
        
        
        
        
    
    
    

    
