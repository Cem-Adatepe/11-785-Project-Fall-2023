import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import time as tm

def get_gpt_layer_representations(seq_len, text_array, remove_chars):
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    #print(text_array.shape)
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

    # inputs = tokenizer(list(text_array), return_tensors="pt", is_split_into_words = True)
    # outputs = model(**inputs, output_hidden_states = True)
    # print(type(outputs.hidden_states))
    #print(outputs.hidden_states[0].detach().numpy().shape)
    #return dict(zip(range(-1, 12), [tensor.detach().numpy() for tensor in outputs.hidden_states]))
    # get the token embeddings
    # token_embeddings = []
    # for word in text_array:
    #     # print("-------------------------------------------------------------------")
    #     # print("Word: " + word)
    #     current_token_embedding = get_bert_token_embeddings([word], tokenizer, model, remove_chars)
    #     token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))

    # where to store layer-wise bert embeddings of particular length
    # BERT = {}
    # for layer in range(12):
    #     BERT[layer] = []
    # BERT[-1] = token_embeddings

    # from_start_word_ind_to_extract = seq_len
    # start_time = tm.time()

    # # before we've seen enough words to make up the sequence length, add the representation for the last word 'seq_len' times
    # word_seq = text_array[:seq_len]
    # for _ in range(seq_len):
    #     BERT = add_avrg_token_embedding_for_specific_word(word_seq,
    #                                                                  tokenizer,
    #                                                                  model,
    #                                                                  remove_chars,
    #                                                                  from_start_word_ind_to_extract,
    #                                                                  BERT)

    # # then add the embedding of the last word in a sequence as the embedding for the sequence
    # for end_curr_seq in range(seq_len, len(text_array)):
    #     word_seq = text_array[end_curr_seq-seq_len+1:end_curr_seq+1]
    #     BERT = add_avrg_token_embedding_for_specific_word(word_seq,
    #                                                       tokenizer,
    #                                                       model,
    #                                                       remove_chars,
    #                                                       from_start_word_ind_to_extract,
    #                                                       BERT)

    #     if end_curr_seq % 100 == 0:
    #         print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time()-start_time))
    #         start_time = tm.time()

    # print('Done extracting sequences of length {}'.format(seq_len))

    # return BERT

# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
def predict_bert_embeddings(words_in_array, tokenizer, model, remove_chars):

    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1

    n_seq_tokens = 0
    seq_tokens = []

    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word

        if word in ['[CLS]', '[SEP]']:     # [CLS] and [SEP] are already tokenized
            word_tokens = [word]
        else:
            word_tokens = tokenizer(word, return_tensors = "pt")

        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1
    # convert token to vocabulary indices
    inputs = tokenizer( seq_tokens, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states = True)
    encoded_layers = dict(zip(range(-1, 12), [tensor.detach().numpy() for tensor in outputs.hidden_states]))
    pooled_output = np.squeeze(model.pooler(encoded_layers[-1]).detach().numpy())

    return encoded_layers, word_ind_to_token_ind, pooled_output

# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_bert_embedding(bert_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        bert_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            bert_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word
    return bert_dict

# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: BERT tokenizer
# model: BERT model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# bert_dict: where to save the extracted embeddings
def add_avrg_token_embedding_for_specific_word(word_seq,tokenizer,model,remove_chars,from_start_word_ind_to_extract,bert_dict):

    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_bert_embeddings(word_seq, tokenizer, model, remove_chars)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    bert_dict = add_word_bert_embedding(bert_dict, all_sequence_embeddings,token_inds_to_avrg)

    return bert_dict


# get the BERT token embeddings
def get_bert_token_embeddings(words_in_array, tokenizer, model, remove_chars):
    #print(words_in_array)
    #print("LEN" + str(len(words_in_array)) + str(words_in_array))
    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1

    n_seq_tokens = 0
    seq_tokens = []
    if(len(words_in_array) > 1):
        print(words_in_array)
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word

        if word in ['[CLS]', '[SEP]']:     # [CLS] and [SEP] are already tokenized
            word_tokens = [word]
        else:
            word_tokens = tokenizer.tokenize(word)
        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1
    #print(word_ind_to_token_ind)
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    token_embeddings = model.embeddings.forward(tokens_tensor)
    print("DIM" + str(token_embeddings.shape) + str(token_embeddings))
    return token_embeddings


# add the embeddings for all individual words
# specific_layer specifies only one layer to add embeddings from
def add_all_bert_embeddings(bert_dict, embeddings_to_add, specific_layer=-1):
    if specific_layer >= 0:
        layer_embedding = embeddings_to_add[specific_layer]
        seq_len = layer_embedding.shape[1]
        full_sequence_embedding = layer_embedding.detach().numpy()

        for word in range(seq_len):
            bert_dict[specific_layer].append(full_sequence_embedding[0,word,:])
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            seq_len = layer_embedding.shape[1]
            full_sequence_embedding = layer_embedding.detach().numpy()

            for word in range(seq_len):
                bert_dict[layer].append(full_sequence_embedding[0,word,:])
    return bert_dict


# add the embeddings for only the last word in the sequence that is not [SEP] token
def add_last_nonsep_bert_embedding(bert_dict, embeddings_to_add, specific_layer=-1):
    if specific_layer >= 0:
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()

        bert_dict[specific_layer].append(full_sequence_embedding[0,-2,:])
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()

            bert_dict[layer].append(full_sequence_embedding[0,-2,:])
    return bert_dict

# add the CLS token embeddings ([CLS] is the first token in each string)
def add_cls_bert_embedding(bert_dict, embeddings_to_add, specific_layer=-1):
    if specific_layer >= 0:
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()

        bert_dict[specific_layer].append(full_sequence_embedding[0,0,:])
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()

            bert_dict[layer].append(full_sequence_embedding[0,0,:])
    return bert_dict