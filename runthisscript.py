import extract_nlp_features
# import predict_brain_from_nlp
import evaluate_brain_predictions
import pickle as pk
import numpy as np
import os
import utils.utils as utils

subject = ['F','H','I','J','K','L','M','N']
seq_lengths = [1, 5, 10, 15, 20, 25, 30]
text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
remove_chars = [",", "\"", "@"]
model_name = 'gpt'
layer = 7
predicted_brain_dir = 'new_predicted_brain/'
eval_predict_dir = 'new_eval_brain_predicts/'
results = {}
for sl in seq_lengths:
    nlp_features = extract_nlp_features.get_gpt_layer_representations(sl, text_array, remove_chars)
    extract_nlp_features.save_layer_representations(nlp_features, model_name, sl, 'new_nlp_features/')

for subject in subject:
    #below copy pasted from predict_brain
    predict_feat_dict = {'nlp_feat_type': model_name,
                         'nlp_feat_dir': 'new_nlp_features/',
                         'layer': layer,
                         'seq_len': sl}
    data = np.load('./data/fMRI/data_subject_{}.npy'.format(subject))
    corrs_t, _, _, preds_t, test_t = utils.run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict)
    fname = 'predict_{}_with_{}_layer_{}_len_{}'.format(subject, model_name, layer, sl)
    print('saving: {}'.format(predicted_brain_dir + fname))
    np.save(predicted_brain_dir + fname + '.npy', {'corrs_t': corrs_t, 'preds_t': preds_t, 'test_t': test_t})
    # above copy pasted from predict_brain

    eval_out_dir = eval_predict_dir + 'predict_'+subject+'_'+model_name+'_layer'+str(layer)+'_seqlen_'+str(sl)
    evaluate_brain_predictions.evaluate(predicted_brain_dir + fname + '.npy', eval_out_dir, subject)

    loaded = pk.load(open('{}_accs.pkl'.format(eval_out_dir), 'rb'))
    mean_subj_acc_across_folds = loaded.mean(1)
    results[subject+'_'+model_name+'_layer'+str(layer)+'_seqlen_'+str(sl)] = mean_subj_acc_across_folds

print(results)
with open('final_results.txt', 'w') as file:
    file.write(str(results))