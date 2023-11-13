# 11-785-Project-Fall-2023

Repository for 11-785 Group Project and Research Paper.

## Deriving representations of text from an NLP model

python extract_nlp_features.py --nlp_model [bert/transformer_xl/elmo/use] --sequence_length s --output_dir nlp_features

## Building encoding model to predict fMRI recordings

python predict_brain_from_nlp.py --subject [F,H,I,J,K,L,M,N] --nlp_feat_type [bert/elmo/transformer_xl/use] --nlp_feat_dir INPUT_FEAT_DIR --layer l --sequence_length s --output_dir OUTPUT_DIR

## Evaluating the predictions of the encoding model using classification accuracy

python evaluate_brain_predictions.py --input_path INPUT_PATH --output_path OUTPUT_PATH --subject [F,H,I,J,K,L,M,N]

## Print per-subject average accuracy per CV fold

extract_brain_predictions.py --input_path INPUT_PATH
