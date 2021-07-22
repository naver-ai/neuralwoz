# Slot-Utterance Matching for Universal and Scalable Belief Tracking

This repository is an adaptation of its [original](https://github.com/SKTBrain/SUMBT) and [another](https://github.com/stanford-oval/SUMBT) implementation for zero/few-shot experiments.
> Note!: We recommend running this repository in a virtual environment to avoid conflicts.

## Setup

```
mkdir sumbt-env
python3 -m venv sumbt-env/
source sumbt-env/bin/activate
pip3 install -r requirements.txt
```

## Data Preparation

First, you should convert the dataset to the GLUE format with a proper preprocessing using below script. <br>
Before running the script, `labeler_(train|dev|test)_data.json` should be in the `../data` directory. (Please see [README]()) <br>
For the `$TARGET_FILE`, you can specify one of synthetic dialogues following its target_domain and fewshot_proportion (e.g. `nwoz_{target_domain}_{fewshot_proportion}.json`). <br>
You can find the synthetic dialogues in [here](https://drive.google.com/drive/folders/1Tp5CsejVMvWWCn8noPoPys7zN8kfTD_P?usp=sharing). <br>

```
python3 code/transform_augmented_data.py \
  --input_dir ../data \
  --target_file $TARGET_FILE \
  --output_dir ../data
```

It produces `nwoz_{target_domain}_{fewshot_proportion}.tsv` when you specify `$TARGET_FILE=nwoz_{target_domain}_{fewshot_proportion}.json`.<br>

## Train

Please run the below script to train SUMBT model with full/synthetic-augmented dataset. <br>
You should specify `$OUTPUT_PATH` and `$TRAIN_DATA` for it. <br>
For the zero/few-shot experiments, the `$TRAIN_DATA` should be `nwoz_{target_domain}_{fewshot_proportion}.tsv` as following its target domain and fewshot proportion.

```
python3 code/main-multislot.py \
  --do_train \
  --do_eval \
  --num_train_epochs 300 \
  --data_dir ../data \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --task_name bert-gru-sumbt \
  --nbt rnn \
  --output_dir $OUTPUT_DIR \
  --target_slot all \
  --warmup_proportion 0.1 \
  --learning_rate 1e-4 \
  --train_batch_size 4 \
  --eval_batch_size 16 \
  --distance_metric euclidean \
  --patience 15 \
  --tf_dir tensorboard \
  --hidden_dim 300 \
  --max_label_length 32 \
  --max_seq_length 64 \
  --max_turn_length 22 \
  --experiment multiwoz2.1 \
  --train_file_name $TRAIN_DATA \
  --save_inference_results
```

## Test

To evaluate pretrainde SUMBT model, please specify $MODEL_PATH for the path of the model which includes `pytorch_model.bin`, `config.json`, and `vocab.txt`.
You can find the pretrained model in this [link](https://drive.google.com/drive/folders/1FI_ITwyAqqYt3nACUe65ZKFF6SsjCyi8).

```
python3 code/main-multislot.py \
  --do_eval \
  --output_dir $MODEL_PATH \
  --data_dir ../data \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --task_name bert-gru-sumbt \
  --nbt rnn \
  --target_slot all \
  --train_batch_size 4 \
  --eval_batch_size 16 \
  --distance_metric euclidean \
  --hidden_dim 300 \
  --max_label_length 32 \
  --max_seq_length 64 \
  --max_turn_length 22 \
  --experiment multiwoz2.1 \
  --save_inference_results
```