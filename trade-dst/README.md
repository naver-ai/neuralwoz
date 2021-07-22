## TRADE Multi-Domain and Unseen-Domain Dialogue State Tracking


This repository is an adaptation of the [its original implementation](https://github.com/jasonwu0731/trade-dst).


## Train


- TRAIN_DATASET: The file name of training dataset
     - Use synthetic data from NeuralWOZ named `nwoz_{target_domain}_{fewshot_proportion}.json` for zero/few-shot learning
     - You can find the data in [here](https://drive.google.com/drive/folders/1Tp5CsejVMvWWCn8noPoPys7zN8kfTD_P)
     - `labeler_train_data.json` for full data training

- TARGET_DOMAIN: target domain (hotel|restaurant|attraction|train|taxi) to evaluate in leave-one-out setup
     - It should be matched with the target domain of the training dataset
     - e.g. `-train_data=nwoz_hotel_{fewshot_proportion}.json -target_domain=hotel`

- OUTPUT_PATH: output path to save model and corresponding outputs

```
python3 myTrain.py \
  -dec=TRADE \
  -bsz=32 \
  -dr=0.2 \
  -lr=0.001 \
  -le=1 \
  -clip=5 \
  -dataset_dir=../data \
  -train_data=$TRAIN_DATASET \
  -target_domain=$TARGET_DOMAIN \
  -output_path=$OUTPUT_PATH
```

## Test

- MODEL_PATH: checkpoint path of pretrained TRADE model
    - Please download the pretrained models in [here](https://drive.google.com/drive/folders/1vXQVf5ONMqHTvUoTL_omgLqYaaEyXuro)
    - Please use `tar -zxvf MODEL.tar.gz` for the unzipping.
    - You should specify the `MODEL_PATH` as `nwoz_TRADE_{target_domain}_{fewshot_proportion}/TRADE-multiwozdst/BEST_CHECKPOINT/`
    - e.g> `-path=nwoz_TRADE_hotel_0/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.4006`
    - For saving storage, the unzipped directory contains only the best checkpoint.

- TARGET_DOMAIN: target domain (hotel|restaurant|attraction|train|taxi) to evaluate in leave-one-out setup
    - It should be matched with the target domain of the checkpoint
    - e.g> `-path=nwoz_TRADE_hotel_0/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.4006 -exceptd=hotel`

```
python3 myTest.py \
  -dataset_dir=../data \
  -path=$MODEL_PATH \
  -exceptd=$TARGET_DOMAIN \
  -gs=1
```

