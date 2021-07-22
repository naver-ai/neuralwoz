"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import json
import h5py
import os
import argparse
import random
from collections import defaultdict
from transformers import BartTokenizer, RobertaTokenizer
from utils.database import Database
from utils.data_utils import make_slot_meta
from utils.constants import SPECIAL_TOKENS, EXPERIMENT_DOMAINS
from utils.labeler_utils import build_labeler_data
from utils.collector_utils import build_collector_data


def get_valid_key(domain_transitions, all_keys, rng, exceptd=None, fewshot_ratio=0., fewshot_key=None):
    select_ids = []
    if exceptd and fewshot_ratio > 0.:
        fk = exceptd + "_" + str(fewshot_ratio)
        if fewshot_key and fewshot_key.get(fk):
            print("select fewshot ids from predefined fewshot key!")
            select_ids = fewshot_key[fk]
        else:
            print("randomly select fewshot ids!")
            reverse = defaultdict(list)
            for k, doms in domain_transitions.items():
                for dom in doms:
                    reverse[dom].append(k)

            all_length = len(reverse[exceptd])
            k = int(all_length * fewshot_ratio)
            to_select_idx = rng.sample(range(all_length), k=k)
            select_ids = [el for idx, el in enumerate(reverse[exceptd]) if idx in to_select_idx]

    print(f"selected ids: {len(select_ids)}")
    
    valid_key = []
    for key in all_keys:
        if key not in domain_transitions:
            continue

        if key in select_ids:
            valid_key.append(key)
            continue

        if exceptd and exceptd in domain_transitions[key]:
            continue

        valid_key.append(key)
    return valid_key


def preprocess_collector(args, valid_key=[]):
    data = json.load(open(os.path.join(args.dataset_path, args.collector_file_name)))
    database = Database(args.dataset_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_data, nkey = build_collector_data(data, database, tokenizer, valid_key)
    print(f"# errorneous cases: {len(nkey)}")
    train_data = [d.to_dict() for d in train_data]
    output_file_name = f"{str(args.exceptd)}_{args.fewshot_ratio}_collector_train.json"
    json.dump(train_data, open(os.path.join(args.output_path, output_file_name), "w", encoding="utf-8"))
    
    if os.path.exists(os.path.join(args.dataset_path, 'collector_dev_data.json')):
        data = json.load(open(os.path.join(args.dataset_path, 'collector_dev_data.json')))
        dev_data, _ = build_collector_data(data, database, tokenizer)
        dev_data = [d.to_dict() for d in dev_data]
        output_file_name = f"{str(args.exceptd)}_{args.fewshot_ratio}_collector_dev.json"
        json.dump(dev_data, open(os.path.join(args.output_path, output_file_name), "w", encoding="utf-8"))
    

def preprocess_labeler(args, valid_key=[]):
    rng = random.Random(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    
    slot_meta, ontology = make_slot_meta(os.path.join(args.dataset_path, "ontology.json"))
    slot_desc = json.load(open(os.path.join(args.dataset_path, "slot_descriptions.json")))
    data = build_labeler_data(os.path.join(args.dataset_path, args.labeler_file_name),
                              slot_meta, tokenizer, slot_desc, ontology, rng, unk="<unk>", valid_key=valid_key)
    
    output_file_name = f"{str(args.exceptd)}_{args.fewshot_ratio}_labeler_train.h5"
    file = h5py.File(os.path.join(args.output_path, output_file_name), "w")
    d_data = file.create_dataset("data",
                                 dtype=h5py.string_dtype(),
                                 shape=(100, ),
                                 chunks=True,
                                 maxshape=(None,),
                                 compression="gzip")
    index = 0
    skip = 0
    instances = []
    for i, d in enumerate(data):
        instances.append(json.dumps(d.to_dict()))
        index += 1

        if index % 100 == 0:
            d_data.resize(index, axis=0)
            d_data[index-100:index] = instances
            instances = []
            print(f"[{i}/{len(data)}]")

    if len(instances) > 0:
        d_data.resize(index, axis=0)
        d_data[index - len(instances):index] = instances
        instances = []
    file.close()
    print(index)


def main(args):
    rng = random.Random(args.seed)
    data = json.load(open(os.path.join(args.dataset_path, args.collector_file_name)))
    domain_transitions = json.load(open(os.path.join(args.dataset_path, "train_domain_transitions.json"), "r", encoding="utf-8"))

    fewshot_key = {}
    if os.path.exists(os.path.join(args.dataset_path, "assets", "fewshot_key.json")):
        fewshot_key = json.load(open(os.path.join(args.dataset_path, "assets", "fewshot_key.json"), "r", encoding="utf-8"))

    all_keys = list(data.keys())
    valid_key = get_valid_key(domain_transitions,
                              all_keys,
                              rng,
                              args.exceptd,
                              args.fewshot_ratio,
                              fewshot_key)
    
    print(f"# valid key: {len(valid_key)}, except domain: {str(args.exceptd)}, few shot ratio: {args.fewshot_ratio}")
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    print("Preprocessing Collector data!")
    preprocess_collector(args, valid_key)

    print("Preprocessing Labeler data!")
    preprocess_labeler(args, valid_key)
    print("Done!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default='data')
    parser.add_argument("--collector_file_name", type=str, default='collector_train_data.json')
    parser.add_argument("--labeler_file_name", type=str, default='labeler_train_data.json')
    parser.add_argument("--output_path", type=str, default='data')
    parser.add_argument("--model_name_or_path", type=str, default='roberta-base')
    parser.add_argument("--exceptd", type=str, default=None)
    parser.add_argument("--fewshot_ratio", type=float, default=0.)
    args = parser.parse_args()
    
    if args.exceptd and args.exceptd not in EXPERIMENT_DOMAINS:
        raise Exception(f"The exceptd should be one of {EXPERIMENT_DOMAINS}")
    
    main(args)
