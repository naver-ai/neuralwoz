"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import json
from copy import deepcopy
from itertools import chain
import h5py
import torch
from torch.utils.data import Dataset
from .data_utils import make_dst_target, split_slot, pad_ids, pad_id_of_matrix
from .constants import EXPERIMENT_DOMAINS, SYS, USER, NONE, SLOT, BOS, EOS, NONE, UNK, DOMAIN_DESC


def build_labeler_data(path,
                       slot_meta,
                       tokenizer,
                       slot_desc,
                       ontology,
                       rng,
                       n_choice=5,
                       unk="<unk>",
                       valid_key=[]):
    data = json.load(open(path, "r", encoding="utf-8"))
    instances = []
    for d in data:
        did = d["dialogue_idx"]
        if valid_key and did not in valid_key:
            continue

        logs = []
        pool, labels = [], []
        for log in d["dialogue"]:
            label = make_dst_target(log["belief_state"], slot_meta)
            labels.append(label)
            for l in label:
                v = split_slot(l)
                if v not in pool:
                    pool.append(v)
        value_pool = list(set([p[-1] for p in pool]))
        for tid, log in enumerate(d["dialogue"]):
            if log["domain"] not in EXPERIMENT_DOMAINS:
                continue

            turn_log = []
            if tid > 0 and logs:
                turn_log.append([SYS] + tokenizer.tokenize(log["system_transcript"]))
                context = deepcopy(logs[-1])
            else:
                context = []
            turn_log.append([USER] + tokenizer.tokenize(log["transcript"]))
            turn_log = context + turn_log
            logs.append(turn_log)

            turn_log = list(chain(*turn_log))
            label = labels[tid]
            for li, (dom, slot, value) in enumerate(pool):
                if dom not in EXPERIMENT_DOMAINS:
                    continue

                l = "-".join([dom, slot, value])
                if l in label:
                    choices = [value, NONE]
                else:
                    choices = [NONE]
                description = rng.choice(slot_desc[dom + "-" + slot][:-1])
                hard = [c[-1] for i, c in enumerate(pool) if i != li and c[:2] == (dom, slot)]

                if hard:
                    choices.extend(hard)
                n_negatives = n_choice - len(choices)
                rest = sorted(list(set(value_pool) - set(choices)))
                while n_negatives > len(rest):
                    n = rng.choice(ontology.get(dom + "-" + slot, [unk]))
                    if n not in choices:
                        rest.append(n)
                    else:
                        rest.append(unk)

                if n_negatives > 0:
                    negatives = rng.sample(rest, n_negatives)
                    choices.extend(negatives)
                ins = LabelerInstance(turn_log,
                                     f"{did}_{tid}",
                                     description, choices[:n_choice],
                                     0, tokenizer.pad_token_id, "dst")
                ins.processing(tokenizer)
                instances.append(ins)

            # active domain
            description = rng.choice(DOMAIN_DESC)
            rest_domain = sorted(list(set(EXPERIMENT_DOMAINS) - set([log["domain"]])))
            choices = [log["domain"]] + rest_domain
            ins = LabelerInstance(turn_log,
                                 f"{did}_{tid}_domain",
                                 description, choices[:n_choice],
                                 0, tokenizer.pad_token_id, "domain_cls")
            ins.processing(tokenizer)
            instances.append(ins)

    return instances


class LabelerInstance:
    def __init__(self, logs, did,
                 description, choices,
                 label=None, pad_token_id=1,
                 task_name="dst",
                 max_seq_length=512):
        self.did = did
        self.logs = logs
        self.description = description
        self.choices = choices
        self.label = label
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.max_seq_length = max_seq_length
    
    def processing(self, tokenizer):
        ys = [BOS] + self.logs + [EOS] + tokenizer.tokenize(self.description)

        inputs = []
        target_mask = []
        for idx, choice in enumerate(self.choices):
            tokenized_choice = [EOS] + tokenizer.tokenize(choice) + [EOS]
            input_id = tokenizer.convert_tokens_to_ids(ys + tokenized_choice)

            # truncation
            if len(input_id) > self.max_seq_length:
                gap = len(input_id) - self.max_seq_length
                input_id = tokenizer.convert_tokens_to_ids([BOS]) + input_id[gap+1:]
            inputs.append(input_id)
            if choice == UNK:
                target_mask.append(0)
            else:
                target_mask.append(1)

        self.input_id = pad_ids(inputs, self.pad_token_id)
        self.target_mask = target_mask

        if self.label is not None:
            not_none_mask = 0.
            if self.choices[self.label] != NONE:
                not_none_mask = 1.
            self.not_none_mask = not_none_mask
            self.target_id = self.label
        else:
            self.target_id = None
            self.not_none_mask = None

    def to_dict(self):
        dic = {}
        dic["did"] = self.did
        dic["logs"] = self.logs
        dic["label"] = self.label
        dic["choices"] = self.choices
        dic["description"] = self.description
        dic["input_id"] = self.input_id
        dic["target_mask"] = self.target_mask
        dic["target_id"] = self.target_id
        dic["not_none_mask"] = self.not_none_mask
        return dic
    
    @classmethod
    def from_dict(cls, dic):
        obj = cls(dic["logs"], dic["did"], dic["description"], dic["choices"], dic["label"])
        obj.input_id = dic["input_id"]
        obj.target_mask = dic["target_mask"]
        obj.target_id = dic["target_id"]
        obj.not_none_mask = dic["not_none_mask"]
        return obj

    
class Labelerdataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.pad_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.length = len(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        input_ids = [torch.LongTensor(b.input_id) for b in batch]
        target_ids = [b.target_id for b in batch]
        target_mask = [b.target_mask for b in batch]
        input_ids = pad_id_of_matrix(input_ids, self.pad_id)
        target_ids = torch.LongTensor(target_ids)
        input_mask = input_ids.ne(self.pad_id).float()
        target_mask = torch.FloatTensor(target_mask)
        return input_ids, input_mask, target_ids, target_mask


class H5Labelerdataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.h5_file = None
        self.h5_dataset = None
        with h5py.File(self.file_path, "r") as f:
            self.length = len(f["data"])
        self.pad_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
    
    def init_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.h5_dataset = self.h5_file["data"]
    
    def __getitem__(self, idx):
        if not self.h5_dataset:
            self.init_hdf5()

        ins = self.h5_dataset[idx]
        return LabelerInstance.from_dict(json.loads(ins))
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, batch):
        input_ids = [torch.LongTensor(b.input_id) for b in batch]
        target_ids = [b.target_id for b in batch]
        input_ids = pad_id_of_matrix(input_ids, self.pad_id)
        target_ids = torch.LongTensor(target_ids)
        input_mask = input_ids.ne(self.pad_id).float()
        not_none_mask = torch.FloatTensor([b.not_none_mask for b in batch])
        target_mask = [b.target_mask for b in batch]
        target_mask = torch.FloatTensor(target_mask)
        return input_ids, input_mask, target_ids, not_none_mask, target_mask
