"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import torch
import json
import os
import wget
from .fix_label import fix_general_label_error
from .constants import *


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))
    
    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]
    return arrays


def pad_id_of_matrix(arrays, padding, max_length=-1, left=False):
    if max_length < 0:
        max_length = max([array.size(-1) for array in arrays])
    
    new_arrays = []
    for i, array in enumerate(arrays):
        n, l = array.size()
        pad = torch.zeros(n, (max_length - l))
        pad[:,:,] = padding
        pad = pad.long()
        m = torch.cat([array , pad], -1)
        new_arrays.append(m.unsqueeze(0))

    return torch.cat(new_arrays, 0)


def make_slot_meta(path):
    ontology = json.load(open(path, "r", encoding="utf-8"))
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split("-")
        if d not in EXPERIMENT_DOMAINS:
            continue
        if "price" in s or "leave" in s or "arrive" in s:
            s = s.replace(" ", "")
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append("-".join([d, s]))
        o = ontology[k]
        o = ["dontcare" if v == "do n't care" else v for v in o]
        change[meta[-1]] = o
    return sorted(meta), change


def make_dst_target(bstate, slot_meta):
    target = []
    fixed_bstate = fix_general_label_error(bstate, False, slot_meta)
    for k, v in fixed_bstate.items():
        if v != "none":
            target.append("%s-%s" % (k, v))
    return target


def split_slot(dom_slot_value):
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()
    return dom, slot, value


def get_domains_from_goal(g):
    domains = []
    for gg in g:
        dom, _, _ = split_slot(gg)
        if dom not in domains:
            domains.append(dom)
    return domains


def download_checkpoint(model_name_or_path):
    """
    Manually download model checkpoints of 'dsksd/roberta-base-dream' from huggingface.models
    since transformers==2.11.0 could not support the download.
    """
    if os.path.exists(model_name_or_path):
        return

    os.makedirs(model_name_or_path)
    
    files = [
        f'https://huggingface.co/dsksd/{model_name_or_path}/resolve/main/config.json',
        f'https://huggingface.co/dsksd/{model_name_or_path}/resolve/main/pytorch_model.bin',
        f'https://huggingface.co/dsksd/{model_name_or_path}/resolve/main/merges.txt',
        f'https://huggingface.co/dsksd/{model_name_or_path}/resolve/main/vocab.json'
    ]
    for file in files:
        wget.download(file, out=model_name_or_path)
