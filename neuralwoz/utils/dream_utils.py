import json
import re
import torch
from torch.utils.data import Dataset
from .constants import USER, SYS, NONE, BOS, EOS, UNK
from .data_utils import pad_ids, pad_id_of_matrix


def load_dream_data(data_path, tokenizer):
    data = json.load(open(data_path, "r", encoding="utf-8"))

    instances = []
    for i, d in enumerate(data):
        dialogs = []
        for t in d[0]:
            if t.startswith("M"):
                role = SYS
            else:
                role = USER
            turn = re.sub(r"^(M|W|F)[:]", "", t)
            turn = role + turn
            dialogs.append(turn)
        for qas in d[1]:
            question = qas["question"]
            choice = qas["choice"]
            label = qas["choice"].index(qas["answer"])
            instance = DreamInstance(dialogs, str(i), question, choice, label, tokenizer.pad_token_id)
            instance.processing(tokenizer)
            instances.append(instance)
    return instances

            
class DreamInstance:
    def __init__(self, logs, did,
                 description, choices,
                 label=None, pad_token_id=1,
                 task_name='dream',
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
        ys = [BOS] + tokenizer.tokenize(' '.join(self.logs)) + [EOS] + tokenizer.tokenize(self.description)

        inputs = []
        target_mask = []
        for idx, choice in enumerate(self.choices):
            tokenized_choice = [EOS] + tokenizer.tokenize(choice) + [EOS]
            input_id = tokenizer.convert_tokens_to_ids(ys + tokenized_choice)
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
            self.target_id = self.label
        else:
            self.target_id = None
            self.not_none_mask = None

    def to_dict(self):
        dic = {}
        dic['did'] = self.did
        dic['logs'] = self.logs
        dic['label'] = self.label
        dic['choices'] = self.choices
        dic['description'] = self.description
        dic['input_id'] = self.input_id
        dic['target_mask'] = self.target_mask
        dic['target_id'] = self.target_id
        dic['not_none_mask'] = self.not_none_mask
        return dic


class Dreamdataset(Dataset):
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
        input_ids = pad_id_of_matrix(input_ids, self.pad_id)
        input_mask = input_ids.ne(self.pad_id).float()
        target_ids = torch.LongTensor(target_ids)
        return input_ids, input_mask, target_ids, []
