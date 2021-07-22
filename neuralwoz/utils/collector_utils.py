from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import re
from .data_utils import pad_ids
from .constants import *
from .database import DOMAIN_PK


class Collectordataset(Dataset):
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
        input_ids = [b.input_id for b in batch]
        target_ids = [b.target_id for b in batch]
        input_ids = torch.LongTensor(pad_ids(input_ids, self.pad_id))
        target_ids = torch.LongTensor(pad_ids(target_ids, -100))
        input_mask = input_ids.ne(self.pad_id).float()
        return input_ids, input_mask, target_ids


class APIInstance:
    def __init__(self, domain, info, pk=None, ref=None):
        self.domain = domain
        self.pk = pk
        self.ref = ref
        self.info = info
        self.make_instance()
    
    def __str__(self):
        return self.flatten
        
    def make_instance(self):
        flatten = [DOMAIN, self.domain]
        for k, v in self.info.items():
            if k in ["introduction", "openhours"]:
                continue

            if isinstance(v, str):
                flatten.append(SLOT)
                flatten.append('%s-%s' % (k, v))
        if self.ref is not None:
            flatten.append(SLOT)
            flatten.append('%s-%s' % ('reference number', self.ref))
        self.flatten = ' '.join(flatten)
        
        
class CollectorInstance:
    def __init__(self, did, goal, apis, logs=None):
        self.did = did
        if isinstance(goal, str):
            self.goal = goal
        else:
            self.goal = ' '.join(goal)
        self.apis = apis
        self.logs = logs
        
    def processing(self, tokenizer):
        x = tokenizer.tokenize(self.goal)
        x = [BOS] + x + [EOS]
        input_id = tokenizer.convert_tokens_to_ids(x)
        apis = ' '.join([str(a) for a in self.apis])
        apis = tokenizer.tokenize(apis)
        api_id = tokenizer.convert_tokens_to_ids(apis)
        self.input_id = input_id + api_id
        
        if not self.logs:
            self.target_id = []
            return
        
        ys = []
        for i, y in enumerate(self.logs):
            if i % 2 == 0:
                role = USER
            else:
                role = SYS
            y = [role] + tokenizer.tokenize(y)
            ys.extend(y)
        ys = [BOS] + ys + [EOS]
        target_id = tokenizer.convert_tokens_to_ids(ys)
        self.target_id = target_id
        
    def to_dict(self):
        dic = {}
        dic['did'] = self.did
        dic['goal'] = self.goal
        dic['apis'] = [str(a) for a in self.apis]
        dic['logs'] = self.logs
        dic['input_id'] = self.input_id
        dic['target_id'] = self.target_id
        return dic
    
    @classmethod
    def from_dict(cls, dic):
        obj = cls(dic['did'], dic['goal'], dic['apis'], dic['logs'])
        obj.input_id = dic['input_id']
        obj.target_id = dic['target_id']
        return obj

    
def preprocess_goal(messages):
    if isinstance(messages, str):
        return [messages]

    processed = []
    for m in messages:
        m = m.strip('"')
        m = re.sub(r"[<]span class[=]'emphasis'[>]", '', m)
        m = m.replace('</span>', '')
        if not m.endswith('.'):
            m = m + '.'
        processed.append(m)
    return processed


def preprocess_collector_instance(did, data, database):
    api_info = []
    logs = []
    overlap_check = []
    for log in data['log']:
        logs.append(log['text'])
        for vv in log.get('span_info', []):
            action, slot, value, _, _ = vv
            slot = slot.lower()
            check = re.search(r'(Police|Hospital|Taxi)', action, flags=re.IGNORECASE)
            if check:
                domain = check.group().lower()
                slot = MAPPER.get(slot, slot)
                value = value.lower()
                if '%s-%s' % (domain, value) not in overlap_check:
                    api_info.append([domain, slot, value])
                    overlap_check.append('%s-%s' % (domain, value))

            if re.search(r'(Recommend|Inform)', action, flags=re.IGNORECASE):
                domain = re.sub(r'-(Recommend|Inform)', '', action, flags=re.IGNORECASE).lower().strip()
                if DOMAIN_PK.get(domain) != slot:
                    continue

                value = database.normalize(vv[2])
                if '%s-%s' % (domain, value) not in overlap_check:
                    api_info.append([domain, value, None])
                    overlap_check.append('%s-%s' % (domain, value))

        for k, v in log['metadata'].items():
            if v['book']['booked']:
                booked_info = v['book']['booked'][0]
                try:
                    pv = database.normalize(booked_info[DOMAIN_PK[k]])
                except KeyError:
                    pv = database.normalize(v['semi'][DOMAIN_PK[k]])
                    
                if '%s-%s' % (k, pv) not in overlap_check:
                    if k in ['taxi']:
                        api_info.append([k, DOMAIN_PK[k], pv])
                        api_info.append([k, 'type', booked_info['type']])
                    else:
                        api_info.append([k, pv, booked_info['reference']])
                    overlap_check.append('%s-%s' % (k, pv))
                elif [k, pv, None] in api_info and k not in ['taxi']: # semi
                    idx = api_info.index([k, pv, None])
                    api_info[idx] = [k, pv, booked_info['reference']]

            for kk, vv in v['semi'].items():
                if kk == DOMAIN_PK.get(k) and vv.strip() not in ['', 'not mentioned']:
                    vv = database.normalize(vv)
                    if '%s-%s' % (k, vv) not in overlap_check:
                        api_info.append([k, vv, None])
                        overlap_check.append('%s-%s' % (k, vv))
    checker = []
    results = []
    overlap_check = []
    for dom, pv, ref in api_info:
        if dom in NON_DB_DOMAIN:
            res = {}
            if dom in checker:
                continue

            for dom2, pv, ref in api_info:
                if dom == dom2:
                    res[pv] = ref
            i = APIInstance(dom, res)
            checker.append(dom)
        else:
            res = database.search(dom, pv)
            if not res:
                continue

            if res in overlap_check:
                continue

            i = APIInstance(dom, res, ref=ref)
            overlap_check.append(res)
        results.append(i)
        
    goal = preprocess_goal(data['goal']['message'])
    return CollectorInstance(did, goal, results, logs)


def build_collector_data(data, database, tokenizer, keys=None):
    if not keys:
        keys = list(data.keys())

    instances = []
    nkeys = []
    for key in keys:
        try:
            instance = preprocess_collector_instance(key, data[key], database)
            instance.processing(tokenizer)
            instances.append(instance)
        except Exception as e:
            print(f"Some error occurs when processing {key}!!")
            nkeys.append(key)
            continue

    return instances, nkeys


def load_collector_data(data,
                        instance_cls=CollectorInstance,
                        max_seq_length=768):
    instances = []
    for d in data:
        instance = instance_cls.from_dict(d)
        if len(instance.input_id) > max_seq_length or len(instance.target_id) > max_seq_length:
            continue
        instances.append(instance)
    return instances

