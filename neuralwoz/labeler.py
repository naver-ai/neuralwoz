import torch
from itertools import chain
import re
from transformers import RobertaTokenizer, RobertaForMultipleChoice

from synthesize.override_utils import DOMAIN
from utils.constants import (
    BOS,
    EOS,
    SLOT,
    USER,
    SYS,
    NONE,
    PAD,
    UNK,
    SPECIAL_TOKENS,
    BOOLEAN_TYPE,
    DOMAIN_DESC,
)
from utils.data_utils import pad_ids, pad_id_of_matrix, make_dst_target, split_slot


class Labeler:
    def __init__(self, model_name_or_path, slot_desc, device, unk="<unk>"):
        self.model = RobertaForMultipleChoice.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.slot_desc = slot_desc
        self.device = device
        self.unk = unk

    def labeling_dst(self, did, dialog, goal_set, na_threshold=0.0):
        dialog = [d.lower() for d in dialog]
        x = DSTLabelingToDialogue(did, dialog, goal_set, self.tokenizer, unk=self.unk)
        x.processing(self.tokenizer, self.slot_desc)

        outputs = {}
        for tid in range(x.n_turn):
            input_ids, target_masks, did = x.get_instance_by_turn(tid)
            input_ids = input_ids.to(self.device)
            target_masks = target_masks.to(self.device)
            with torch.no_grad():
                input_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
                o = self.model(input_ids, attention_mask=input_mask)
                pred_index = self.output_threshold(o[0], target_masks, na_threshold)
                pred, domain = x.recover(pred_index)
            outputs[did] = [pred, domain]
        dst_result = self.postprocessing_dst(outputs, dialog, did)
        return dst_result

    def output_threshold(self, logits, masks, na_threshold):
        logits = logits.masked_fill(masks.ne(1), float("-inf"))
        probabilites = torch.softmax(logits, -1)
        probs, indices = probabilites.sort(descending=True)
        max_p, max_i = probs[:, 0], indices[:, 0]
        result = []
        for i, (p, idx) in enumerate(zip(max_p.cpu().tolist(), max_i.cpu().tolist())):
            if idx != 1 and p < na_threshold:  # not-none
                result.append(1)
            else:
                result.append(idx)
        return result

    def postprocessing_dst(self, outputs, dialog, did):
        result = {"dialogue_idx": did, "domains": [], "dialogue": []}
        dialog.insert(0, "")
        domains = []
        idx = 0
        for k, v in outputs.items():
            did, tid = k.split("_")
            base = 0 if tid == "0" else (int(tid) - 1) * 2 + 2
            sys, user = dialog[base : int(tid) * 2 + 2]
            if PAD in user:
                continue

            turn = {}
            turn["system_transcript"] = sys
            turn["transcript"] = user
            turn["system_acts"] = []
            turn["domain"] = v[1]
            turn["turn_idx"] = idx
            turn["turn_label"] = []
            turn["belief_state"] = []
            if v[1] not in domains:
                domains.append(v[1])
            for vv in v[0]:
                dom, slot, value = split_slot(vv)
                turn["belief_state"].append(
                    {"slots": [[dom + "-" + slot, value]], "act": "inform"}
                )
            result["dialogue"].append(turn)
            idx += 1
        result["domains"] = domains
        return result


def get_instance_from_multiwoz(data, idx, slot_meta, tokenizer, slot_desc):
    logs = []
    labels = []
    did = data[idx]["dialogue_idx"]
    for i, log in enumerate(data[idx]["dialogue"]):
        if i > 0:
            logs.append(log["system_transcript"])
        logs.append(log["transcript"])

        label = make_dst_target(log["belief_state"], slot_meta)

        for l in label:
            if l not in labels:
                labels.append(l)

    x = DSTLabelingToDialogue(did, logs, labels, tokenizer)
    x.processing(tokenizer, slot_desc)
    return x


class DSTLabelingToDialogue:
    def __init__(self, did, logs, goal_set, tokenizer, max_seq_length=512, unk="<unk>"):
        self.did = did
        self.logs = logs
        self.goal_set = goal_set
        self.domains = self.get_domains()
        self.n_turn = round(len(logs) / 2)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.slot_token_id = tokenizer.encode(SLOT, add_special_tokens=False)[0]
        self.max_seq_length = max_seq_length
        self.unk = unk

    def get_domains(self):
        domains = []
        for s in self.goal_set:
            dom, _, _ = split_slot(s)
            if dom not in domains:
                domains.append(dom)
        return domains

    def processing(self, tokenizer, slot_desc):
        tokenized = []
        for i, log in enumerate(self.logs):
            if i % 2 == 0:
                role = USER
            else:
                role = SYS
            tokenized.append(
                tokenizer.convert_tokens_to_ids([role] + tokenizer.tokenize(log))
            )
        self.tokenized = tokenized

        descs = []
        choices = []
        slots = []
        values = []
        masks = []
        slot_type_check = {}
        max_value = 2
        for i, g in enumerate(self.goal_set):
            temp = []
            dom, slot, value = split_slot(g)

            slot_type = dom + "-" + slot
            if slot_type_check.get(slot_type) is not None:
                idx = slot_type_check[slot_type]
                values[idx].append(value)
                choices[idx].append(
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value))
                )
                masks[idx].append(1.0)
                if len(choices[idx]) > max_value:
                    max_value = len(choices[idx])
                continue
            desc_idx = -1 if slot_type in BOOLEAN_TYPE else 0
            desc = slot_desc.get(slot_type)[desc_idx]
            descs.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(desc)))
            temp.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value)))
            temp.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(NONE)))
            slot_type_check[slot_type] = len(values)
            choices.append(temp)
            slots.append(slot_type)
            values.append([value, NONE])
            masks.append([1.0, 1.0])

        # DOMAIN
        desc = DOMAIN_DESC[0]
        descs.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(desc)))
        temp = [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dom))
            for dom in self.domains
        ]
        choices.append(temp)
        slots.append("domain")
        values.append(self.domains)
        masks.append([1.0] * len(self.domains))
        if len(self.domains) > max_value:
            max_value = len(self.domains)

        self.descs = descs
        self.choices = choices
        self.slots = slots
        self.values = values
        self.masks = masks
        for i, c in enumerate(self.choices):
            if len(c) == max_value:
                continue
            gap = max_value - len(c)
            for _ in range(gap):
                self.choices[i].append(
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.unk))
                )
                self.values[i].append(self.unk)
                self.masks[i].append(0.0)

    def get_instance_by_turn(self, turn):
        context = (
            [self.bos_token_id]
            + list(chain(*self.tokenized[: turn * 2 + 1]))
            + [self.eos_token_id]
        )
        input_ids = []
        for desc, choice in zip(self.descs, self.choices):
            input_id = []
            for c in choice:
                i = context + desc + [self.eos_token_id] + c + [self.eos_token_id]
                if len(i) > self.max_seq_length:
                    gap = len(i) - self.max_seq_length
                    i = [self.bos_token_id] + i[gap + 1 :]
                input_id.append(i)

            input_id = pad_ids(input_id, self.pad_token_id)
            input_ids.append(torch.LongTensor(input_id))
        return (
            pad_id_of_matrix(input_ids, self.pad_token_id),
            torch.FloatTensor(self.masks),
            self.did + "_" + str(turn),
        )

    def recover(self, preds):
        result = []
        domain = None
        for p, s, v in zip(preds, self.slots, self.values):
            value = v[p]
            if s == "domain":
                domain = value
                continue

            if value != NONE:
                result.append(s + "-" + value)
        return result, domain


class LabelingToDialogue:
    def __init__(
        self, did, logs, candidates, tokenizer, max_seq_length=512, unk="<unk>"
    ):
        self.did = did
        self.logs = logs
        self.candidates = candidates
        self.n_turn = round(len(logs) / 2)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.slot_token_id = tokenizer.encode(SLOT, add_special_tokens=False)[0]
        self.max_seq_length = max_seq_length
        self.unk = unk

    def processing(self, tokenizer, desc):
        tokenized = []
        for i, log in enumerate(self.logs):
            if i % 2 == 0:
                role = USER
            else:
                role = SYS
            tokenized.append(
                tokenizer.convert_tokens_to_ids([role] + tokenizer.tokenize(log))
            )
        self.tokenized = tokenized
        self.desc = tokenizer.encode(desc[0], add_special_tokens=False)

        choices = []
        for cand in self.candidates:
            c = tokenizer.encode(cand, add_special_tokens=False)
            choices.append(c)
        self.choices = choices

    def get_instance_by_turn(self, turn):
        context = (
            [self.bos_token_id]
            + list(chain(*self.tokenized[: turn * 2 + 1]))
            + [self.eos_token_id]
        )
        input_ids = []
        for c in self.choices:
            i = context + self.desc + [self.eos_token_id] + c + [self.eos_token_id]
            input_ids.append(i)

        input_ids = torch.LongTensor(pad_ids(input_ids, self.pad_token_id)).unsqueeze(0)
        return input_ids, self.did + "_" + str(turn)

    def recover(self, prediction):
        return self.candidates[prediction]


def output_threshold(logits, masks, na_threshold=0.5):
    logits = logits.masked_fill(masks.ne(1), float("-inf"))
    probabilites = torch.softmax(logits, -1)
    probs, indices = probabilites.sort(descending=True)
    max_p, max_i = probs[:, 0], indices[:, 0]
    original, result = [], []
    for i, (p, idx) in enumerate(zip(max_p.cpu().tolist(), max_i.cpu().tolist())):
        if idx != 1 and p < na_threshold:  # not-none
            result.append(1)
        else:
            result.append(idx)
        original.append(idx)
    return original, result


def heuristic_domain_annotation(data, slot_meta):
    def domain_check(utter):
        for k, v in DOMAIN.items():
            c = [k] + v
            check = re.search("(%s)" % "|".join(c), utter)
            if check:
                return k
        return None

    for d_idx in range(len(data)):
        last_domain = False
        domains = []
        for i, d in enumerate(data[d_idx]["dialogue"]):
            l = make_dst_target(d["belief_state"], slot_meta)
            doms = [split_slot(ll)[0] for ll in l]
            doms = [d for d in doms if d not in domains]
            check = domain_check(d["transcript"])
            if doms:
                last_domain = doms[0]
                domains.append(doms[0])
                data[d_idx]["dialogue"][i]["domain"] = doms[0]
            elif check:
                last_domain = check
                if check not in domains:
                    domains.append(check)
                data[d_idx]["dialogue"][i]["domain"] = check
            else:
                data[d_idx]["dialogue"][i]["domain"] = last_domain
            data[d_idx]["dialogue"][i]["turn_idx"] = i
        data[d_idx]["domains"] = domains
    return data
