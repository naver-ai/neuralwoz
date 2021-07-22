"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import re
import torch
from synthesize.goal import SCHEMAS
from models import BartTokenizer, BartForConditionalGeneration
from utils.constants import NON_DB_DOMAIN, ENTITY_DOMAIN, NOT_DOMAIN
from utils.database import DOMAIN_PK
from utils.data_utils import pad_ids, SPECIAL_TOKENS, split_slot


class Collector:
    def __init__(self, model_name_or_path, device, input_max_seq_length=768):
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.device = device
        self.input_max_seq_length = input_max_seq_length

    def sampling_dialogue(
        self,
        instances,
        num_beams=4,
        top_k=30,
        top_p=0.0,
        temperature=None,
        do_sample=True,
    ):

        if isinstance(instances, list):
            for i in instances:
                i.processing(self.tokenizer)
            input_ids = torch.LongTensor(
                pad_ids(
                    [i.input_id[: self.input_max_seq_length] for i in instances],
                    self.tokenizer.pad_token_id,
                )
            )
        else:
            instances.processing(self.tokenizer)
            input_ids = torch.LongTensor(
                [instances.input_id[: self.input_max_seq_length]]
            )

        input_ids = input_ids.to(self.device)
        self.model.input_ids = input_ids
        outputs = self.model.generate(
            input_ids,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            max_length=512,
            temperature=temperature,
            early_stopping=True,
        )
        dialogs = self.postprocessing(outputs)
        return dialogs

    def postprocessing(self, outputs):
        dialogs = []
        for output in outputs.cpu().tolist():
            result = self.tokenizer.decode(output)
            split_result = re.sub(
                r"[<](user|sys)[>]", "</s>", result.strip("<s>")
            ).split("</s>")
            split_result = [s for s in split_result if s.strip()]
            dialog = []
            for i, r in enumerate(split_result):
                r = re.sub(r"\s+", " ", r.replace("\n", " ").replace("</", "").strip())
                dialog.append(r)
            dialogs.append(dialog)
        return dialogs


def construct_state_candidate_from_multiwoz(x, key, goal_aligner):
    state_candidate = []
    slot_check = []
    for g in goal_aligner.goals[key]:
        dom, slot, value = split_slot(g)
        state_candidate.append(g)
        slot_check.append(f"{dom}-{slot}")

    for a in x.apis:
        if a.domain in NOT_DOMAIN:
            continue

        informables = SCHEMAS[a.domain].informable
        pk_slot = None
        if a.domain in ENTITY_DOMAIN:
            pk_slot = DOMAIN_PK[a.domain]
            informables.append(pk_slot)

        for inf in informables:
            v = a.info.get(inf)
            if not v:
                continue

            new = f"{a.domain}-{inf}-{v}"
            if new in state_candidate:
                continue

            state_candidate.append(new)
            if pk_slot in slot_check:
                slot_check.append(f"{a.domain}-{inf}")

            if inf != pk_slot and (f"{a.domain}-{inf}" not in slot_check):
                new = f"{a.domain}-{inf}-dontcare"
                if new not in state_candidate:
                    state_candidate.append(new)
    x.state_candidate = state_candidate
    return x
