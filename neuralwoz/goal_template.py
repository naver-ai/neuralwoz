"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import json
import random
from copy import deepcopy
import re
import os
import numpy as np
from collections import Counter, defaultdict
from synthesize.override_utils import override_existing_goal
from synthesize.value import VALUE_META, coref_exp, BOOLEAN_EXP
from synthesize.goal import SCHEMAS, GoalProfiler
from utils.database import Database
from utils.data_utils import make_slot_meta, split_slot
from utils.collector_utils import preprocess_goal, APIInstance, CollectorInstance
from utils.collector_utils import preprocess_collector_instance
from collector import construct_state_candidate_from_multiwoz


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class GoalTemplate:
    def __init__(
        self, messages, slot_scheme, domain_transitions, is_booked, key="test"
    ):
        self.messages = messages
        self.slot_scheme = slot_scheme
        self.domain_transitions = domain_transitions
        self.is_booked = is_booked
        self.key = key

    @classmethod
    def from_json(cls, file):
        file = json.load(open(file))
        obj = cls(
            file["messages"],
            file["slot_scheme"],
            file["domain_transitions"],
            file["is_booked"],
            file["key"],
        )
        return obj


class GoalSampler:
    def __init__(self, args, rng, include_missing_dontcare=True):
        slot_meta, ontology = make_slot_meta(os.path.join(args.dataset_dir, "ontology.json"))
        self.slot_meta = slot_meta
        self.ontology = ontology
        
        data = json.load(open(os.path.join(args.dataset_dir, args.target_data)))
        self.goal_profiler = GoalProfiler(data)
        self.db = Database(args.dataset_dir)
        self.init_frequency()
        self.ontology["restaurant-book day"] = self.ontology["restaurant-book day"][:-3]
        self.rng = rng
        self.include_missing_dontcare = include_missing_dontcare

    def init_frequency(self):
        frequency = defaultdict(list)
        for key, value in self.db.db.items():
            frequency[key] = [1] * len(value)
        self.frequency = frequency

    def db_sampling(self, domain):
        f = [1 / i for i in self.frequency[domain]]
        p = softmax(f)
        i = np.random.choice(list(range(len(self.db.db[domain]))), p=np.array(p))
        self.frequency[domain][i] += 1
        return self.db.db[domain][i]

    def sampling_template(self, key):
        data = self.goal_profiler.data[key]
        x = preprocess_collector_instance(key, data, self.db)
        x = construct_state_candidate_from_multiwoz(x, key, self.goal_profiler)
        for a in self.goal_profiler.goal_check[key]:
            if a not in x.state_candidate:
                x.state_candidate.append(a)

        messages, slot_scheme, domains, is_booked = override_existing_goal(
            data["goal"]["message"], x.state_candidate, self.slot_meta
        )
        template = GoalTemplate(messages, slot_scheme, domains, is_booked, key)
        return template

    def state_align(self, template):
        instances = []
        state_candidate = []
        cloned_slots = deepcopy(template.slot_scheme)
        
        # DB instance sampling
        for domain in template.domain_transitions:
            coref_slots, dontcare = get_domain_specific_slot(
                domain, cloned_slots, self.include_missing_dontcare
            )

            for d in dontcare:
                state_candidate.append(f"{domain}-{d}-dontcare")

            if not self.db.db.get(domain):
                db_inst = {}
            elif coref_slots and instances:
                pool = coref_instance_search(coref_slots, instances, self.db)
                if pool:
                    db_inst = self.rng.choice(pool)
                else:
                    db_inst = self.db_sampling(domain)  # self.rng.choice(self.db.db[domain])
            else:
                db_inst = self.db_sampling(domain)  # self.rng.choice(self.db.db[domain])
            instances.append(db_inst)
            
            # construct State Candidate for labeling
            for s in SCHEMAS[domain].informable + [SCHEMAS[domain].entity]:
                v = db_inst.get(s)
                if not v:
                    continue
                slot_value = f"{domain}-{s}-{v}"
                if slot_value not in state_candidate:
                    state_candidate.append(slot_value)
                    try:
                        cloned_slots.pop(cloned_slots.index(f"{domain}-{s}"))
                    except:
                        pass
            
            # Check coreference value from the slot scheme
            while cloned_slots:
                slot = cloned_slots[0]
                if "=>" in slot:
                    leaf, ana = slot.split("=>")
                    if domain not in leaf:
                        break
                    cloned_slots.pop(cloned_slots.index(slot))
                    anaphora = [g for g in state_candidate if ana in g][0]
                    _, _, anaphora = split_slot(anaphora)
                    slot = leaf
                else:
                    anaphora = None

                dom, slot = slot.split("-")

                if dom != domain:
                    break

                if anaphora:
                    slot_value = f"{dom}-{slot}-{anaphora}"

                else:
                    pool = self.ontology[f"{dom}-{slot}"]
                    pool = [
                        p for p in pool if p not in ["dontcare", "none", "not given"]
                    ]
                    slot_value = "%s-%s-%s" % (
                        dom,
                        slot,
                        self.rng.choice(pool).strip(". "),
                    )

                if slot_value not in state_candidate:
                    state_candidate.append(slot_value)

                try:
                    cloned_slots.pop(cloned_slots.index(f"{domain}-{slot}"))
                except:
                    pass
        
        # multiple slot-types in a scenario 
        # The value of the type should be changed
        # We assume the last value of the type is finally selected
        overlap = ["-".join(split_slot(g)[:2]) for g in state_candidate]
        final = {}
        for k, v in Counter(overlap).items():
            if v == 1:
                continue

            domain, slot = k.split("-")
            if slot in SCHEMAS[domain].informable:
                idx = template.domain_transitions.index(domain)
                v = instances[idx][slot]
                fi = state_candidate.index(f"{k}-{v}")
                final[k] = fi
        
        # inject boolean type expression
        m = "&&&".join(template.messages)
        for idx, g in enumerate(state_candidate):
            dom, slot, value = split_slot(g)
            dom_slot = f"{dom}-{slot}"
            if BOOLEAN_EXP.get(dom_slot):
                if value == "dontcare":
                    continue
                # TODO
                exp = BOOLEAN_EXP[dom_slot][value][0]
                m = m.replace("should <%s>" % dom_slot, exp, 1)
                m = m.replace("<%s>" % dom_slot, exp, 1)
            else:
                if final.get(dom_slot) == idx:
                    m = m.replace(f"<{dom_slot}>", "[MARKING]", 1)
                m = m.replace(f"<{dom_slot}>", value, 1)
                if final.get(dom_slot) == idx:
                    m = m.replace("[MARKING]", f"<{dom_slot}>", 1)
        
        # coref slot switching
        c = re.search(r"[<][\w\s]+[-][\w\s]+=>[\w\s]+[-][\w\s]+[>]", m)
        while c:
            span = c.group()
            _, x = span.split("=>")
            exp = coref_exp(x.strip("<>")).replace(" book ", " ")
            m = m.replace(span, exp)
            c = re.search(r"[<][\w\s]+[-][\w\s]+=>[\w\s]+[-][\w\s]+[>]", m)

        goal = preprocess_goal(m.split("&&&"))
        
        # Make API Instance
        apis = []
        for i, domain in enumerate(template.domain_transitions):
            if template.is_booked.get(domain):
                if domain == "taxi":  # make virtual instance for taxi domain
                    car_type = VALUE_META["carType"]()
                    phone = VALUE_META["phone"]()
                    color = VALUE_META["color"]()
                    instances[i]["phone"] = phone
                    instances[i]["car type"] = car_type
                    instances[i]["color"] = color
                else:
                    ref = VALUE_META["reference"]()
                    instances[i]["reference number"] = ref
            apis.append(APIInstance(domain, instances[i]))
        x = CollectorInstance(template.key, goal, apis)
        x.state_candidate = state_candidate
        return x


def get_domain_specific_slot(domain, slots, include_missing_dontcare=True):
    coref = []
    types = []
    for slot in slots:
        if not slot.startswith(domain):
            continue

        if "=>" in slot:
            coref.append(slot)
            slot, _ = slot.split("=>")

        dom, slot = slot.split("-")
        types.append(slot)

    if not include_missing_dontcare:
        return coref, []

    if SCHEMAS[domain].entity in types:
        dontcare = []
    else:
        dontcare = [
            s
            for s in SCHEMAS[domain].informable
            if s not in types and s != SCHEMAS[domain].entity
        ]
    return coref, list(set(dontcare))


def coref_instance_search(coref, instances, database):
    info = {}
    for c in coref:
        from_, to = c.split("=>")
        dom, slot = from_.split("-")
        to_dom, _ = to.split("-")

        if slot not in SCHEMAS[dom].informable:
            continue
        value = instances[-1][slot]
        info[slot] = value
    return database.search_info(dom, info)


def goal_to_dict(goal):
    dic = {}
    for g in goal:
        d, s, v = split_slot(g)
        dic[d + "-" + s] = v
    return dic


def make_consistency(slot_scheme, state_candidate):
    conds = {}
    gd = goal_to_dict(state_candidate)
    for s in slot_scheme:
        if "=>" in s:
            ans, cs = s.split("=>")
            conds[ans] = gd[cs]
        else:
            conds[s] = gd[s]
    return conds
