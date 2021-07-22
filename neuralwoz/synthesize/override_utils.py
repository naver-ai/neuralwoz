"""
NeuralWOZ
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu
import re


BOOLEAN = {
    "hotel-parking": {
        "yes": [r"(include|have) (free )?parking"],
        "no": [r"doesn't need to (include|have) (free )?parking"],
    },
    "hotel-internet": {
        "yes": [r"(include|have) (free )?(wifi|internet)"],
        "no": [r"doesn't need to (include|have) (free )?internet"],
    },
}


DOMAIN = {
    "train": [],
    "taxi": [],
    "restaurant": ["places? to dine", "particular restaurant"],
    "attraction": ["places? to go", "particular attraction"],
    "hotel": ["places? to stay", "particular hotel"],
}


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


def parse_time(sent):
    result = None
    check = re.search(r"<span class='emphasis'>\d+[:]\d+</span>", sent)
    if check:
        result = check.group()
    return result


def get_activate_domain(sent):
    activate = None
    for domain, synoms in DOMAIN.items():
        pat = "<span class='emphasis'>(" + "|".join([domain] + synoms) + ")</span>"
        c = re.search(pat, sent)
        if c:
            activate = domain
            break
    return activate


def tokenize(text):
    text = text.replace("book ", "")
    text = text.replace("pricerange", "price range").replace("-", " ")
    return text.split()


def get_coref_slot_from_domain(domain, target_slot, slot_meta):
    slots = [s for s in slot_meta if domain in s]
    maximum = 0
    target = None
    for slot in slot_meta:
        if domain not in slot:
            continue
        slot = slot.replace(domain + "-", "")
        score = sentence_bleu([tokenize(target_slot)], tokenize(slot), [1.0])
        if score > maximum:
            maximum = score
            target = slot
    return target


def override_existing_goal(template, goal, slot_meta):
    new_template = []
    goal = [split_slot(g) for g in goal]
    goal_scheme = []
    activate_domain = None
    commute = False
    domain_transition = []
    is_booked = {}
    for sent in template:
        dc = get_activate_domain(sent)
        if dc and activate_domain != dc:
            activate_domain = dc
            if dc not in domain_transition:
                domain_transition.append(dc)

        if not domain_transition:
            continue

        for d, k, v in goal:
            slot = d + "-" + k
            if v in ["dontcare", "none"]:
                continue

            if v == "free":
                v = "yes"

            if d != domain_transition[-1]:
                continue

            if BOOLEAN.get(slot):
                patterns = BOOLEAN[slot]
                pattern = patterns.get(v)
                v = pattern[0]

            check = re.search(
                r"<span class='emphasis'>(\d|\w|\s)*%s(\d|\w|\s)*</span>" % v, sent
            )
            if not check:
                continue

            start, end = check.span()
            span = sent[start:end]
            span = (
                re.sub(v, "<%s>" % slot, span)
                .replace("<span class='emphasis'>", "")
                .replace("</span>", "")
            )
            goal_scheme.append(slot)
            sent = sent[:start] + span + sent[end:]

        # Coref check
        check = re.search(
            "<span class='emphasis'>(\d|\w|\s)*same(\d|\w|\s)*</span>", sent
        )
        while check:
            start, end = check.span()
            span = (
                sent[start:end]
                .replace("<span class='emphasis'>", "")
                .replace("</span>", "")
            )
            anaphora = [
                s
                for s in slot_meta
                if any([True if d in s else False for d in domain_transition[:-1]])
            ]
            anaphora = [s for s in anaphora if s not in BOOLEAN]
            scores = sorted(
                [
                    [g, sentence_bleu([tokenize(g)], span.split(), [1.0])]
                    for g in anaphora
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            coref_key = scores[0][0]
            coref_domain, slot = coref_key.split("-")
            if len(domain_transition) > 1:
                target_slot = get_coref_slot_from_domain(
                    domain_transition[-1], slot, slot_meta
                )
                if not target_slot:
                    target_slot = slot
                coref_key = "%s-%s=>%s-%s" % (
                    domain_transition[-1],
                    target_slot,
                    coref_domain,
                    slot,
                )

            goal_scheme.append(coref_key)
            sent = sent[:start] + "<%s>" % coref_key + sent[end:]
            check = re.search(
                "<span class='emphasis'>(\d|\w|\s)*same(\d|\w|\s)*</span>", sent
            )

        target_doms = [d for d in domain_transition if d not in ["taxi", "train"]]
        if "commute" in sent and len(target_doms) > 1:
            depart = target_doms[0] + "-name"
            destination = target_doms[1] + "-name"
            goal_scheme.append("taxi-departure=>" + depart)
            goal_scheme.append("taxi-destination=>" + destination)
            commute = True

        # realign
        if commute and re.search("(leave|arrive)", sent):
            d = get_activate_domain(sent)
            if not d:
                continue
            check = re.search("(leave|arrive)", sent).group()
            departure = [
                i for i, g in enumerate(goal_scheme) if "taxi-departure=>" in g
            ][0]
            destination = [
                i for i, g in enumerate(goal_scheme) if "taxi-destination=>" in g
            ][0]
            departure_anaphora = goal_scheme[departure].split("=>")[-1]
            destination_anaphora = goal_scheme[destination].split("=>")[-1]

            leaveat = [g for i, g in enumerate(goal_scheme) if "taxi-leaveat" == g]
            arriveby = [g for i, g in enumerate(goal_scheme) if "taxi-arriveby" == g]

            reorder = False
            parsed_time = parse_time(sent)
            if check == "leave" and parsed_time and not leaveat:
                sent = re.sub(parsed_time, "<taxi-leaveat>", sent)
                goal_scheme.append("taxi-leaveat")

            if check == "arrive" and parsed_time and not arriveby:
                sent = re.sub(parsed_time, "<taxi-arriveby>", sent)
                goal_scheme.append("taxi-arriveby")

            # departure
            if check == "leave" and d + "-name" != departure_anaphora:
                reorder = True

            if check == "arrive" and d + "-name" != destination_anaphora:
                reorder = True

            if reorder:
                goal_scheme[departure] = "taxi-departure=>%s" % destination_anaphora
                goal_scheme[destination] = "taxi-destination=>%s" % departure_anaphora

        if re.search(
            "<span class='emphasis'>(\d|\w|\s)*(reference|contact) number(\d|\w|\s)*</span>",
            sent,
        ):
            is_booked[domain_transition[-1]] = True

        new_template.append(sent)
    return new_template, goal_scheme, domain_transition, is_booked
