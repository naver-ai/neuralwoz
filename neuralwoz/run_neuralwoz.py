from collector import construct_state_candidate_from_multiwoz, Collector
from labeler import Labeler
from utils.constants import SPECIAL_TOKENS
from utils.data_utils import make_slot_meta, make_dst_target
from utils.seed import set_seed
from goal_template import GoalSampler

import torch
from torch.utils.data import DataLoader
import json, os
import random
import argparse
import math
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IntermediateResult:
    def __init__(self, did, dialog, state_candidate):
        self.did = did
        self.dialog = dialog
        self.state_candidate = state_candidate


def make_input_from_goal_template(args, rng):
    print(f"Profiling existing goals in {args.target_data}")
    sampler = GoalSampler(args, rng, args.include_missing_dontcare)
    keys = []
    for key, domains in sampler.goal_profiler.domains.items():

        if args.include_domain and args.include_domain not in domains:
            continue

        if args.only_domain and [args.only_domain] != domains:
            continue

        keys.append(key)

    print(f"Overall goal template keys: {len(keys)}")

    inputs = []
    conflict = []
    while len(inputs) < args.num_dialogues:
        key = rng.choice(keys)
        try:
            template = sampler.sampling_template(key)

            if not template.slot_scheme:
                continue

            x = sampler.state_align(template)
            if not x:
                conflict.append(key)
                continue

            if not len(x.state_candidate):
                conflict.append(key)
                continue

            if re.search(r"[<][\w\s]+[-][\w\s]+[>]", x.goal):
                conflict.append(key)
                continue

        except:
            conflict.append(key)
            continue
        inputs.append(x)

    print(f"# Conflict: {len(conflict)}, # Inputs: {len(inputs)}")
    return inputs


def main(args):
    set_seed(args.seed)
    slot_meta, _ = make_slot_meta(os.path.join(args.dataset_dir, "ontology.json"))
    slot_desc = json.load(
        open(os.path.join(args.dataset_dir, "slot_descriptions.json"))
    )
    rng = random.Random(args.seed)

    inputs = make_input_from_goal_template(args, rng)
    collector = Collector(args.collector_path, device)
    labeler = Labeler(args.labeler_path, slot_desc, device)

    dialogs = []
    n_step = math.ceil(len(inputs) / args.batch_size)
    print(f"Start Collection using {args.collector_path}!")
    temp = 1.0 if not args.temperature else args.temperature
    print(
        "Num Beams: %d, Top k: %d, Top p: %.2f, Temperature: %.2f"
        % (args.num_beams, args.top_k, args.top_p, temp)
    )
    if args.greedy_ratio > 0.0:
        n_greedy_step = math.ceil(n_step * args.greedy_ratio)
    else:
        n_greedy_step = -1
    print("Greedy ratio: %.3f, Greedy step: %d" % (args.greedy_ratio, n_greedy_step))

    for i in range(n_step):
        batch = inputs[i * args.batch_size : (i + 1) * args.batch_size]
        if i < n_greedy_step:
            top_k, top_p, temperature, num_beams = 0, 0.0, None, 1
        else:
            top_k, top_p, temperature, num_beams = (
                args.top_k,
                args.top_p,
                args.temperature,
                args.num_beams,
            )

        dialog = collector.sampling_dialogue(
            batch,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
        )
        for b, d in zip(batch, dialog):
            dialogs.append(IntermediateResult(b.did, d, b.state_candidate))
        print("[%d/%d]" % (i, n_step))

    results = []
    print(f"\nStart Labeling using {args.labeler_path}!")
    for i, instance in enumerate(dialogs):
        try:
            result = labeler.labeling_dst(
                instance.did, instance.dialog, instance.state_candidate, args.na_threshold
            )
        except Exception as e:
            print(instance.did, str(e))
            continue

        results.append(result)
        print("[%d/%d]" % (i, len(dialogs)))

    json.dump(
        results,
        open(os.path.join(args.output_dir, args.output_file_name), "w"),
        indent=2
    )
    print("\nAll done! The synthesized dialogues are saved at %s" % os.path.join(args.output_dir, args.output_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--target_data", type=str, default="collector_dev_data.json")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_file_name", type=str, default="neuralwoz-output.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--collector_path", type=str, required=True, default=None)
    parser.add_argument("--labeler_path", type=str, required=True, default=None)
    parser.add_argument("--num_dialogues", type=int, default=1000)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_domain", type=str, default=None)
    parser.add_argument("--include_domain", type=str, default=None)
    parser.add_argument("--greedy_ratio", type=float, default=0.0)
    parser.add_argument(
        "--include_missing_dontcare", action="store_true", default=False
    )
    parser.add_argument("--na_threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if os.path.exists(os.path.join(args.output_dir, args.output_file_name)):
        print(f"There is already {args.output_file_name} in the {args.output_dir}. Please doule check the output path if you don't want to overwrite!")
    
    main(args)
