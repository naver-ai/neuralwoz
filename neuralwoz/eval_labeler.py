import torch
from utils.data_utils import make_dst_target
from utils.eval_utils import Evaluator, DSTEvaluator
from labeler import get_instance_from_multiwoz, output_threshold


def dst_evaluation(
    model,
    data,
    tokenizer,
    slot_desc,
    slot_meta,
    verbose=True,
    threshold=0.5,
    specific_domain=None,
):
    domain_evaluator = Evaluator()
    evaluator_1 = DSTEvaluator(slot_meta)
    evaluator_2 = DSTEvaluator(slot_meta)
    outputs = {}
    all_length = len(data)
    model.eval()
    for idx in range(all_length):
        instance = get_instance_from_multiwoz(
            data, idx, slot_meta, tokenizer, slot_desc
        )
        for tid in range(instance.n_turn):
            input_ids, target_masks, did = instance.get_instance_by_turn(tid)
            input_ids = input_ids.to(device)
            target_masks = target_masks.to(device)
            with torch.no_grad():
                input_mask = input_ids.ne(tokenizer.pad_token_id).float()
                o = model(input_ids, attention_mask=input_mask)
                before_threshold_idx, pred_idx = output_threshold(
                    o[0], target_masks, threshold
                )
                bti_pred, domain = instance.recover(before_threshold_idx)
                pred, _ = instance.recover(pred_idx)

            label = make_dst_target(
                data[idx]["dialogue"][tid]["belief_state"], slot_meta
            )
            if specific_domain:
                label = [l for l in label if l.startswith(specific_domain)]
                pred = [l for l in pred if l.startswith(specific_domain)]
                bti_pred = [l for l in bti_pred if l.startswith(specific_domain)]

            outputs[did] = [pred, label]

            domain_evaluator.update(data[idx]["dialogue"][tid]["domain"], domain)
            evaluator_1.update(label, bti_pred)
            evaluator_2.update(label, pred)

        if verbose:
            print("[%d/%d]" % (idx, all_length))
        elif idx % 100 == 0:
            print("[%d/%d]" % (idx, all_length))

    domain_accuracy = domain_evaluator.compute()
    eval_before_threshold = evaluator_1.compute()
    eval_with_threshold = evaluator_2.compute()
    eval_before_threshold["domain_accuracy"] = domain_accuracy
    return eval_before_threshold, eval_with_threshold, outputs


def find_best_threshold(
    model,
    tokenizer,
    slot_desc,
    slot_meta,
    dev_dataset,
    test_dataset,
    specific_domain=None,
):
    print("Find best threshold based on dev set... %s" % str(specific_domain))
    greed = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_score = 0
    best_threshold = 0.5
    for threshold in greed:
        eval_before, eval_after, _ = dst_evaluation(
            model,
            dev_dataset,
            tokenizer,
            slot_desc,
            slot_meta,
            False,
            threshold,
            specific_domain=specific_domain,
        )
        print(eval_before)
        print(eval_after)
        print(threshold, ">")
        if eval_after["joint_goal_accuracy"] > best_score:
            best_score = eval_after["joint_goal_accuracy"]
            best_threshold = threshold

    print("Best threshold: %.2f" % best_threshold)
    print("Eval on test set...")
    eval_before, eval_after, _ = dst_evaluation(
        model,
        test_dataset,
        tokenizer,
        slot_desc,
        slot_meta,
        False,
        best_threshold,
        specific_domain=specific_domain,
    )
    print(eval_before)
    print(eval_after)
