from transformers import AdamW, get_linear_schedule_with_warmup
from models import BartForConditionalGeneration, BartTokenizer

from utils.data_utils import SPECIAL_TOKENS
from utils.seed import set_seed
from utils.collector_utils import Collectordataset, CollectorInstance, load_collector_data

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import argparse
import numpy as np
import torch
import random
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(model, dev_loader, tokenizer, device):
    total_loss = 0.
    model.eval()
    eval_result = {}
    for step, batch in enumerate(dev_loader):
        with torch.no_grad():
            input_ids, input_masks, target_ids = (b.to(device) for b in batch)
            y_ids = target_ids[:, :-1].contiguous()
            y_ids[y_ids == -100] = tokenizer.pad_token_id
            lm_labels = target_ids[:, 1:].clone()
            outputs = model(input_ids,
                            attention_mask=input_masks,
                            decoder_input_ids=y_ids,
                            lm_labels=lm_labels)
            loss = outputs[0]
            loss = loss.mean()
            total_loss += loss.float().item()
    ppl = np.exp(total_loss / len(dev_loader))
    eval_result['ppl'] = ppl.item()
    return eval_result


def main(args):
    set_seed(args.seed)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_data = json.load(open(os.path.join(args.dataset_dir, args.train_data), 'r'))
    train_data = load_collector_data(train_data, CollectorInstance, args.max_seq_length)
    train_dataset = Collectordataset(train_data, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )
    
    dev_data = json.load(open(os.path.join(args.dataset_dir, args.dev_data), 'r'))
    dev_data = load_collector_data(dev_data, CollectorInstance, args.max_seq_length)
    dev_dataset = Collectordataset(dev_data, tokenizer)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=dev_sampler,
        batch_size=args.train_batch_size,
        collate_fn=dev_dataset.collate_fn
    )
    
    print('Batch size: ', args.train_batch_size * args.gradient_accumulation_steps)
    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    config = model.config.to_dict()
    json.dump(config, open(os.path.join(args.output_path, 'config.json'), 'w'), indent=4)
    tokenizer.save_vocabulary(args.output_path)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    best_score = float('inf')
    best_epoch = 0
    actual_all_step = int(len(train_dataloader) / args.gradient_accumulation_steps)
    for epoch in range(args.num_train_epochs):
        mean_loss = []
        accum_loss = []
        actual_step = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, input_masks, target_ids = (b.to(device) for b in batch)
            y_ids = target_ids[:, :-1].contiguous()
            y_ids[y_ids == -100] = tokenizer.pad_token_id
            lm_labels = target_ids[:, 1:].clone()
            outputs = model(input_ids,
                            attention_mask=input_masks,
                            decoder_input_ids=y_ids,
                            lm_labels=lm_labels)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            accum_loss.append(loss.item())
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                mean_loss.append(np.sum(accum_loss))
                accum_loss = []
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                actual_step += 1

                if actual_step % 100 == 0:
                    print("[%d/%d] [%d/%d] %f" % (epoch, args.num_train_epochs, actual_step, actual_all_step, np.mean(mean_loss)))
                    mean_loss = []
        eval_result = evaluation(model, dev_dataloader, tokenizer, device)
        print(eval_result)
        if eval_result['ppl'] < best_score:
            best_score = eval_result['ppl']
            best_epoch = epoch
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(args.output_path, 'pytorch_model.bin'))
    
    print(best_score, best_epoch)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='data')
    parser.add_argument("--train_data", type=str, required=True, default=None)
    parser.add_argument("--dev_data", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)
    parser.add_argument("--n_gpu", type=int, default=0)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default='facebook/bart-large')
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--fp16", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    else:
        print(f"There is already {args.output_path}. Please doule check the output path if you don't want to overwrite!")
    main(args)
