from tqdm import tqdm
import torch.nn as nn

from utils.config import *
from models.TRADE import *
import os
import json
import random
import numpy as np
import torch

'''
python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -td=train_dials.json
'''

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

early_stop = args['earlyStop']

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

dataset_dir = args['dataset_dir'] #'../data'

if args['train_data']:
    args['train_path'] = os.path.join(args['dataset_dir'], args['train_data'])
else:
    args['train_path'] = os.path.join(args['dataset_dir'], 'labeler_train_data.json')
args['dev_path'] = os.path.join(args['dataset_dir'], 'labeler_dev_data.json')  # same as MultiWOZ2.1 DST dataset
args['test_path'] = os.path.join(args['dataset_dir'], 'labeler_test_data.json')
args['ont_path'] = os.path.join(args['dataset_dir'], 'ontology.json')
args['emb_path'] = os.path.join(args['dataset_dir'], 'embed{}.json')
if not args['output_path']:
    args['output_path'] = args['train_data'].replace('.json', '')

if not os.path.exists(args['output_path']):
    os.makedirs(args['output_path'])

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))

model = globals()[args['decoder']](
    hidden_size=int(args['hidden']), 
    lang=lang, 
    path=args['path'], 
    task=args['task'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict, 
    nb_train_vocab=max_word)

# print("[Info] Slots include ", SLOTS_LIST)
# print("[Info] Unpointable Slots include ", gating_dict)

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))
        model.optimize(args['clip'])
        pbar.set_description(model.print_loss())
        # print(data)
        # exit(1)

    if((epoch+1) % int(args['evalp']) == 0):
        
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
            best_model = model
        else:
            cnt+=1

        if(cnt == args["patience"] or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 


## Test Phase
exp_name = 'TRADE-%smultiwozdst'

if args['except_domain']:
    c = 'Except' + args['except_domain']
elif args['only_domain']:
    c = 'Only' + args['only_domain']
else:
    c = ''
exp_name = exp_name % c

# Follow the revised zero-shot scheme from Campagna et al.
if args['target_domain']:
    args['except_domain'] = args['target_domain']

best_score = 0.
best_directory = None
for directory in os.listdir(os.path.join(args['output_path'], exp_name)):
    score = float(directory.split('-')[-1])
    if score > best_score:
        best_score = score
        best_directory = directory

args['path'] =  os.path.join(args['output_path'], exp_name, directory)
directory = args['path'].split("/")
HDD = directory[-1].split('HDD')[1].split('BSZ')[0]
decoder = directory[-2].split('-')[0] 
BSZ = int(args['batch']) if args['batch'] else int(directory[-1].split('BSZ')[1].split('DR')[0])
args["decoder"] = decoder
args["HDD"] = HDD
args["genSample"] = 1
print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)

train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, args['task'], False, batch_size=BSZ)

model = globals()[decoder](
    int(HDD), 
    lang=lang, 
    path=args['path'], 
    task=args["task"], 
    lr=0, 
    dropout=0,
    slots=SLOTS_LIST,
    gating_dict=gating_dict,
    nb_train_vocab=max_word)

# if args["run_dev_testing"]:
#     print("Development Set ...")
#     acc_dev = model.evaluate(dev, 1e7, SLOTS_LIST[2]) 

# if args['except_domain']!="" and args["run_except_4d"]:
#     print("Test Set on 4 domains...")
#     acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2]) 

print("Test Set ...")
acc_test = model.evaluate(test, 1e7, SLOTS_LIST[3]) 
