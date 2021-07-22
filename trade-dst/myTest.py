from utils.config import *
from models.TRADE import *

'''
python3 myTest.py -path=..
'''

args['train_path'] = os.path.join(args['dataset_dir'], 'labeler_train_data.json')
args['dev_path'] = os.path.join(args['dataset_dir'], 'labeler_dev_data.json')
args['test_path'] = os.path.join(args['dataset_dir'], 'labeler_test_data.json')
args['ont_path'] = os.path.join(args['dataset_dir'], 'ontology.json')
args['genSample'] = True
print(args['path'])

directory = args['path'].strip("/").split("/")
args['output_path'] = "/".join(directory[:-2])

HDD = directory[-1].split('HDD')[1].split('BSZ')[0]
decoder = directory[-2].split('-')[0] 
BSZ = int(args['batch']) if args['batch'] else int(directory[-1].split('BSZ')[1].split('DR')[0])
args["decoder"] = decoder
args["HDD"] = HDD
print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
else:
    print("You need to provide the --dataset information")

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


