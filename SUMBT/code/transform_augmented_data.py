import json
import os
import argparse
from unidecode import unidecode


def get_path(dir, file_name):
    return os.path.join(dir, file_name)


def fix_tokenization(phrase):
    phrase = phrase.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" :", ":").replace(" !", "!")
    phrase = phrase.replace(" -s ", "s ").replace(" it s ", " it's ").replace(" i'", " I'")
    phrase = remove_non_ascii(phrase)  # remove null byte
    return phrase


def remove_non_ascii(text):
    text = text.replace('\x00', '')
    return unidecode(text)


repl_dict = {'hotel-pricerange': 'hotel-price range',
 'restaurant-pricerange': 'restaurant-price range',
 'taxi-arriveby': 'taxi-arrive by',
 'taxi-leaveat': 'taxi-leave at',
 'train-arriveby': 'train-arrive by',
 'train-leaveat': 'train-leave at'}


SLOTS = ['attraction-area', 'attraction-name', 'attraction-type', 'hotel-area', 'hotel-book day', \
'hotel-book people', 'hotel-book stay', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-price range', \
'hotel-stars', 'hotel-type', 'restaurant-area', 'restaurant-book day', 'restaurant-book people', \
'restaurant-book time', 'restaurant-food', 'restaurant-name', 'restaurant-price range', \
'taxi-arrive by', 'taxi-departure', 'taxi-destination', 'taxi-leave at', 'train-arrive by', \
'train-book people', 'train-day', 'train-departure', 'train-destination', 'train-leave at']


GENERAL_TYPO = {
    # type
    "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
    "mutiple sports": "multiple sports",
    "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
    "concerthall": "concert hall",
    "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
    "ol": "architecture",
    "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum", "churches": "church",
    # area
    "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
    "cen": "centre", "east side": "east",
    "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
    "centre of cambridge": "centre",
    "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre", "in town": "centre",
    "north part of town": "north",
    "centre of town": "centre", "cb30aq": "none",
    # price
    "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
    # day
    "next friday": "friday", "monda": "monday",
    # parking
    "free parking": "free",
    # internet
    "free internet": "yes",
    # star
    "4 star": "4", "4 stars": "4", "0 star rarting": "none",
    # others
    "y": "yes", "any": "do not care", "dontcare": "do not care", "n": "no", "does not care": "do not care", "not men": "none", "not": "none",
    "not mentioned": "none",
    '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
}


def check_if_in_ontology(key, value, ontology):
    domain, slot = key.split('-')
    skipped = False
    if value in GENERAL_TYPO.keys():
        value = GENERAL_TYPO[value]
    if domain not in ontology:
        #print("domain (%s) is not defined" % domain)
        skipped = True
        return skipped, None, "domain (%s) is not defined" % domain
          
    if slot not in ontology[domain]:
        #print("slot (%s) in domain (%s) is not defined" % (slot, domain))   # bus-arriveBy not defined
        skipped = True
        return skipped, None, "slot (%s) in domain (%s) is not defined" % (slot, domain)

    if value not in ontology[domain][slot] and value != 'none' and value != "do not care":
        old_val = value
        value = 'none'
        skipped = True
        return skipped, value, "value (%s) in slot (%s) in domain (%s) is not defined" % (old_val, slot, domain)
        
    return skipped, value, ""


def dialogue_filtering(data, ontology):
    filtered = []
    for idx, diag in enumerate(data):
        filtering = False
        for turn in diag['dialogue']:
            for slot in turn['belief_state']:
                for s in slot['slots']:
                    stype = repl_dict.get(s[0], s[0])
                    value = GENERAL_TYPO.get(s[-1], s[-1])
                    if value in ['do not care']:
                        continue
                    if value not in ontology[stype]:
                        filtering = True
        if not filtering:
            filtered.append(diag)
    print("Filtered dialogue: %d" % len(filtered))
    return filtered


def generate_file(split, fp_in, aug_dir, ontology):
    fp_out = open(get_path(aug_dir, split+".tsv"), 'w', encoding='utf-8')
    fp_out.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
    fp_out.write('\t'.join(SLOTS))
    fp_out.write('\n')

    n_dial = 0
    n_skipped = 0
    n_slot = 0
    for d_idx, dial in enumerate(fp_in):
        #new_dict[dial["dialogue_idx"]] = {}
        dialogue_idx = dial["dialogue_idx"]
        for t_idx, turn in enumerate(dial["dialogue"]):
            if t_idx > 21:
                print("t_idx", t_idx)
                continue
            user_utter = fix_tokenization(turn['transcript'])
            system_utter = fix_tokenization(turn['system_transcript'])
            slot_dict = dict(zip(SLOTS, ["none"]*len(SLOTS)))
            
            fp_out.write(dialogue_idx)         # 0: dialogue ID
            fp_out.write('\t' + str(t_idx))           # 1: turn index
            fp_out.write('\t' + str(user_utter))     # 2: user utterance
            fp_out.write('\t' + str(system_utter))    # 3: system response

            for slot in turn["belief_state"]:
                sl = slot['slots'][0]
                if sl[0].split('-')[0] == "hospital":
                    continue
                if sl[0] in repl_dict.keys():
                    sl[0] = repl_dict[sl[0]]
                slot_dict[sl[0]] = sl[1]

            for key, value in slot_dict.items():
                skipped, value, message = check_if_in_ontology(key, value, ontology)
                if skipped:
                    print(value, message)
                    n_skipped += 1
                assert(value is not None)
                n_slot +=1
                fp_out.write('\t' + value)

            fp_out.write('\n')
        n_dial +=1

    fp_out.close()
    print("n_dial", n_dial, "n_slot", n_slot, "n_skipped", n_skipped, "percent skipped", n_skipped/n_slot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default="../data",
                        type=str)
    parser.add_argument("--target_file",
                        default=None,
                        type=str)
    parser.add_argument("--output_dir",
                        default="../data",
                        type=str)
    args = parser.parse_args()
    
    ### Read ontology file
    ontology_file = get_path(args.input_dir, "ontology.json")
    fp_ont = open(ontology_file, "r")
    data_ont = json.load(fp_ont)
    ontology = {}
    for domain_slot in data_ont:
        domain, slot = domain_slot.split('-')
        if domain in ["bus", "hospital"]:
            continue
        if domain not in ontology:
            ontology[domain] = {}
        ontology[domain][slot] = {}
        for value in data_ont[domain_slot]:
            ontology[domain][slot][value] = 1
    fp_ont.close()

    if args.target_file:
        aug_train_file = get_path(args.input_dir, args.target_file)
        aug_train = json.load(open(aug_train_file, 'r', encoding='utf-8'))
        save_name = args.target_file.replace('.json', '')
        generate_file(save_name, aug_train, args.output_dir, ontology)
    
    if not os.path.exists(os.path.join(args.output_dir, 'train.tsv')):
        aug_train_file = get_path(args.input_dir, 'labeler_train_data.json')
        aug_train = json.load(open(aug_train_file, 'r', encoding='utf-8'))
        generate_file('train', aug_train, args.output_dir, ontology)

    if not os.path.exists(os.path.join(args.output_dir, 'dev.tsv')):
        aug_dev_file = get_path(args.input_dir, 'labeler_dev_data.json')
        aug_dev = json.load(open(aug_dev_file, 'r', encoding='utf-8'))
        generate_file('dev', aug_dev, args.output_dir, ontology)

    if not os.path.exists(os.path.join(args.output_dir, 'test.tsv')):
        aug_test_file = get_path(args.input_dir, 'labeler_test_data.json')
        aug_test = json.load(open(aug_test_file, 'r', encoding='utf-8'))
        generate_file('test', aug_test, args.output_dir, ontology)


if __name__ == "__main__":
    main()
