BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"
USER = "<user>"
SYS = "<sys>"
GOAL = "<goal>"
ENDGOAL = "</goal>"
API = "<api>"
ENDAPI = "</api>"
DOMAIN = "<domain>"
ENDDOMAIN = "</domain>"
SLOT = "<slot>"
ENDSLOT = "</slot>"
NONE = "none"
DONTCARE = "dontcare"

SPECIAL_TOKENS = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "unk_token": UNK,
    "additional_special_tokens": [USER, SYS, DOMAIN, SLOT],
}


MAPPER = {
    "dest": "destination",
    "depart": "departure",
    "leave": "leaveat",
    "arrive": "arriveby",
}


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
NOT_DOMAIN = ["hospital", "police"]
ENTITY_DOMAIN = ["restaurant", "attraction", "hotel"]
NON_DB_DOMAIN = ["taxi", "police", "hospital"]
BOOLEAN_TYPE = ["hotel-internet", "hotel-parking"]
DOMAIN_DESC = ["domain of the current turn", "topic of the current turn"]
