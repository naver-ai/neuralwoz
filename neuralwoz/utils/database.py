import os
import json
from fuzzywuzzy import fuzz
import re


DOMAIN_PK = {
    "train": "trainID",
    "hotel": "name",
    "attraction": "name",
    "restaurant": "name",
    "taxi": "phone",
    "police": "addr",
    "hospital": "department",
}


class Database:
    def __init__(self, root_dir):
        self.db = {}
        for file in os.listdir(root_dir):
            if not file.endswith("db.json"):
                continue

            domain = file.replace("_db.json", "")
            try:
                self.db[domain] = json.load(open(os.path.join(root_dir, file), "r"))
            except:
                # taxi
                pass

    def search_info(self, dom, info):
        candits = []
        for k, v in info.items():
            if not candits:
                candits = self.search(dom, v, k, return_all=True, fuzzy=False)
            else:
                candits = self._filter(k, v, candits)
        return candits

    def _filter(self, k, v, candits):
        r = [[d, self._match(v, d[k], False)] for d in candits]
        r = sorted(r, key=lambda x: x[1], reverse=True)
        r = [x[0] for x in r if x[1] > 0.0]
        return r

    def _match(self, v, v2, fuzzy=True):
        v = self.normalize(v)
        v2 = self.normalize(v2)
        if fuzzy:
            return fuzz.ratio(v, v2)
        if v == v2:
            return 1.0
        else:
            return 0.0

    def search(self, dom, pv, pk=None, return_all=False, fuzzy=True):
        n = self.normalize(pv)
        if not pk:
            pk = DOMAIN_PK[dom]
        r = [[d, self._match(pv, d[pk], fuzzy)] for d in self.db[dom]]
        r = sorted(r, key=lambda x: x[1], reverse=True)
        r = [x[0] for x in r if x[1] > 0.0]
        if not r:
            return None
        if not return_all:
            return r[0]
        return r

    def normalize(self, text):
        RE_ART = re.compile(r"\b(a|an|the)\b")
        RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
        text = RE_PUNC.sub(" ", text.lower())
        text = RE_ART.sub(" ", text)
        return re.sub("\s+", " ", text).strip()
