from copy import deepcopy
import random
from .multiwoz_utils import get_summary_bstate


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


class GoalSchema:
    def __init__(
        self,
        domain: str,
        informable: list,
        book: list,
        entity: str,
        requestable: list,
        book_request: list,
        booking_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.domain = domain
        self.informable = informable
        self.book = book
        self.entity = entity
        self.requestable = requestable
        self.book_request = book_request
        self.booking_ratio = booking_ratio
        self.rng = random.Random(seed)

    def generate_goal_from_entity(self, entity):
        goal = {}
        key = self.domain + "-" + self.entity
        value = entity
        goal[key] = value
        pool = deepcopy(self.requestable + self.informable)
        num_req = self.rng.choice(range(1, len(pool)))
        for _ in range(num_req):
            slot = self.rng.choice(pool)
            pool.pop(pool.index(slot))
            key = self.domain + "-" + slot
            goal[key] = "?"
        return goal

    def generate_goal_from_instance(self, instance, ontology):
        goal = {}
        booking = True if self.rng.random() < self.booking_ratio else False

        for slot, value in instance.items():
            if slot in self.informable:
                key = self.domain + "-" + slot
                goal[key] = value

        pool = deepcopy(self.requestable)
        num_req = self.rng.choice(range(1, len(pool)))
        for _ in range(num_req):
            slot = self.rng.choice(pool)
            pool.pop(pool.index(slot))
            key = self.domain + "-" + slot
            goal[key] = "?"

        if booking:
            for slot in self.book:
                key = self.domain + "-" + slot
                value = self.rng.choice(ontology[key])
                goal[key] = value

            for slot in self.book_request:
                key = self.domain + "-" + slot
                goal[key] = "?"

        return goal


class GoalProfiler:
    def __init__(self, data):
        self.data = data
        self.goals = {}
        self.goal_check = {}
        self.goal_arch_keys = {}
        self.domains = {}
        for k, v in data.items():
            self.goals[k], self.domains[k] = self.profile_existing_log(v["log"])
            self.goal_check[k] = self.profile_existing_goal(v["goal"])

    def profile_existing_log(self, log):
        goal_pool = []
        domains = []
        for i, l in enumerate(log):
            if i % 2 == 0:
                continue

            _, states = get_summary_bstate(l["metadata"])
            domain = get_summary_bstate(l["metadata"], True)
            for state in states:
                sv = "-".join(state)
                if sv not in goal_pool:
                    goal_pool.append(sv)
            for d in domain:
                if d not in domains:
                    domains.append(d)
        return goal_pool, domains

    def profile_existing_goal(self, goal):
        profiled_goal = {}
        for domain in EXPERIMENT_DOMAINS:
            gd = goal[domain]
            if not gd:
                continue
            info = gd.get("info", {})
            for k, v in info.items():
                profiled_goal["%s-%s" % (domain, k.lower())] = v

            fail_info = gd.get("fail_info", {})
            for k, v in fail_info.items():
                if profiled_goal.get("%s-%s" % (domain, k.lower())) != v:
                    profiled_goal["%s_fail-%s" % (domain, k.lower())] = v

            book = gd.get("book", {})
            for k, v in book.items():
                if "valid" in k:
                    continue
                profiled_goal["%s-book %s" % (domain, k.lower())] = v
            fail_book = gd.get("fail_book", {})
            for k, v in fail_book.items():
                profiled_goal["%s_fail-book %s" % (domain, k.lower())] = v

            for k in gd.get("reqt", []):
                profiled_goal["%s-%s" % (domain, k.lower())] = "?"

        goal_list = []
        for k, v in profiled_goal.items():
            if v == "?":
                continue

            k = k.replace("_fail", "")
            goal_list.append(k + "-" + v)

        return goal_list


## Pre-defined Schema

restaurant = GoalSchema(
    "restaurant",
    informable=["area", "food", "pricerange"],
    book=["book day", "book people", "book time"],
    entity="name",
    requestable=["address", "phone", "postcode", "signature"],
    book_request=["reference number"],
)


attraction = GoalSchema(
    "attraction",
    informable=["area", "type"],
    book=[],
    entity="name",
    requestable=["pricerange", "phone", "postcode", "entrance fee", "openhours"],
    book_request=[],
)

hotel = GoalSchema(
    "hotel",
    informable=["area", "type", "internet", "parking", "pricerange", "stars"],
    book=["book day", "book people", "book stay"],
    entity="name",
    requestable=["address", "phone", "postcode"],
    book_request=["reference number"],
)

train = GoalSchema(
    "train",
    informable=["arriveby", "day", "departure", "destination", "leaveat"],
    book=["book people"],
    entity=None,
    requestable=["trainID", "price", "duration"],
    book_request=["reference number"],
)

taxi = GoalSchema(
    "taxi",
    informable=["arriveby", "departure", "destination", "leaveat"],
    book=[],
    entity=None,
    requestable=["color", "types", "phone"],
    book_request=[],
)

SCHEMAS = {
    "restaurant": restaurant,
    "hotel": hotel,
    "attraction": attraction,
    "train": train,
    "taxi": taxi,
}
