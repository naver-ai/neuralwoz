import random


def coref_exp(target):
    domain, slot = target.split("-")
    if "book" in slot:
        return f"same {slot}"
    return f"same {slot} of the {domain}"


def numerical_value_generator(min_=1, max_=10):
    return str(random.choice(range(min_, max_)))


def day_value_generator():
    candidates = ["mon", "thues", "wednes", "thurs", "fri", "satur", "sun"]
    prefix = random.choice(candidates)
    return prefix + "day"


def time_value_generator(base_hour=None, base_min=None):
    if not base_hour:
        base_hour = 1
    hour = random.choice(range(base_hour, 25))
    if not base_min:
        base_min = 1
    minute = random.choice(range(base_min, 60))
    return "%d:%d" % (hour, minute)


def date_value_generator(base):
    return "none"


def reference_value_generator():
    candidates = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ref = ""
    while len(ref) < 4:
        ref += random.choice(list(candidates))
    n = random.choice(range(10, 100))
    return ref + str(n)


def color_value_generator():
    return random.choice(["black", "white", "red", "yellow", "blue", "grey"])


def carType_value_generator():
    return random.choice(
        [
            "toyota",
            "skoda",
            "bmw",
            "honda",
            "ford",
            "audi",
            "lexus",
            "volvo",
            "volkswagen",
            "tesla",
        ]
    )


def phone_value_generator():
    pool = list(range(100, 1000))
    nums = random.sample(pool, 3)
    nums = [str(n) for n in nums]
    return "0" + "".join(nums)


VALUE_META = {
    "numeric": numerical_value_generator,
    "day": day_value_generator,
    "time": time_value_generator,
    "date": date_value_generator,
    "reference": reference_value_generator,
    "color": color_value_generator,
    "carType": carType_value_generator,
    "phone": phone_value_generator,
}


BOOLEAN_EXP = {
    "hotel-parking": {
        "yes": ["should include free parking"],
        "no": ["doesn't need to include parking"],
    },
    "hotel-internet": {
        "yes": ["should include free internet"],
        "no": ["doesn't need to include internet"],
    },
}
