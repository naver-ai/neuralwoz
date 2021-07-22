from collector import  Collector
from utils.data_utils import download_checkpoint
from utils.constants import SPECIAL_TOKENS
from utils.collector_utils import APIInstance, CollectorInstance, preprocess_goal

import torch
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


download_checkpoint('collector_multiwoz')
collector = Collector('collector_multiwoz', device)


domain = 'entertainment'

goal = [
    "You are looking for entertainment",
    "The entertainment should be type of movie and hold on monday",
    "The entertainment should be held at Los Angeles",
    "If you find the entertainment, you want to book it for 2 people",
    "Make sure you get the reference number and running time of the movie",
    "Please ask about they have snack bar"
]

api = {
    'type': 'movie',
    'day': 'monday',
    'location': 'Los Angeles',
    'start time': '19:00',
    'end time': '21:00',
    'running time': '2 hours',
    'name': 'Lion King',
    'reference number': 'XQSD3',
    "etc": 'snack bar is available'
}

goal = preprocess_goal(goal)

api = APIInstance(domain, api)
x = CollectorInstance('test', goal, [api])
x.processing(collector.tokenizer)

dialog = collector.sampling_dialogue(x,
            top_p=0.98,
            temperature=0.7,
            do_sample=True,
            num_beams=1)

pprint(dialog)