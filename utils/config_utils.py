import json

class objdict():
    def __init__(self, d):
        self.__dict__ = d

def save_config(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        json.dump(FLAGS_dict, fp, indent=4)
        
def load_config(in_path):
    with open(in_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
    return objdict(FLAGS_dict)