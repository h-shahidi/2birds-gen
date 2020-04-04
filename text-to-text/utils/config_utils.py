import json


class objdict():
    def __init__(self, d):
        self.__dict__ = d


def save_config(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        json.dump(FLAGS_dict, fp, indent=4)


def merge_two_dicts(dict1, dict2):
    res = dict1.copy()
    res.update(dict2)
    return res


def load_config(args):
    config_path = args.config_path
    with open(config_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
    FLAGS_dict = merge_two_dicts(FLAGS_dict, vars(args))
    return objdict(FLAGS_dict)