import numpy as np

np.random.seed(1010)


class Attribute:

    stddev_perc_of_mean = 0.15

    options = {
        "cid": {"type": int},
        "num_clients": {"low": 50, "mid": 100, "high": 150, "type": int},
        "batch_size": {"low": 16, "mid": 32, "high": 64, "type": int},
        "local_epochs": {
            "low": 2,
            "mid": 5,
            "high": 8,
            "min": 1,
            "max": 10,
            "type": int,
        },
        "num_data_points": {
            "low": 50,
            "mid": 100,
            "high": 150,
            "min": 10,
            "max": 200,
            "type": int,
        },
        "perc_new_data": {
            "low": 0.0,
            "mid": 0.1,
            "high": 0.2,
            "min": 0.0,
            "max": 0.3,
            "type": lambda x: round(float(x), 2),
        },
        "perc_missing_labels": {
            "low": 0.0,
            "mid": 0.15,
            "high": 0.3,
            "min": 0.0,
            "max": 0.5,
            "type": lambda x: round(float(x), 2),
        },
        "len_train_set": {"type": int},
        "len_test_set": {"type": int},
    }

    rng = np.random.default_rng(seed=1010)

    def __init__(self, name: str, variation: str):
        assert (
            name in Attribute.options.keys()
        ), f"Attribute {name} not found in options"
        self.name = name
        assert (
            variation in Attribute.options[name].keys()
        ), f"Variation {variation} not found in options for {name}"
        self.variation = variation

    def generate(self):
        options = Attribute.options[self.name]
        value = options[self.variation]

        sample = self.rng.normal(value, value * self.stddev_perc_of_mean)

        if "min" in options and "max" in options:
            sample = np.clip(sample, options["min"], options["max"])

        return options["type"](sample)

    @classmethod
    def get(cls, name, value):
        return cls.options[name]["type"](value)

    def __repr__(self):
        return f"Attribute({self.name},{self.variation})"
