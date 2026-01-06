import numpy as np
import os

class Cache:
    def __init__(self, path):
        self.path = path

        dir = os.path.dirname(self.path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(self.path):
            os.remove(self.path)

    def _load(self):
        if os.path.exists(self.path):
            with np.load(self.path, allow_pickle=True) as data:
                return dict(data)

        return {}

    def set(self, key, value):
        data = self._load()
        data[str(key)] = value
        np.savez(self.path, **data)

        return str(key)

    def has(self, key):
        data = self._load()
        return str(key) in data

    def fetch(self, key):
        data = self._load()
        key = str(key)

        if key not in data:
            raise KeyError(f"No '{key}' in {self.path}")

        return data[key]