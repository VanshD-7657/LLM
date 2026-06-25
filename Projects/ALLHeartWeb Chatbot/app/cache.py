import collections
from typing import Any, Optional

class SimpleCache:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache = collections.OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        if key in self.cache:
            # Move to end to represent most recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: Any, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Pop the first item (least recently used)
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

# Global cache instances
embedding_cache = SimpleCache(max_size=500)
retrieval_cache = SimpleCache(max_size=200)
response_cache = SimpleCache(max_size=200)
