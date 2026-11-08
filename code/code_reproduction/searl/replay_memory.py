"""Sequential replay memory, equivalent to the non-multiprocessing
`ReplayMemory` class in searl/neuroevolution/components/replay_memory.py
(and searl/rl_algorithms/components/replay_memory.py, which is the same
logic). The original repo's `MPReplayMemory` runs this behind a
multiprocessing.Manager queue so many worker processes can share one replay
buffer; this reproduction runs everything single-process/sequential, so a
plain in-process buffer is behaviorally identical.
"""
from typing import List

import numpy as np


class ReplayMemory:
    def __init__(self, capacity: int, batch_size: int):
        self.storage = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.ptr = 0

    def add(self, transitions):
        if isinstance(transitions, List):
            for transition in transitions:
                self._add(transition)
        else:
            self._add(transitions)

    def put(self, transitions):
        self.add(transitions)

    def _add(self, transition):
        if len(self.storage) == self.capacity:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.capacity
        else:
            self.storage.append(transition)

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        return [self.storage[i] for i in ind]

    def __len__(self):
        return len(self.storage)
