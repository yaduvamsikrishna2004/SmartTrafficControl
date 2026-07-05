class Counter:
    def __init__(self):
        self.counts = {"vehicles": 0}

    def increment(self):
        self.counts["vehicles"] += 1
        return self.counts
