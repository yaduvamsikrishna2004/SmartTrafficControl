class DensityAnalyzer:
    def __init__(self):
        self.level = "low"

    def evaluate(self, count):
        if count > 10:
            self.level = "high"
        elif count > 5:
            self.level = "medium"
        else:
            self.level = "low"
        return self.level
