class SignalController:
    def __init__(self):
        self.mode = "auto"

    def set_mode(self, mode):
        self.mode = mode
        return self.mode
