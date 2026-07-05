class EmergencyHandler:
    def __init__(self):
        self.active = False

    def trigger(self):
        self.active = True
        return self.active
