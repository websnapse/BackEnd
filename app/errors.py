class WebsnapseError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def to_dict(self):
        return {"message": self.message, "status_code": self.status_code}
