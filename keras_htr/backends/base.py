class ToolSetFactory:
    def get_preprocessor(self):
        raise NotImplementedError

    def get_adapter(self):
        raise NotImplementedError

    def get_model(self, *args, **kwargs):
        raise NotImplementedError
