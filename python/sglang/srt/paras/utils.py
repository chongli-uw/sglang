


def paras_func(func):
    """
    Decorator to ensure that the function is called with the `paras_configure_helper`
    """
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if hasattr(self, 'paras_configure_helper'):
            self.paras_configure_helper()
        return result
    return wrapper