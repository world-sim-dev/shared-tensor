

__all__ = ["SharedTensorError", "SharedTensorServerError", "SharedTensorClientError", "SharedTensorProviderError"]


class SharedTensorError(Exception):
    pass

class SharedTensorServerError(SharedTensorError):
    pass

class SharedTensorClientError(SharedTensorError):
    pass

class SharedTensorProviderError(SharedTensorError):
    pass