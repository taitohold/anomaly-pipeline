"""Choose which device training runs on."""


def pick_device(cuda_available: bool, mps_available: bool) -> str:
    """Return "cuda", "mps", or "cpu", preferring them in that order.

    The availability flags are passed in so this function can be tested
    without PyTorch installed. The training code will call it with
    torch.cuda.is_available() and torch.backends.mps.is_available().
    """
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"
