from anomaly.device import pick_device


def test_prefers_cuda_then_mps_then_cpu():
    assert pick_device(cuda_available=True, mps_available=True) == "cuda"
    assert pick_device(cuda_available=False, mps_available=True) == "mps"
    assert pick_device(cuda_available=False, mps_available=False) == "cpu"
