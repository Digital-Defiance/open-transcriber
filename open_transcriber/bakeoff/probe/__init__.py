"""The identical probe for the encoder bake-off: extractor protocol, shared body + heads, train, eval, decide.

Pure-PyTorch and MPS-agnostic: every unit test runs on CPU with tiny synthetic tensors and no encoder
download. Encoders sit behind the ``Extractor`` protocol so feature extraction (GPU) is separable from probe
training (GPU) and from the CPU-testable model/decode/decision code.
"""
