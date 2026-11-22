"""Model pipeline of TCC v2: PyTorch -> ONNX -> TensorRT and environment manifests.

Wave 1: environment probe and viability spike. Wave 2A: the FP32 network
builder for expressed genomes (``builder``), a short builder-to-TensorRT smoke
(``builder_smoke``) and the data-access barrier for the official test split
(``data_access``). Physical pruning, Q/DQ quantization, the isolated benchmark
and the evaluator are not implemented yet.
"""
