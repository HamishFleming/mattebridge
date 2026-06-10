from pathlib import Path

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ONNX_FILE = Path("rvm_simplified.onnx")
ENGINE_FILE = Path("rvm_fp16.engine")


INPUT_SHAPES = {
    "src": (1, 3, 512, 512),
    "r1i": (1, 16, 256, 256),
    "r2i": (1, 20, 128, 128),
    "r3i": (1, 40, 64, 64),
    "r4i": (1, 64, 32, 32),
}


def build_engine(onnx_file: Path = ONNX_FILE, engine_file: Path = ENGINE_FILE) -> None:
    if not onnx_file.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_file}")

    builder = trt.Builder(TRT_LOGGER)
    #network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    #network = builder.create_network(network_flags)
    network = builder.create_network()
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with onnx_file.open("rb") as model_stream:
        if not parser.parse(model_stream.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errors}")

    profile = builder.create_optimization_profile()
    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)
        shape = INPUT_SHAPES.get(tensor.name)
        if shape is None:
            raise KeyError(f"No optimization shape configured for input '{tensor.name}'")
        profile.set_shape(tensor.name, shape, shape, shape)

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with engine_file.open("wb") as engine_stream:
        engine_stream.write(serialized_engine)

    print(f"Engine built: {engine_file}")


if __name__ == "__main__":
    build_engine()
