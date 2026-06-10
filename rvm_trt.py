from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_FILE = Path("rvm_fp16.engine")
FRAME_SIZE = (512, 512)
STATE_BINDINGS = [("r1i", "r1o"), ("r2i", "r2o"), ("r3i", "r3o"), ("r4i", "r4o")]


def main() -> None:
    with ENGINE_FILE.open("rb") as engine_stream:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_stream.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    buffers = {}
    bindings = []

    for binding_name in engine:
        binding_index = engine.get_binding_index(binding_name)
        shape = tuple(engine.get_binding_shape(binding_index))
        dtype = trt.nptype(engine.get_binding_dtype(binding_index))
        host = cuda.pagelocked_empty(trt.volume(shape), dtype)
        device = cuda.mem_alloc(host.nbytes)
        buffers[binding_name] = {"host": host, "device": device, "shape": shape}
        bindings.append(int(device))

    cap = cv2.VideoCapture(0)
    rec_outputs = [name for _, name in STATE_BINDINGS]
    recurrent_state = {
        input_name: np.zeros_like(buffers[input_name]["host"])
        for input_name, _ in STATE_BINDINGS
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            src = rgb.astype(np.float32) / 255.0
            src = np.transpose(src, (2, 0, 1))[None]

            np.copyto(buffers["src"]["host"], src.ravel())
            cuda.memcpy_htod_async(buffers["src"]["device"], buffers["src"]["host"], stream)

            for input_name, output_name in STATE_BINDINGS:
                np.copyto(buffers[input_name]["host"], recurrent_state[input_name])
                cuda.memcpy_htod_async(buffers[input_name]["device"], buffers[input_name]["host"], stream)

            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            for binding_name in ["fgr", "pha", *rec_outputs]:
                cuda.memcpy_dtoh_async(buffers[binding_name]["host"], buffers[binding_name]["device"], stream)

            stream.synchronize()
            for input_name, output_name in STATE_BINDINGS:
                recurrent_state[input_name] = buffers[output_name]["host"].copy()

            fgr = buffers["fgr"]["host"].reshape(buffers["fgr"]["shape"])[0].transpose(1, 2, 0)
            pha = buffers["pha"]["host"].reshape(buffers["pha"]["shape"])[0, 0]
            out = np.clip(fgr * pha[..., None], 0.0, 1.0)
            out = (out * 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            cv2.imshow("rvm", out)

            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
