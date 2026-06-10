from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = Path("rvm_fp16.engine")
FRAME_SIZE = (512, 512)
V4L2_DEVICE = "/dev/video10"
STATE_BINDINGS = [("r1i", "r1o"), ("r2i", "r2o"), ("r3i", "r3o"), ("r4i", "r4o")]
TENSOR_SHAPES = {
    "src": (1, 3, 512, 512),
    "r1i": (1, 16, 256, 256),
    "r2i": (1, 20, 128, 128),
    "r3i": (1, 40, 64, 64),
    "r4i": (1, 64, 32, 32),
    "fgr": (1, 3, 512, 512),
    "pha": (1, 1, 512, 512),
    "r1o": (1, 16, 256, 256),
    "r2o": (1, 20, 128, 128),
    "r3o": (1, 40, 64, 64),
    "r4o": (1, 64, 32, 32),
}


def open_capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def build_input_source(args: argparse.Namespace) -> str | int:
    if args.input_camera is not None:
        return args.input_camera
    if args.input_rtsp is not None:
        return args.input_rtsp
    if args.input_udp is not None:
        return f"udp://@:{args.input_udp}"
    return 0


def wait_for_capture_open(source: str | int, timeout_s: int = 15) -> cv2.VideoCapture:
    deadline = time.time() + timeout_s
    last_cap = None

    while time.time() < deadline:
        cap = open_capture(source)
        if cap.isOpened():
            return cap
        last_cap = cap
        cap.release()
        time.sleep(0.5)

    if last_cap is not None:
        last_cap.release()
    raise RuntimeError(f"Timed out waiting for input source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--input-camera", type=int, help="Local webcam device index")
    source_group.add_argument("--input-rtsp", type=str, help="RTSP stream URL")
    source_group.add_argument("--input-udp", type=int, help="UDP port to listen on")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a local preview window with the processed output",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show a local preview window and skip writing to /dev/video10",
    )
    args = parser.parse_args()

    if args.preview_only:
        args.display = True

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Missing TensorRT engine: {ENGINE_PATH}")

    with ENGINE_PATH.open("rb") as engine_stream:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_stream.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    buffers = {}

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = TENSOR_SHAPES[tensor_name]
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        host = cuda.pagelocked_empty(trt.volume(shape), dtype)
        device = cuda.mem_alloc(host.nbytes)
        buffers[tensor_name] = {"host": host, "device": device, "shape": shape}
        context.set_tensor_address(tensor_name, int(device))

    for tensor_name in ["src", "r1i", "r2i", "r3i", "r4i"]:
        context.set_input_shape(tensor_name, TENSOR_SHAPES[tensor_name])

    input_source = build_input_source(args)
    cap = wait_for_capture_open(input_source)

    out_cam = None
    if not args.preview_only:
        out_cam = cv2.VideoWriter(
            V4L2_DEVICE,
            cv2.CAP_V4L2,
            cv2.VideoWriter_fourcc(*"YUYV"),
            30,
            FRAME_SIZE,
            True,
        )
        if not out_cam.isOpened():
            raise RuntimeError(f"Cannot open {V4L2_DEVICE} (v4l2loopback)")

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

            for input_name, _ in STATE_BINDINGS:
                np.copyto(buffers[input_name]["host"], recurrent_state[input_name])
                cuda.memcpy_htod_async(
                    buffers[input_name]["device"],
                    buffers[input_name]["host"],
                    stream,
                )

            context.execute_async_v3(stream_handle=stream.handle)

            for output_name in ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"]:
                cuda.memcpy_dtoh_async(
                    buffers[output_name]["host"],
                    buffers[output_name]["device"],
                    stream,
                )

            stream.synchronize()

            for input_name, output_name in STATE_BINDINGS:
                recurrent_state[input_name] = buffers[output_name]["host"].copy()

            fgr = buffers["fgr"]["host"].reshape(buffers["fgr"]["shape"])[0].transpose(1, 2, 0)
            pha = buffers["pha"]["host"].reshape(buffers["pha"]["shape"])[0, 0]
            out = np.clip(fgr * pha[..., None], 0.0, 1.0)
            out = (out * 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            if out_cam is not None:
                out_cam.write(out)
            if args.display:
                cv2.imshow("rvm", out)
                if cv2.waitKey(1) == 27:
                    break

    finally:
        cap.release()
        if out_cam is not None:
            out_cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
