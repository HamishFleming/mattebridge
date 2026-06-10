#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_PATH = SCRIPT_DIR / "rvm_fp16.engine"
FRAME_SIZE = (512, 512)
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


def letterbox_frame(
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    src_h, src_w = frame.shape[:2]
    dst_w, dst_h = FRAME_SIZE

    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame, (resized_w, resized_h))
    pad_w = dst_w - resized_w
    pad_h = dst_h - resized_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    canvas = np.zeros((dst_h, dst_w, 3), dtype=frame.dtype)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    return canvas, (pad_left, pad_top, resized_w, resized_h), (src_w, src_h)


def restore_frame_size(
    frame: np.ndarray,
    content_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    pad_left, pad_top, content_w, content_h = content_box
    src_w, src_h = output_size
    cropped = frame[pad_top : pad_top + content_h, pad_left : pad_left + content_w]
    return cv2.resize(cropped, (src_w, src_h))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RVM TensorRT background remover on a video file and write a new video."
    )
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the output FPS. Defaults to the input video FPS or 30 if unknown.",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=None,
        help="Override the output codec fourcc, for example mp4v or XVID.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a preview window while processing.",
    )
    return parser.parse_args()


def infer_fourcc(output_path: Path, override: str | None) -> str:
    if override:
        if len(override) != 4:
            raise ValueError("--fourcc must be exactly 4 characters")
        return override

    suffix = output_path.suffix.lower()
    if suffix in {".avi", ".divx"}:
        return "XVID"
    if suffix in {".mov", ".m4v", ".mp4"}:
        return "mp4v"
    return "mp4v"


def load_engine() -> tuple[trt.ICudaEngine, trt.IExecutionContext, cuda.Stream, dict[str, dict[str, object]]]:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Missing TensorRT engine: {ENGINE_PATH}")

    with ENGINE_PATH.open("rb") as engine_stream:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_stream.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine from {ENGINE_PATH}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    stream = cuda.Stream()
    buffers: dict[str, dict[str, object]] = {}

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

    return engine, context, stream, buffers


def process_frame(
    frame: np.ndarray,
    context: trt.IExecutionContext,
    stream: cuda.Stream,
    buffers: dict[str, dict[str, object]],
    recurrent_state: dict[str, np.ndarray],
) -> np.ndarray:
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
        cuda.memcpy_dtoh_async(buffers[output_name]["host"], buffers[output_name]["device"], stream)

    stream.synchronize()

    for input_name, output_name in STATE_BINDINGS:
        recurrent_state[input_name] = buffers[output_name]["host"].copy()

    fgr = buffers["fgr"]["host"].reshape(buffers["fgr"]["shape"])[0].transpose(1, 2, 0)
    pha = buffers["pha"]["host"].reshape(buffers["pha"]["shape"])[0, 0]
    out_rgb = np.clip(fgr * pha[..., None], 0.0, 1.0)
    out_bgr = (out_rgb * 255).astype(np.uint8)
    return cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    _engine, context, stream, buffers = load_engine()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    try:
        fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Input video contains no frames: {input_path}")

        output_size = (first_frame.shape[1], first_frame.shape[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = infer_fourcc(output_path, args.fourcc)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            output_size,
            True,
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Cannot open output video writer for {output_path} using fourcc {fourcc}"
            )

        recurrent_state = {
            input_name: np.zeros_like(buffers[input_name]["host"])
            for input_name, _ in STATE_BINDINGS
        }

        try:
            frame = first_frame
            while True:
                letterboxed, content_box, native_size = letterbox_frame(frame)
                out = process_frame(letterboxed, context, stream, buffers, recurrent_state)
                out = restore_frame_size(out, content_box, native_size)
                writer.write(out)

                if args.display:
                    cv2.imshow("rvm-background-removal", out)
                    if cv2.waitKey(1) == 27:
                        break

                ret, frame = cap.read()
                if not ret:
                    break
        finally:
            writer.release()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
