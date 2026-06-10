# RVM TensorRT Webcam Runtime

This repo contains a small TensorRT webcam runtime for the Robust Video Matting model.
It reads frames from a local webcam or a network stream, runs the matting engine, and writes the result to a v4l2loopback device.

## Files

- `rvm_runtime.py`: TensorRT 11 runtime with recurrent state handling.
- `bridge_rvm_output.py`: helper that runs the runtime and republishes `/dev/video10`.
- `rvm_remove_background_video.py`: processes a video file and writes a new video with the background removed.
- `export_clean_onnx.py`: clean export path from the bundled RVM checkpoint to ONNX.
- `build_engine.py`: builds the TensorRT engine from the cleaned ONNX export.
- `rvm_fp16.engine`: TensorRT engine file expected by the runtime.
- `modnet_runtime.py`: separate runtime for the MODNet ONNX/TensorRT model.
- `RobustVideoMatting/`: vendored upstream source snapshot for the RVM model code and exporter reference.

## Bundled Model Artifacts

The repo includes the artifacts that were used while bringing the runtime up, so it is easier to remember what each file is for later:

- `rvm_mobilenetv3.pth`: upstream PyTorch checkpoint for the MobileNetV3 RVM variant.
- `rvm_simplified.onnx`: cleaned ONNX export used as the TensorRT build input.
- `rvm_fp16.engine`: serialized TensorRT 11 engine built from `rvm_simplified.onnx`.
- `rvm_mobilenetv3_fp16.onnx` and `rvm.onnx`: earlier export outputs kept for reference.
- `modnet_photographic_portrait_matting.onnx`: separate MODNet model used by `modnet_runtime.py`.

Rebuild the RVM artifacts in this order:

```bash
python export_clean_onnx.py
python build_engine.py
```

## Requirements

- Linux
- Python packages: `opencv-python`, `numpy`, `pycuda`, `tensorrt`
- A working `v4l2loopback` device at `/dev/video10`
- OpenCV built with FFmpeg support for RTSP and UDP input

## Create The Virtual Camera

If `/dev/video10` does not exist yet, load `v4l2loopback` first:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=rvm exclusive_caps=1
```

Verify the device is present:

```bash
ls -l /dev/video10
```

## Run

Local webcam:

```bash
python rvm_runtime.py --input-camera 0
```

Local webcam with a preview window on the same machine:

```bash
python rvm_runtime.py --input-camera 0 --display
```

Local webcam preview only, without `/dev/video10`:

```bash
python rvm_runtime.py --input-camera 0 --preview-only
```

RTSP stream:

```bash
python rvm_runtime.py --input-rtsp rtsp://SOURCE_IP:8554/webcam
```

UDP receiver:

```bash
python rvm_runtime.py --input-udp 5000
```

If you do not pass any input flag, the script defaults to camera index `0`.
Use `--display` to open a local preview window while still writing the processed frames to `/dev/video10`.
Use `--preview-only` when you only want the local preview and do not want to create or use `v4l2loopback`.

Video file input and output:

```bash
python rvm_remove_background_video.py --input input.mp4 --output output.mp4
```

The output video keeps the same dimensions as the input, preserves the aspect ratio, and composites the foreground onto a black background.

## Startup Script

For the bridge workflow, use the wrapper script from the repo root:

```bash
./start_bridge_rvm_output.sh --input-udp 5000 --output-udp A_IP:6000
```

The script uses `venv/bin/python` when available and otherwise falls back to `python3`.

## Streaming From Another Computer

The simplest setup is to encode the source webcam with `ffmpeg` and send it over the network.

### UDP

On the source computer:

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts \
  udp://TARGET_IP:5000
```

On this computer:

```bash
python rvm_runtime.py --input-udp 5000
```

### RTSP

If you already have an RTSP server, stream the webcam to it and point the runtime at the RTSP URL:

```bash
python rvm_runtime.py --input-rtsp rtsp://SERVER_IP:8554/webcam
```

## One Command On Machine B

If machine B receives the webcam stream, runs matting, and sends the background-free result back out, use the bridge script on machine B.

### Input to machine B

From machine A to machine B over UDP:

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts \
  udp://B_IP:5000
```

Or from machine A to machine B over RTSP:

```bash
python rvm_runtime.py --input-rtsp rtsp://A_OR_SOURCE_IP:8554/webcam
```

### Machine B bridge

UDP output back to machine A:

```bash
./bridge_rvm_output.py --input-udp 5000 --output-udp A_IP:6000
```

RTSP output to an RTSP server:

```bash
./bridge_rvm_output.py --input-udp 5000 --output-rtsp rtsp://A_IP:8554/rvm
```

### View on machine A

If B sends UDP back:

```bash
ffplay udp://@:6000
```

If B publishes RTSP:

```bash
ffplay rtsp://A_IP:8554/rvm
```

Note: `0.0.0.0` is only valid as a listen address for a server or receiver. It is not a client URL. When connecting from `rvm_runtime.py`, use the actual source machine IP or hostname.

## Notes

- The runtime resizes every frame to `512x512`.
- The output is written to `/dev/video10` via a YUYV v4l2loopback writer, and the bridge reads it back from that device as `yuyv422`.
- The model keeps four recurrent state tensors between frames, so do not restart the process if you want stable temporal results.
- The RVM runtime uses the MobileNetV3 checkpoint, not the upstream ResNet50 variant.
- If `cv2.VideoCapture` cannot open your network stream, check that your OpenCV build has FFmpeg enabled.

## Performance Notes

What this setup does well:

- Uses a fixed-shape TensorRT 11 engine, which keeps execution predictable and avoids per-frame graph rebuilding.
- Preallocates pinned host buffers and device buffers, then reuses them for the whole run instead of allocating on every frame.
- Runs inference asynchronously with `execute_async_v3`, so TensorRT and the CUDA stream can overlap work where the driver allows it.
- Preserves the recurrent state tensors across frames, which is required for RVM quality and avoids resetting temporal context.
- Keeps the deployment path simple: one engine, one fixed input size, and no dynamic-shape overhead in the hot path.

What could be better:

- Frame preprocessing is still CPU-bound: resize, color conversion, normalization, and transpose all happen on the host every frame.
- Every frame still moves through host memory before inference, so there is room to reduce transfer overhead with a more GPU-centric pipeline.
- The runtime depends on OpenCV capture and decode, which is convenient but not the fastest option for high-throughput streams.
- The bridge path adds extra I/O and encode/decode work through `v4l2loopback` and `ffmpeg`, which is useful operationally but not optimal for raw latency.
- The repo ships an FP16 engine, but there is no INT8 path yet for users who want to trade calibration work for more throughput.
