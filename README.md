# RVM TensorRT Webcam Runtime

This repo contains a small TensorRT webcam runtime for the Robust Video Matting model.
It reads frames from a local webcam or a network stream, runs the matting engine, and writes the result to a v4l2loopback device.

## Files

- `rvm_runtime.py`: TensorRT 11 runtime with recurrent state handling.
- `bridge_rvm_output.py`: helper that runs the runtime and republishes `/dev/video10`.
- `rvm_remove_background_video.py`: processes a video file and writes a new video with the background removed.
- `rvm_fp16.engine`: TensorRT engine file expected by the runtime.
- `modnet_runtime.py`: separate runtime for the MODNet ONNX/TensorRT model.

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
- If `cv2.VideoCapture` cannot open your network stream, check that your OpenCV build has FFmpeg enabled.
