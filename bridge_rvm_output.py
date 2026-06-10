#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RVM_RUNTIME = SCRIPT_DIR / "rvm_runtime.py"
VIDEO_DEVICE = "/dev/video10"


def build_input_args(args: argparse.Namespace) -> list[str]:
    if args.input_camera is not None:
        input_args = ["--input-camera", str(args.input_camera)]
    elif args.input_rtsp is not None:
        input_args = ["--input-rtsp", args.input_rtsp]
    elif args.input_udp is not None:
        input_args = ["--input-udp", str(args.input_udp)]
    else:
        input_args = ["--input-camera", "0"]

    if args.display:
        input_args.append("--display")
    return input_args


def build_ffmpeg_cmd(args: argparse.Namespace) -> list[str]:
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "v4l2",
        "-input_format",
        "yuyv422",
        "-framerate",
        str(args.fps),
        "-video_size",
        f"{args.width}x{args.height}",
        "-i",
        VIDEO_DEVICE,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
    ]

    if args.output_udp is not None:
        return base + ["-f", "mpegts", f"udp://{args.output_udp}"]
    if args.output_rtsp is not None:
        return base + ["-f", "rtsp", "-rtsp_transport", "tcp", args.output_rtsp]
    raise RuntimeError("No output target configured")


def wait_for_video_device(path: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.25)
    raise FileNotFoundError(
        f"{path} did not appear. Load v4l2loopback and make sure rvm_runtime.py is running."
    )


def terminate_process(proc: subprocess.Popen[object] | None, timeout_s: int = 5) -> None:
    if proc is None or proc.poll() is not None:
        return

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            continue


def poll_processes(
    rvm_proc: subprocess.Popen[object],
    ffmpeg_proc: subprocess.Popen[object],
    poll_interval_s: float = 0.25,
) -> int:
    while True:
        rvm_rc = rvm_proc.poll()
        ffmpeg_rc = ffmpeg_proc.poll()

        if rvm_rc is not None:
            if ffmpeg_rc is None:
                terminate_process(ffmpeg_proc, timeout_s=10)
            return rvm_rc

        if ffmpeg_rc is not None:
            terminate_process(rvm_proc, timeout_s=10)
            return 1

        time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-camera", type=int, help="Local webcam device index")
    input_group.add_argument("--input-rtsp", type=str, help="RTSP stream URL")
    input_group.add_argument("--input-udp", type=int, help="UDP port to listen on")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a local preview window with the processed output",
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output-udp",
        type=str,
        help="UDP destination in host:port form, for example 192.168.1.50:6000",
    )
    output_group.add_argument(
        "--output-rtsp",
        type=str,
        help="RTSP publish URL, for example rtsp://192.168.1.50:8554/rvm",
    )

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device-timeout", type=int, default=10)
    args = parser.parse_args()

    if not RVM_RUNTIME.exists():
        raise FileNotFoundError(f"Missing runtime script: {RVM_RUNTIME}")

    rvm_proc = subprocess.Popen(
        [sys.executable, str(RVM_RUNTIME), *build_input_args(args)],
        start_new_session=True,
    )
    ffmpeg_proc = None

    try:
        wait_for_video_device(VIDEO_DEVICE, args.device_timeout)
        time.sleep(0.5)
        ffmpeg_proc = subprocess.Popen(build_ffmpeg_cmd(args), start_new_session=True)
        rc = poll_processes(rvm_proc, ffmpeg_proc)
        raise SystemExit(rc)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_process(ffmpeg_proc)
        terminate_process(rvm_proc)


if __name__ == "__main__":
    main()
