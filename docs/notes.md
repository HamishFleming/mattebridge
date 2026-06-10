# Notes

This repo is a TensorRT runtime wrapper around the `RobustVideoMatting` MobileNetV3 checkpoint, plus a separate MODNet runtime for comparison and fallback testing.

## Model Artifacts

- `rvm_mobilenetv3.pth`: upstream PyTorch checkpoint from Robust Video Matting.
- `rvm_simplified.onnx`: clean ONNX export from the RVM checkpoint, used as the TensorRT build input.
- `rvm_fp16.engine`: serialized TensorRT engine built from `rvm_simplified.onnx`.
- `rvm_mobilenetv3_fp16.onnx`: earlier ONNX export kept for reference.
- `rvm.onnx`: legacy export artifact kept for comparison.
- `modnet_photographic_portrait_matting.onnx`: separate MODNet model used by `modnet_runtime.py`.

## Source Material

- `RobustVideoMatting/` is the vendored upstream project snapshot used to inspect the model code and export path.
- `export_clean_onnx.py` is the export path to use when rebuilding the ONNX model.
- `build_engine.py` builds the TensorRT engine from the cleaned ONNX export.

## Rebuild Flow

1. Export ONNX:

   ```bash
   python export_clean_onnx.py
   ```

2. Build the TensorRT engine:

   ```bash
   python build_engine.py
   ```

3. Run the runtime:

   ```bash
   python rvm_runtime.py --input-camera 0
   ```

## Runtime Files

- `rvm_runtime.py`: TensorRT runtime with recurrent state handling.
- `bridge_rvm_output.py`: runs the runtime and republishes the result to `/dev/video10` or RTSP/UDP.
- `rvm_remove_background_video.py`: batch processing for video files.
- `modnet_runtime.py`: separate TensorRT runtime for the MODNet ONNX model.

## Performance Notes

What this setup does well:

- Fixed input shapes keep the TensorRT path stable and avoid dynamic-shape overhead.
- Host and device buffers are allocated once and reused, which keeps the hot loop lean.
- The runtime uses asynchronous CUDA execution and carries RVM recurrent state forward frame to frame.
- The engine is built from the cleaned ONNX export, so there is less exporter noise in the inference path.

What could be better:

- CPU preprocessing is still doing resize, normalization, channel reorder, and transpose for every frame.
- Input frames still cross host memory before they reach TensorRT.
- Capture and bridge I/O are done with OpenCV and `ffmpeg`, which are practical but not the lowest-latency choices.
- Only the FP16 engine is documented here; an INT8 path could improve throughput if calibration is worth the extra work.





 Deleted docs/notes.md (+0 -136)
      1 -
      2 -
      3 -
      4 -
      5 -
      6 -.rw-r--r--  hamish hamish  6 Jun 21:53  󰡯 activate
      7 -.rw-r--r--  hamish hamish  6 Jun 21:53   activate.csh
      8 -.rw-r--r--  hamish hamish  6 Jun 21:53   activate.fish
      9 -.rw-r--r--  hamish hamish  6 Jun 21:53   Activate.ps1
     10 -.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 backend-test-tools
     11 -.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 check-model
     12 -.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 check-node
     13 -.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 coloredlogs
     14 -.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 f2py
     15 -.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 humanfriendly
     16 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 imageio_download_bin
     17 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 imageio_remove_bin
     18 -.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 isympy
     19 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 lsm2bin
     20 -.rwxr-xr-x  hamish hamish  6 Jun 21:54  󰡯 mako-render
     21 -.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 markdown-it
     22 -.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 numpy-config
     23 -.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 onnxruntime_test
     24 -.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 onnxsim
     25 -.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 pip
     26 -.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 pip3
     27 -.rwxr-xr-x  hamish hamish  6 Jun 21:53   pip3.10
     28 -.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 proton
     29 -.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 proton-viewer
     30 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 pyav
     31 -.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 pygmentize
     32 -lrwxrwxrwx  hamish hamish  6 Jun 21:53  󰡯 python -> /mnt/code/ai/ai-stream/venv/bin/python
     33 -lrwxrwxrwx  hamish hamish  6 Jun 21:53  󰡯 python3 -> python
     34 -lrwxrwxrwx  hamish hamish  6 Jun 21:53   python3.10 -> python
     35 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tiff2fsspec
     36 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tiffcomment
     37 -.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tifffile
     38 -.rwxr-xr-x  hamish hamish  6 Jun 21:58  󰡯 torchfrtrace
     39 -.rwxr-xr-x  hamish hamish  6 Jun 21:58  󰡯 torchrun
     40 -.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 tqdm
     41 -
     42 -
     43 -
     44 -
     45 -
     46 -
     47 -
     48 -
     49 -
     50 -
     51 -
     52 -
     53 -
     54 -6 directories, 18 files
     55 -❯ tree -L 2 .
     56 -.
     57 -├── bridge_rvm_output.py
     58 -├── build_engine.py
     59 -├── docker-compose.yml
     60 -├── docs
     61 -│   └── notes.md
     62 -├── export_clean_onnx.py
     63 -├── export_onnx.py
     64 -├── mediamtx.yml
     65 -├── modnet_photographic_portrait_matting.onnx
     66 -├── modnet_runtime.py
     67 -├── __pycache__
     68 -│   ├── bridge_rvm_output.cpython-310.pyc
     69 -│   ├── build_engine.cpython-310.pyc
     70 -│   ├── export_onnx.cpython-310.pyc
     71 -│   ├── modnet_runtime.cpython-310.pyc
     72 -│   ├── rvm_remove_background_video.cpython-310.pyc
     73 -│   ├── rvm_runtime.cpython-310.pyc
     74 -│   └── rvm_trt.cpython-310.pyc
     75 -├── README.md
     76 -├── RobustVideoMatting
     77 -│   ├── 593ee8d4-02f7-481f-82d5-1a181842fa9d?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-06-06T12:43:41Z&rscd=attachment;+file
         name=rvm_mobilenetv3_fp16.onnx&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654
     78 -│   ├── dataset
     79 -│   ├── documentation
     80 -│   ├── evaluation
     81 -│   ├── export_clean_onnx.py
     82 -│   ├── export_onnx.py
     83 -│   ├── export_onnx.py_
     84 -│   ├── export_onnx.py_0
     85 -│   ├── hubconf.py
     86 -│   ├── inference.py
     87 -│   ├── inference_speed_test.py
     88 -│   ├── inference_utils.py
     89 -│   ├── LICENSE
     90 -│   ├── model
     91 -│   ├── __pycache__
     92 -│   ├── README.md
     93 -│   ├── README_zh_Hans.md
     94 -│   ├── requirements_inference.txt
     95 -│   ├── requirements_training.txt
     96 -│   ├── rvm_mobilenetv3_1280x720_s0.375_int8.mlmodel
     97 -│   ├── rvm_mobilenetv3_fp16.onnx
     98 -│   ├── rvm_mobilenetv3.pth
     99 -│   ├── rvm_mobilenetv3.pth_
    100 -│   ├── train_config.py
    101 -│   ├── train_loss.py
    102 -│   └── train.py
    103 -├── rvm_fp16.engine
    104 -├── rvm_mobilenetv3_fp16.onnx
    105 -├── rvm_mobilenetv3.pth
    106 -├── rvm.onnx
    107 -├── rvm_remove_background_video.py
    108 -├── rvm_runtime.py
    109 -├── rvm_simplified.onnx
    110 -├── rvm_trt.py
    111 -├── start_bridge_rvm_output.sh
    112 -├── tensorrt
    113 -│   ├── 010-tensorrt-use-local-protobuf-sources.patch
    114 -│   ├── 020-tensorrt-fix-python.patch
    115 -│   ├── build.log
    116 -│   ├── cub-nvlabs
    117 -│   ├── onnx
    118 -│   ├── onnx-tensorrt
    119 -│   ├── pkg
    120 -│   ├── PKGBUILD
    121 -│   ├── protobuf
    122 -│   ├── protobuf-cpp-3.20.3.tar.gz
    123 -│   ├── pybind11
    124 -│   ├── src
    125 -│   ├── TensorRT
    126 -│   ├── TensorRT-Enterprise-11.0.0.114-Linux-x86_64-cuda-13.2-Release-external.tar.zst
    127 -│   └── TensorRT-LICENSE-AGREEMENT.txt
    128 -└── venv
    129 -    ├── bin
    130 -    ├── include
    131 -    ├── lib
    132 -    ├── lib64 -> lib
    133 -    ├── pyvenv.cfg
    134 -    └── share
    135 -
    136 -24 directories, 55 files
