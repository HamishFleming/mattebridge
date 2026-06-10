




.rw-r--r--  hamish hamish  6 Jun 21:53  󰡯 activate
.rw-r--r--  hamish hamish  6 Jun 21:53   activate.csh
.rw-r--r--  hamish hamish  6 Jun 21:53   activate.fish
.rw-r--r--  hamish hamish  6 Jun 21:53   Activate.ps1
.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 backend-test-tools
.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 check-model
.rwxr-xr-x  hamish hamish  6 Jun 22:15  󰡯 check-node
.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 coloredlogs
.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 f2py
.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 humanfriendly
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 imageio_download_bin
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 imageio_remove_bin
.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 isympy
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 lsm2bin
.rwxr-xr-x  hamish hamish  6 Jun 21:54  󰡯 mako-render
.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 markdown-it
.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 numpy-config
.rwxr-xr-x  hamish hamish  7 Jun 21:43  󰡯 onnxruntime_test
.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 onnxsim
.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 pip
.rwxr-xr-x  hamish hamish  6 Jun 21:53  󰡯 pip3
.rwxr-xr-x  hamish hamish  6 Jun 21:53   pip3.10
.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 proton
.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 proton-viewer
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 pyav
.rwxr-xr-x  hamish hamish  7 Jun 21:31  󰡯 pygmentize
lrwxrwxrwx  hamish hamish  6 Jun 21:53  󰡯 python -> /mnt/code/ai/ai-stream/venv/bin/python
lrwxrwxrwx  hamish hamish  6 Jun 21:53  󰡯 python3 -> python
lrwxrwxrwx  hamish hamish  6 Jun 21:53   python3.10 -> python
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tiff2fsspec
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tiffcomment
.rwxr-xr-x  hamish hamish  6 Jun 22:16  󰡯 tifffile
.rwxr-xr-x  hamish hamish  6 Jun 21:58  󰡯 torchfrtrace
.rwxr-xr-x  hamish hamish  6 Jun 21:58  󰡯 torchrun
.rwxr-xr-x  hamish hamish  6 Jun 21:57  󰡯 tqdm













6 directories, 18 files
❯ tree -L 2 .
.
├── bridge_rvm_output.py
├── build_engine.py
├── docker-compose.yml
├── docs
│   └── notes.md
├── export_clean_onnx.py
├── export_onnx.py
├── mediamtx.yml
├── modnet_photographic_portrait_matting.onnx
├── modnet_runtime.py
├── __pycache__
│   ├── bridge_rvm_output.cpython-310.pyc
│   ├── build_engine.cpython-310.pyc
│   ├── export_onnx.cpython-310.pyc
│   ├── modnet_runtime.cpython-310.pyc
│   ├── rvm_remove_background_video.cpython-310.pyc
│   ├── rvm_runtime.cpython-310.pyc
│   └── rvm_trt.cpython-310.pyc
├── README.md
├── RobustVideoMatting
│   ├── 593ee8d4-02f7-481f-82d5-1a181842fa9d?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-06-06T12:43:41Z&rscd=attachment;+filename=rvm_mobilenetv3_fp16.onnx&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654
│   ├── dataset
│   ├── documentation
│   ├── evaluation
│   ├── export_clean_onnx.py
│   ├── export_onnx.py
│   ├── export_onnx.py_
│   ├── export_onnx.py_0
│   ├── hubconf.py
│   ├── inference.py
│   ├── inference_speed_test.py
│   ├── inference_utils.py
│   ├── LICENSE
│   ├── model
│   ├── __pycache__
│   ├── README.md
│   ├── README_zh_Hans.md
│   ├── requirements_inference.txt
│   ├── requirements_training.txt
│   ├── rvm_mobilenetv3_1280x720_s0.375_int8.mlmodel
│   ├── rvm_mobilenetv3_fp16.onnx
│   ├── rvm_mobilenetv3.pth
│   ├── rvm_mobilenetv3.pth_
│   ├── train_config.py
│   ├── train_loss.py
│   └── train.py
├── rvm_fp16.engine
├── rvm_mobilenetv3_fp16.onnx
├── rvm_mobilenetv3.pth
├── rvm.onnx
├── rvm_remove_background_video.py
├── rvm_runtime.py
├── rvm_simplified.onnx
├── rvm_trt.py
├── start_bridge_rvm_output.sh
├── tensorrt
│   ├── 010-tensorrt-use-local-protobuf-sources.patch
│   ├── 020-tensorrt-fix-python.patch
│   ├── build.log
│   ├── cub-nvlabs
│   ├── onnx
│   ├── onnx-tensorrt
│   ├── pkg
│   ├── PKGBUILD
│   ├── protobuf
│   ├── protobuf-cpp-3.20.3.tar.gz
│   ├── pybind11
│   ├── src
│   ├── TensorRT
│   ├── TensorRT-Enterprise-11.0.0.114-Linux-x86_64-cuda-13.2-Release-external.tar.zst
│   └── TensorRT-LICENSE-AGREEMENT.txt
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share

24 directories, 55 files
