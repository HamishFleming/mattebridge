import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ENGINE_PATH = "modnet.engine"
INPUT_SHAPE = (512, 512)

# -------------------------
# Load TensorRT engine
# -------------------------
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# -------------------------
# Allocate buffers (TensorRT 11 IO tensor API)
# -------------------------
inputs = []
outputs = []
stream = cuda.Stream()

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))

    size = trt.volume(shape)

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    mode = engine.get_tensor_mode(name)

    if mode == trt.TensorIOMode.INPUT:
        inputs.append((name, host_mem, device_mem))
    else:
        outputs.append((name, host_mem, device_mem))

# -------------------------
# Bind tensor addresses (TensorRT 11)
# -------------------------
for name, host_mem, device_mem in inputs:
    context.set_tensor_address(name, int(device_mem))

for name, host_mem, device_mem in outputs:
    context.set_tensor_address(name, int(device_mem))

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# -------------------------
# v4l2loopback output (IMPORTANT: init once)
# -------------------------
out_cam = cv2.VideoWriter(
    "/dev/video10",
    cv2.CAP_V4L2,
    cv2.VideoWriter_fourcc(*"YUYV"),
    30,
    (512, 512),
    True
)

if not out_cam.isOpened():
    raise RuntimeError("Cannot open /dev/video10 (v4l2loopback)")

print("MODNet running (TensorRT 11)...")

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, INPUT_SHAPE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # normalize
    inp = rgb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]

    # -------------------------
    # GPU input copy
    # -------------------------
    input_name, host_in, device_in = inputs[0]
    np.copyto(host_in, inp.ravel())

    cuda.memcpy_htod_async(device_in, host_in, stream)

    # -------------------------
    # inference (TRT 11)
    # -------------------------
    context.execute_async_v3(stream_handle=stream.handle)

    # -------------------------
    # GPU output copy
    # -------------------------
    output_name, host_out, device_out = outputs[0]

    cuda.memcpy_dtoh_async(host_out, device_out, stream)
    stream.synchronize()

    # -------------------------
    # postprocess
    # -------------------------
    matte = host_out.reshape(512, 512)

    fg = rgb.astype(np.float32) / 255.0
    out = fg * matte[..., None]
    out = (out * 255).astype(np.uint8)

    # -------------------------
    # output to virtual camera
    # -------------------------
    out_cam.write(out)

    # optional debug
    # cv2.imshow("MODNet", out)
    # if cv2.waitKey(1) == 27:
    #     break

# -------------------------
# cleanup
# -------------------------
cap.release()
out_cam.release()
cv2.destroyAllWindows()
