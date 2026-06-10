import torch

# disable all tracing systems that break RVM
torch.set_grad_enabled(False)
torch._dynamo.disable()

from model import MattingNetwork  # we define this locally below

# load model
model = MattingNetwork("mobilenetv3").eval()
model.load_state_dict(torch.load("rvm_mobilenetv3.pth", map_location="cpu"))

dummy = torch.randn(1, 3, 512, 512)

""" torch.onnx.export( """
"""     model, """
"""     (dummy, None, None, None), """
"""     "rvm.onnx", """
"""     opset_version=12, """
"""     input_names=["src", "r1", "r2", "r3"], """
"""     output_names=["fgr", "pha", "r1o", "r2o", "r3o"], """
"""     do_constant_folding=True, """
"""     dynamic_axes=None  # IMPORTANT: remove dynamic axes if possible """
""" ) """
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None  # IMPORTANT: remove dynamic axes if possible
)


print("Export complete -> rvm.onnx")
