from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "RobustVideoMatting"
ONNX_FILE = REPO_ROOT / "rvm_simplified.onnx"
CKPT_FILE = REPO_ROOT / "rvm_mobilenetv3.pth"

sys.path.insert(0, str(MODEL_ROOT))

from model import MattingNetwork


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, src, r1i, r2i, r3i, r4i):
        fgr, pha, r1o, r2o, r3o, r4o = self.model(
            src,
            r1i,
            r2i,
            r3i,
            r4i,
            1.0,
            False,
        )
        return fgr, pha, r1o, r2o, r3o, r4o


def main() -> None:
    torch.set_grad_enabled(False)
    if hasattr(torch, "_dynamo"):
        torch._dynamo.disable()

    model = MattingNetwork("mobilenetv3").eval()
    model.load_state_dict(torch.load(CKPT_FILE, map_location="cpu"))
    wrapper = ExportWrapper(model).eval()

    dummy_inputs = (
        torch.randn(1, 3, 512, 512),
        torch.zeros(1, 16, 256, 256),
        torch.zeros(1, 20, 128, 128),
        torch.zeros(1, 40, 64, 64),
        torch.zeros(1, 64, 32, 32),
    )

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        ONNX_FILE.as_posix(),
        opset_version=17,
        input_names=["src", "r1i", "r2i", "r3i", "r4i"],
        output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported {ONNX_FILE}")


if __name__ == "__main__":
    main()
