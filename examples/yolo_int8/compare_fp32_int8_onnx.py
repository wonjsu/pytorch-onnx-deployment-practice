"""Compare FP32 and explicit-Q/DQ INT8 ONNX outputs with ORT CUDA."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from examples.yolo_fp16.compare_fp32_fp16_onnx import main as _main
def main(argv=None):
    values=list(sys.argv[1:] if argv is None else argv)
    values=["--fp16-onnx-path" if x=="--int8-onnx-path" else x for x in values]
    return _main(values)
if __name__=="__main__":main()
