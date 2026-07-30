"""Quantize FP32 YOLO ONNX with ModelOpt's supported lazy-reader API."""
from __future__ import annotations
import argparse, importlib, inspect, json, os, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from examples.yolo_int8.generate_calibration_data import sha256
from examples.yolo_int8.inspect_int8_qdq_onnx import inspect_model

class LazyNpzCalibrationDataReader:
    def __init__(self,directory:Path):
        self.directory=directory; self.files=sorted(directory.glob("batch_*.npz")); self._index=0
        if not self.files: raise ValueError(f"no calibration batches in {directory}")
    def get_next(self):
        if self._index>=len(self.files): return None
        import numpy as np
        path=self.files[self._index];self._index+=1
        with np.load(path) as data:return {key:data[key].copy() for key in data.files}
    def rewind(self):self._index=0

def quantize(source:Path, output:Path, calibration_dir:Path, method:str) -> dict:
    metadata=json.loads((calibration_dir/"metadata.json").read_text(encoding="utf-8"))
    if metadata["fp32_onnx_sha256"]!=sha256(source):raise ValueError("calibration metadata source SHA-256 does not match ONNX")
    module=importlib.import_module("modelopt.onnx.quantization"); function=getattr(module,"quantize")
    signature=inspect.signature(function); supported=set(signature.parameters)
    requested={"onnx_path":str(source),"quantize_mode":"int8","calibration_method":method,"high_precision_dtype":"fp16",
               "calibration_data_reader":LazyNpzCalibrationDataReader(calibration_dir)}
    missing=[x for x in requested if x not in supported]
    output_key=next((x for x in ("output_path","output_model_path") if x in supported),None)
    if missing or output_key is None:raise RuntimeError(f"Unsupported ModelOpt quantize signature {signature}; missing supported parameters: {missing}, output_path")
    output.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(suffix=".onnx",dir=output.parent);os.close(fd);os.unlink(tmp)
    kwargs={k:v for k,v in requested.items()};kwargs[output_key]=tmp;started=time.perf_counter()
    try:
        function(**kwargs); inspection=inspect_model(Path(tmp));os.replace(tmp,output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    import numpy,onnx,modelopt
    result={"source_sha256":sha256(source),"output_sha256":sha256(output),"modelopt_version":getattr(modelopt,"__version__","unknown"),"onnx_version":onnx.__version__,"numpy_version":numpy.__version__,
      "quantize_mode":"int8","calibration_method":method,"calibration_count":metadata["count"],"calibration_seed":metadata["seed"],"calibration_image_ids":metadata["image_ids"],
      "high_precision_fallback_dtype":"fp16","external_io_dtype":"FP32","conversion_duration_seconds":time.perf_counter()-started,"qdq_inspection":inspection}
    Path(str(output)+".conversion.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--onnx-path",type=Path,default=Path("examples/yolo_onnx/artifacts/yolov8n.onnx"));p.add_argument("--output-path",type=Path,default=Path("examples/yolo_int8/artifacts/yolov8n_int8_qdq.onnx"));p.add_argument("--calibration-data-dir",type=Path,required=True);p.add_argument("--calibration-method",choices=("entropy","max"),default="entropy");a=p.parse_args(argv);print(json.dumps(quantize(a.onnx_path,a.output_path,a.calibration_data_dir,a.calibration_method),indent=2))
if __name__=="__main__":main()
