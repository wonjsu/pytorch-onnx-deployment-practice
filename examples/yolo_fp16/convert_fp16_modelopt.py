"""Convert FP32 ONNX to mixed FP32/FP16 with ModelOpt AutoCast, atomically."""
from __future__ import annotations
import argparse, importlib.metadata, inspect, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.yolo_fp16.inspect_mixed_precision_onnx import inspect_model, sha256

def validate_paths(source:Path, output:Path, force:bool=False)->None:
    source,output=Path(source).resolve(),Path(output).resolve()
    if not source.is_file() or source.stat().st_size==0: raise ValueError(f'Input ONNX missing or empty: {source}')
    if source==output: raise ValueError('Refusing to overwrite the original ONNX model')
    if output.exists() and not force: raise FileExistsError(f'Output exists; pass --force to replace it: {output}')

def load_calibration_data(path:Path)->list[dict[str,np.ndarray]]:
    path=Path(path); files=sorted(path.glob('*.npz')) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files): raise ValueError(f'No NPZ calibration/reference batches found: {path}')
    batches=[]
    for file in files:
        with np.load(file) as values: batches.append({k:values[k] for k in values.files})
    return batches

def _version(name:str)->str:
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return 'unknown'

def convert(source:Path, output:Path, calibration_path:Path, providers:list[str], opset:int|None=None, force:bool=False)->dict:
    validate_paths(source,output,force); output.parent.mkdir(parents=True,exist_ok=True)
    batches=load_calibration_data(calibration_path)
    try: from modelopt.onnx.autocast import convert_to_mixed_precision
    except ImportError as exc: raise RuntimeError('ModelOpt AutoCast is unavailable. Use the separate requirements-modelopt.txt environment.') from exc
    signature=inspect.signature(convert_to_mixed_precision)
    kwargs={'onnx_path':str(source),'low_precision_type':'fp16','keep_io_types':True,'calibration_data':batches,'providers':providers}
    unsupported=[k for k in kwargs if k not in signature.parameters]
    if unsupported: raise RuntimeError(f'Installed ModelOpt AutoCast API does not support required arguments {unsupported}; signature is {signature}')
    if opset is not None:
        candidate=next((x for x in ('opset','target_opset') if x in signature.parameters),None)
        if candidate is None: raise RuntimeError(f'--opset requested but installed API has no opset parameter: {signature}')
        kwargs[candidate]=opset
    fd,tmp_name=tempfile.mkstemp(prefix=output.stem+'.',suffix='.tmp.onnx',dir=output.parent); os.close(fd); tmp=Path(tmp_name); tmp.unlink()
    started=time.perf_counter()
    try:
        converted=convert_to_mixed_precision(**kwargs)
        import onnx
        if isinstance(converted,onnx.ModelProto): onnx.save_model(converted,str(tmp))
        elif isinstance(converted,(str,Path)):
            produced=Path(converted)
            if not produced.is_file(): raise RuntimeError(f'ModelOpt returned a missing path: {produced}')
            os.replace(produced,tmp)
        elif converted is None and output.is_file(): raise RuntimeError('ModelOpt unexpectedly wrote the final output path; refusing non-atomic output')
        else: raise RuntimeError(f'Unsupported ModelOpt return type: {type(converted).__name__}')
        inspection=inspect_model(tmp); elapsed=time.perf_counter()-started
        metadata={'conversion':'ModelOpt AutoCast mixed FP32/FP16','modelopt_version':_version('nvidia-modelopt'),'onnx_version':onnx.__version__,'providers':providers,'calibration_data':str(calibration_path),'calibration_batch_count':len(batches),'conversion_seconds':elapsed,'requested_opset':opset,'source_onnx_path':str(source),'source_onnx_sha256':sha256(source),'output_onnx_path':str(output),'output_onnx_sha256':inspection['sha256'],'inputs':inspection['inputs'],'outputs':inspection['outputs'],'opsets':inspection['opsets']}
        os.replace(tmp,output)
        metadata['output_onnx_sha256']=sha256(output)
        Path(str(output)+'.conversion.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return metadata
    except Exception:
        tmp.unlink(missing_ok=True); raise

def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--onnx-path',type=Path,default=Path('examples/yolo_onnx/artifacts/yolov8n.onnx')); p.add_argument('--output-path',type=Path,default=Path('examples/yolo_fp16/artifacts/yolov8n_mixed_fp16.onnx')); p.add_argument('--calibration-data',type=Path,required=True); p.add_argument('--providers',nargs='+',default=['cpu']); p.add_argument('--opset',type=int); p.add_argument('--force',action='store_true'); return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv); print(json.dumps(convert(a.onnx_path,a.output_path,a.calibration_data,a.providers,a.opset,a.force),indent=2))
if __name__=='__main__':main()
