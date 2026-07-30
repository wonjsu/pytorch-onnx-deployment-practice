"""Run reproducible FP16 and explicit-Q/DQ INT8 experiments with pinned venvs."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FP32=Path("examples/yolo_onnx/artifacts/yolov8n.onnx")
FP16=Path("examples/yolo_fp16/artifacts/yolov8n_mixed_fp16.onnx")
INT8=Path("examples/yolo_int8/artifacts/yolov8n_int8_qdq.onnx")
ENGINES={"fp32":Path("examples/yolo_tensorrt/artifacts/yolov8n_fp32_strict.engine"),"fp16":Path("examples/yolo_tensorrt/artifacts/yolov8n_mixed_fp16.engine"),"int8":Path("examples/yolo_tensorrt/artifacts/yolov8n_int8.engine")}

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def artifact_is_current(path:Path, metadata_path:Path, source:Path, source_key="source_onnx_sha256")->bool:
 try:
  metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
  expected=sha256(source)
  return path.is_file() and path.stat().st_size>0 and metadata.get(source_key,metadata.get("source_sha256"))==expected
 except (OSError,ValueError,json.JSONDecodeError):return False
def python_path(kind:str)->Path:
 base=ROOT/(".venv-modelopt" if kind=="modelopt" else ".venv")
 windows=base/"Scripts/python.exe"; posix=base/"bin/python"
 return windows if windows.exists() else posix
def command(module:str,*args:object,environment="runtime")->list[str]:
 return [str(python_path(environment)),"-m",module,*map(str,args)]

def build_commands(a:argparse.Namespace)->list[tuple[str,list[str]]]:
 image=Path("input/custom/mouse.jpg"); val_images=Path("input/coco/images/val2017");ann=Path("input/coco/annotations/instances_val2017.json")
 limit=10 if a.scope=="smoke" else 5000; out=a.output_dir;cal=out/"calibration"
 jobs=[]
 if a.stage in ("fp16","all"):
  jobs += [("fp16_ort_compare",command("examples.yolo_fp16.compare_fp32_fp16_onnx","--image-path",image,"--fp32-onnx-path",FP32,"--fp16-onnx-path",FP16)),
   ("fp16_build",command("examples.yolo_tensorrt.build_engine","--onnx-path",FP16,"--engine-path",ENGINES["fp16"],"--model-precision","mixed-fp16","--tf32","off")),
   ("fp16_trt_compare",command("examples.yolo_tensorrt.compare_onnx_tensorrt","--image-path",image,"--onnx-path",FP16,"--engine-path",ENGINES["fp16"])),
   ("fp16_accuracy",command("examples.yolo_coco.evaluate_coco","--backend","tensorrt","--engine-path",ENGINES["fp16"],"--images-dir",val_images,"--annotation-path",ann,"--limit",limit,"--conf-threshold","0.001","--iou-threshold","0.7","--output-json",out/"fp16_predictions.json"))]
 if a.stage in ("int8","all"):
  if not a.calibration_images_dir or not a.calibration_annotation_path:raise ValueError("INT8 requires explicit calibration image and annotation paths; val2017 is never assumed")
  jobs += [("calibration",command("examples.yolo_int8.generate_calibration_data","--images-dir",a.calibration_images_dir,"--annotation-path",a.calibration_annotation_path,"--count",a.calibration_count,"--seed",a.calibration_seed,"--output-dir",cal,"--onnx-path",FP32)),
   ("int8_quantize",command("examples.yolo_int8.quantize_int8_modelopt","--onnx-path",FP32,"--output-path",INT8,"--calibration-data-dir",cal,"--calibration-method",a.calibration_method,environment="modelopt")),
   ("int8_inspect",command("examples.yolo_int8.inspect_int8_qdq_onnx","--onnx-path",INT8,"--output-json",str(INT8)+".inspection.json")),
   ("int8_ort_compare",command("examples.yolo_int8.compare_fp32_int8_onnx","--image-path",image,"--fp32-onnx-path",FP32,"--int8-onnx-path",INT8)),
   ("int8_build",command("examples.yolo_tensorrt.build_engine","--onnx-path",INT8,"--engine-path",ENGINES["int8"],"--model-precision","int8","--tf32","off")),
   ("int8_trt_compare",command("examples.yolo_tensorrt.compare_onnx_tensorrt","--image-path",image,"--onnx-path",INT8,"--engine-path",ENGINES["int8"])),
   ("int8_accuracy",command("examples.yolo_coco.evaluate_coco","--backend","tensorrt","--engine-path",ENGINES["int8"],"--images-dir",val_images,"--annotation-path",ann,"--limit",limit,"--conf-threshold","0.001","--iou-threshold","0.7","--output-json",out/"int8_predictions.json"))]
 labels=["fp32"]+(["fp16"] if a.stage in ("fp16","all") else [])+(["int8"] if a.stage in ("int8","all") else [])
 bench=["--mode","both"]
 for label in labels:bench += ["--engine",f"{label}={ENGINES[label]}"]
 if a.scope=="smoke":bench += ["--limit","100","--engine-warmup","5","--engine-iterations","20","--engine-rounds","2","--pipeline-rounds","2","--discard-rounds","1"]
 else:bench += ["--limit","5000","--engine-warmup","50","--engine-iterations","500","--engine-rounds","4","--pipeline-rounds","4","--discard-rounds","1"]
 bench += ["--images-dir",val_images,"--annotation-path",ann,"--conf-threshold","0.25","--iou-threshold","0.45","--output-json",out/f"benchmark_{a.scope}.json","--output-csv",out/f"benchmark_{a.scope}.csv"]
 jobs.append(("rotating_benchmark",command("examples.yolo_benchmark.benchmark_precision",*bench)))
 return jobs

def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("fp16","int8","all"),default="all");p.add_argument("--scope",choices=("smoke","full"),default="smoke");p.add_argument("--resume",action="store_true");p.add_argument("--force",action="store_true")
 p.add_argument("--calibration-images-dir",type=Path);p.add_argument("--calibration-annotation-path",type=Path);p.add_argument("--calibration-count",type=int,default=256);p.add_argument("--calibration-seed",type=int,default=0);p.add_argument("--calibration-method",choices=("entropy","max"),default="entropy");p.add_argument("--output-dir",type=Path,default=Path("precision-experiment-results"));a=p.parse_args(argv)
 if a.resume and a.force:p.error("--resume and --force are mutually exclusive")
 if a.calibration_count<=0:p.error("--calibration-count must be positive")
 return a
def main(argv=None):
 a=parse_args(argv);a.output_dir.mkdir(parents=True,exist_ok=True);logdir=a.output_dir/("logs_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));logdir.mkdir()
 jobs=build_commands(a);manifest=[]
 for index,(name,cmd) in enumerate(jobs,1):
  # Resume only skips artifacts whose source hash is proven by adjacent metadata.
  target={"fp16_build":ENGINES["fp16"],"int8_build":ENGINES["int8"],"int8_quantize":INT8}.get(name)
  source={"fp16_build":FP16,"int8_build":INT8,"int8_quantize":FP32}.get(name)
  if a.resume and target and source and artifact_is_current(target,Path(str(target)+(".conversion.json" if name=="int8_quantize" else ".json")),source):continue
  started=datetime.now(timezone.utc).isoformat();result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
  ended=datetime.now(timezone.utc).isoformat();(logdir/f"{index:02d}_{name}.stdout.log").write_text(result.stdout,encoding="utf-8");(logdir/f"{index:02d}_{name}.stderr.log").write_text(result.stderr,encoding="utf-8")
  manifest.append({"name":name,"command":cmd,"started_utc":started,"ended_utc":ended,"returncode":result.returncode})
  (logdir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
  if result.returncode:raise SystemExit(f"stage {name} failed ({result.returncode}); see {logdir}")
 print(f"Completed {a.scope} protocol. Logs: {logdir}")
 if a.scope=="smoke":print("Smoke measurements are validation only and must not be published as final performance results.")
 else:print("Full raw results are ready for reviewed documentation update; existing FP32 files were not overwritten.")
if __name__=="__main__":main()
