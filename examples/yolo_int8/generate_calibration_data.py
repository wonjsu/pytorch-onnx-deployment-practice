"""Create deterministic, streaming ModelOpt calibration batches from COCO images."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def select_images(annotation_path: Path, count: int, seed: int) -> list[dict]:
    images=json.loads(annotation_path.read_text(encoding="utf-8"))["images"]
    if count <= 0: raise ValueError("count must be positive")
    if count > len(images): raise ValueError(f"requested {count} images, dataset contains {len(images)}")
    rng=np.random.default_rng(seed)
    return [images[i] for i in rng.choice(len(images), count, replace=False)]

def generate(images_dir: Path, annotation_path: Path, count: int, seed: int,
             output_dir: Path, onnx_path: Path) -> dict:
    import onnx
    model=onnx.load(str(onnx_path), load_external_data=False)
    if len(model.graph.input)!=1: raise ValueError("calibration ONNX must have exactly one input")
    input_name=model.graph.input[0].name; selected=select_images(annotation_path,count,seed)
    output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for index,item in enumerate(selected):
        source=images_dir/item["file_name"]
        with Image.open(source) as image: tensor,*_=letterbox_preprocess(image.convert("RGB"))
        if tensor.shape!=(1,3,640,640) or tensor.dtype!=np.float32: raise RuntimeError("unexpected preprocessing result")
        filename=f"batch_{index:04d}.npz"; np.savez(output_dir/filename, **{input_name:tensor})
        records.append({"image_id":int(item["id"]),"file_name":item["file_name"],"batch_file":filename})
    overlap="val2017" in str(images_dir).lower() or "val2017" in str(annotation_path).lower()
    metadata={"purpose":"ModelOpt INT8 PTQ calibration (not evaluation)","source":{"images_dir":str(images_dir),"annotation_path":str(annotation_path)},
      "seed":seed,"count":count,"images":records,"image_ids":[x["image_id"] for x in records],"input_name":input_name,
      "shape":[1,3,640,640],"dtype":"float32","preprocessing":"RGB letterbox 640x640, NCHW, float32 0..1, padding 114",
      "fp32_onnx_path":str(onnx_path),"fp32_onnx_sha256":sha256(onnx_path),"calibration_evaluation_overlap_possible":overlap}
    (output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--images-dir",type=Path,required=True);p.add_argument("--annotation-path",type=Path,required=True)
    p.add_argument("--count",type=int,default=256);p.add_argument("--seed",type=int,default=0);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--onnx-path",type=Path,default=Path("examples/yolo_onnx/artifacts/yolov8n.onnx"));return p.parse_args(argv)
def main(argv=None):
    a=parse_args(argv); print(json.dumps(generate(a.images_dir,a.annotation_path,a.count,a.seed,a.output_dir,a.onnx_path),indent=2))
if __name__=="__main__": main()
