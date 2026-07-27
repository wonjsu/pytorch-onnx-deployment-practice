"""Generate deterministic real-image reference inputs for ModelOpt AutoCast (not INT8 calibration)."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess

def select_images(images: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0: raise ValueError('count must be positive')
    ordered=sorted(images,key=lambda x:int(x['id']))
    if count>len(ordered): raise ValueError(f'count {count} exceeds {len(ordered)} available images')
    return sorted(random.Random(seed).sample(ordered,count),key=lambda x:int(x['id']))

def onnx_input_name(path: Path) -> str:
    import onnx
    model=onnx.load(str(path),load_external_data=False)
    if len(model.graph.input)!=1: raise ValueError(f'Expected one ONNX input, got {len(model.graph.input)}')
    return model.graph.input[0].name

def generate(onnx_path:Path,images_dir:Path,annotation_path:Path,output_dir:Path,count:int,seed:int)->dict:
    data=json.loads(annotation_path.read_text(encoding='utf-8')); selected=select_images(data.get('images',[]),count,seed)
    name=onnx_input_name(onnx_path); output_dir.mkdir(parents=True,exist_ok=True); records=[]
    for i,item in enumerate(selected):
        image_path=images_dir/item['file_name']
        with Image.open(image_path) as source: tensor,*_=letterbox_preprocess(source.convert('RGB'))
        if tensor.shape!=(1,3,640,640) or tensor.dtype!=np.float32: raise ValueError(f'Invalid tensor: {tensor.shape} {tensor.dtype}')
        target=output_dir/f'batch_{i:04d}.npz'; np.savez(target,**{name:tensor})
        records.append({'image_id':int(item['id']),'file_name':item['file_name'],'npz':target.name})
    metadata={'purpose':'ModelOpt AutoCast FP16 sensitivity reference input; not INT8 calibration','seed':seed,'count':count,'onnx_path':str(onnx_path),'input_name':name,'tensor_shape':[1,3,640,640],'tensor_dtype':'float32','normalization':'RGB 0..1','images':records}
    (output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8'); return metadata

def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--onnx-path',type=Path,default=Path('examples/yolo_onnx/artifacts/yolov8n.onnx')); p.add_argument('--images-dir',type=Path,default=Path('input/coco/images/val2017')); p.add_argument('--annotation-path',type=Path,default=Path('input/coco/annotations/instances_val2017.json')); p.add_argument('--output-dir',type=Path,default=Path('examples/yolo_fp16/artifacts/autocast_data')); p.add_argument('--count',type=int,default=32); p.add_argument('--seed',type=int,default=0); return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv); print(json.dumps(generate(a.onnx_path,a.images_dir,a.annotation_path,a.output_dir,a.count,a.seed),indent=2))
if __name__=='__main__': main()
