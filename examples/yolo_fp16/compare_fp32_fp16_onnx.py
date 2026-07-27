"""Compare original FP32 and ModelOpt mixed-FP16 ONNX using ORT CUDA."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from examples.yolo_onnx.postprocess_onnx import letterbox_preprocess_image,postprocess_output
from examples.yolo_tensorrt.compare_backends import Detection,match_detections

def _session(path):
 import onnxruntime as ort
 if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDAExecutionProvider unavailable; CPU fallback is forbidden')
 s=ort.InferenceSession(str(path),providers=['CUDAExecutionProvider'])
 if 'CUDAExecutionProvider' not in s.get_providers():raise RuntimeError('ORT rejected CUDA; CPU fallback is forbidden')
 return s

def _detections(raw,meta,conf,iou):
 c,s,b=postprocess_output(raw,*meta,conf,iou);return [Detection(int(x),float(y),tuple(map(float,z))) for x,y,z in zip(c,s,b)]
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image-path',type=Path,required=True);p.add_argument('--fp32-onnx-path',type=Path,required=True);p.add_argument('--fp16-onnx-path',type=Path,required=True);p.add_argument('--conf-threshold',type=float,default=.25);p.add_argument('--iou-threshold',type=float,default=.45);p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-2);a=p.parse_args(argv)
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
 tensor,size,ratio,px,py=letterbox_preprocess_image(a.image_path); sessions=[_session(x) for x in (a.fp32_onnx_path,a.fp16_onnx_path)]
 values=[]
 for s in sessions: values.append({o.name:v for o,v in zip(s.get_outputs(),s.run(None,{s.get_inputs()[0].name:tensor}))})
 if list(values[0])!=list(values[1]):raise RuntimeError(f'Output names differ: {list(values[0])} vs {list(values[1])}')
 for name in values[0]:
  x,y=values[0][name],values[1][name]
  if x.shape!=y.shape or x.dtype!=y.dtype:raise RuntimeError(f'{name} shape/dtype differ: {x.shape}/{x.dtype}, {y.shape}/{y.dtype}')
  diff=np.abs(x-y); rel=diff/np.maximum(np.abs(x),np.finfo(np.float32).eps); idx=np.unravel_index(int(np.argmax(diff)),diff.shape)
  print(f'{name}: shape={x.shape} dtype={x.dtype} max_abs={diff.max():.9g} mean_abs={diff.mean():.9g} RMSE={np.sqrt(np.mean(diff**2)):.9g} relative_mean={rel.mean():.9g} relative_max={rel.max():.9g} allclose={np.allclose(x,y,atol=a.atol,rtol=a.rtol)} max_index={idx} fp32={x[idx]:.9g} mixed_fp16={y[idx]:.9g}')
  meta=(size,ratio,px,py);d0=_detections(x,meta,a.conf_threshold,a.iou_threshold);d1=_detections(y,meta,a.conf_threshold,a.iou_threshold);matches,u0,u1=match_detections(d0,d1,.5)
  print(f'detections: matched={len(matches)} unmatched_fp32={len(u0)} unmatched_fp16={len(u1)}')
  for m in matches:
   q,r=d0[m.reference_index],d1[m.candidate_index];print(f'  class={q.class_index} confidence_fp32={q.confidence:.6f} confidence_fp16={r.confidence:.6f} confidence_absolute_difference={abs(q.confidence-r.confidence):.6g} bbox_fp32={q.box} bbox_fp16={r.box} bbox_IoU={m.iou:.6f}')
 print('This is a conversion sanity check; final accuracy requires the identical 5,000-image COCO evaluation.')
if __name__=='__main__':main()
