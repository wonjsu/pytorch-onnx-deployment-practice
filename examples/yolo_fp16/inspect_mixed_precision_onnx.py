"""Validate that an ONNX graph has FP32 external I/O and real internal FP16."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def _tensor_info(value: Any) -> dict[str, Any]:
    import onnx
    tt=value.type.tensor_type
    shape=[d.dim_value if d.HasField('dim_value') else None for d in tt.shape.dim]
    return {'name':value.name,'shape':shape,'dtype':onnx.TensorProto.DataType.Name(tt.elem_type)}

def inspect_model(path: Path, require_mixed: bool=True) -> dict[str, Any]:
    import onnx
    path=Path(path)
    if not path.is_file() or path.stat().st_size == 0: raise ValueError(f'ONNX file missing or empty: {path}')
    model=onnx.load(str(path)); onnx.checker.check_model(model)
    inputs=[_tensor_info(v) for v in model.graph.input]
    outputs=[_tensor_info(v) for v in model.graph.output]
    if len(inputs)!=1 or inputs[0]['shape'] != EXPECTED_INPUT_SHAPE:
        raise ValueError(f'Expected one static input shape {EXPECTED_INPUT_SHAPE}, got {inputs}')
    non_fp32=[v for v in inputs+outputs if v['dtype']!='FLOAT']
    if non_fp32: raise ValueError(f'External ONNX I/O must remain FP32: {non_fp32}')
    init=Counter(onnx.TensorProto.DataType.Name(v.data_type) for v in model.graph.initializer)
    cast=Counter()
    for node in model.graph.node:
        if node.op_type=='Cast':
            attr=next((a for a in node.attribute if a.name=='to'),None)
            if attr is not None: cast[onnx.TensorProto.DataType.Name(attr.i)] += 1
    result={'path':str(path),'sha256':sha256(path),'file_size_bytes':path.stat().st_size,
            'opsets':{x.domain or 'ai.onnx':x.version for x in model.opset_import},
            'inputs':inputs,'outputs':outputs,'initializer_counts':dict(sorted(init.items())),
            'cast_destination_counts':dict(sorted(cast.items())),
            'float16_initializer_count':init['FLOAT16'],'float32_initializer_count':init['FLOAT'],
            'cast_to_float16_count':cast['FLOAT16'],'cast_to_float32_count':cast['FLOAT'],
            'node_count':len(model.graph.node),'op_type_counts':dict(sorted(Counter(n.op_type for n in model.graph.node).items()))}
    if require_mixed and not (init['FLOAT16'] or cast['FLOAT16']):
        raise ValueError('Model labelled mixed-fp16 has no FLOAT16 initializer or Cast-to-FLOAT16 node')
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--onnx-path',type=Path,required=True); p.add_argument('--output-json',type=Path)
    a=p.parse_args(argv); result=inspect_model(a.onnx_path)
    print(json.dumps(result,indent=2))
    if a.output_json: a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Internal FP16 exists, but this does not imply that every operation executes in FP16.')
if __name__=='__main__': main()
