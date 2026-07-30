"""Validate an explicit-Q/DQ YOLO ONNX model without claiming every op is INT8."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inspect_model(path: Path, require_qdq: bool=True) -> dict:
    import onnx
    from onnx import TensorProto
    model=onnx.load(str(path),load_external_data=False); onnx.checker.check_model(model)
    def info(value):
        t=value.type.tensor_type; return {"name":value.name,"dtype":TensorProto.DataType.Name(t.elem_type),"shape":[d.dim_value for d in t.shape.dim]}
    inputs=[info(x) for x in model.graph.input]; outputs=[info(x) for x in model.graph.output]
    if len(inputs)!=1 or inputs[0]["shape"]!=[1,3,640,640]: raise ValueError(f"unexpected input: {inputs}")
    if len(outputs)!=1 or outputs[0]["shape"]!=[1,84,8400]: raise ValueError(f"unexpected output: {outputs}")
    if inputs[0]["dtype"]!="FLOAT" or outputs[0]["dtype"]!="FLOAT": raise ValueError("external input/output must remain FLOAT")
    nodes=collections.Counter(n.op_type for n in model.graph.node); inits=collections.Counter(TensorProto.DataType.Name(x.data_type) for x in model.graph.initializer)
    init_by_name={x.name:x for x in model.graph.initializer}; scale=[]; zero=[]
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear","DequantizeLinear"):
            if len(node.input)>1 and node.input[1] in init_by_name: scale.append(TensorProto.DataType.Name(init_by_name[node.input[1]].data_type))
            if len(node.input)>2 and node.input[2] in init_by_name: zero.append(TensorProto.DataType.Name(init_by_name[node.input[2]].data_type))
    if require_qdq and (nodes["QuantizeLinear"]==0 or nodes["DequantizeLinear"]==0): raise ValueError("explicit Q/DQ nodes are required")
    result={"path":str(path),"sha256":sha256(path),"file_size":path.stat().st_size,"checker":"passed","inputs":inputs,"outputs":outputs,
      "node_counts":dict(sorted(nodes.items())),"quantize_linear_count":nodes["QuantizeLinear"],"dequantize_linear_count":nodes["DequantizeLinear"],
      "initializer_dtype_counts":dict(sorted(inits.items())),"int8_uint8_initializer_count":inits["INT8"]+inits["UINT8"],
      "scale_initializer_count":len(scale),"scale_initializer_dtypes":dict(collections.Counter(scale)),"zero_point_initializer_count":len(zero),"zero_point_initializer_dtypes":dict(collections.Counter(zero)),
      "claim":"This proves explicit Q/DQ insertion only; it does not prove that every operation executes in INT8."}
    return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--onnx-path",type=Path,required=True);p.add_argument("--output-json",type=Path);a=p.parse_args(argv)
    result=inspect_model(a.onnx_path); print(json.dumps(result,indent=2));
    if a.output_json: a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(result,indent=2),encoding="utf-8")
if __name__=="__main__":main()
