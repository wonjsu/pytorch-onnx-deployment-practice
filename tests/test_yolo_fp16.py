"""CPU-only tests for the mixed-FP16 workflow."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image
from examples.yolo_fp16.convert_fp16_modelopt import validate_paths
from examples.yolo_fp16.generate_autocast_data import generate, select_images
from examples.yolo_fp16.inspect_mixed_precision_onnx import inspect_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script_name', [
 'generate_autocast_data.py',
 'inspect_mixed_precision_onnx.py',
 'compare_fp32_fp16_onnx.py',
 'convert_fp16_modelopt.py',
])
def test_scripts_support_direct_help(script_name):
 script=REPOSITORY_ROOT/'examples'/'yolo_fp16'/script_name
 result=subprocess.run([sys.executable,str(script),'--help'],cwd=REPOSITORY_ROOT,check=False)
 assert result.returncode==0


def test_convert_script_supports_module_help():
 result=subprocess.run(
  [sys.executable,'-m','examples.yolo_fp16.convert_fp16_modelopt','--help'],
  cwd=REPOSITORY_ROOT,
  check=False,
 )
 assert result.returncode==0

def model(path:Path,mixed=True,io_dtype=TensorProto.FLOAT):
 x=helper.make_tensor_value_info('images',io_dtype,[1,3,640,640]); y=helper.make_tensor_value_info('output0',io_dtype,[1,3,640,640]); nodes=[]; initializers=[]
 if mixed:
  initializers.append(helper.make_tensor('half_weight',TensorProto.FLOAT16,[1],[1.0])); nodes=[helper.make_node('Cast',['images'],['half'],to=TensorProto.FLOAT16),helper.make_node('Cast',['half'],['output0'],to=TensorProto.FLOAT)]
 else:nodes=[helper.make_node('Identity',['images'],['output0'])]
 onnx.save(helper.make_model(helper.make_graph(nodes,'test',[x],[y],initializers),opset_imports=[helper.make_opsetid('',13)]),path)

def test_deterministic_selection():
 images=[{'id':i,'file_name':f'{i}.jpg'} for i in range(20)]
 assert select_images(images,5,7)==select_images(list(reversed(images)),5,7)
 assert select_images(images,5,7)!=select_images(images,5,8)

def test_npz_key_dtype_shape(tmp_path):
 onnx_path=tmp_path/'m.onnx';model(onnx_path,False); images=tmp_path/'images';images.mkdir();Image.new('RGB',(20,10)).save(images/'x.jpg'); ann=tmp_path/'a.json';ann.write_text(json.dumps({'images':[{'id':1,'file_name':'x.jpg'}]}))
 meta=generate(onnx_path,images,ann,tmp_path/'out',1,0)
 with np.load(tmp_path/'out'/'batch_0000.npz') as z: assert z.files==['images'] and z['images'].dtype==np.float32 and z['images'].shape==(1,3,640,640)
 json.dumps(meta)

def test_inspection_and_io_validation(tmp_path):
 path=tmp_path/'mixed.onnx';model(path); result=inspect_model(path)
 assert result['float16_initializer_count']==1 and result['cast_to_float16_count']==1 and result['cast_to_float32_count']==1
 plain=tmp_path/'plain.onnx';model(plain,False)
 with pytest.raises(ValueError,match='no FLOAT16'):inspect_model(plain)
 bad=tmp_path/'bad.onnx';model(bad,True,TensorProto.FLOAT16)
 with pytest.raises(ValueError,match='External ONNX I/O'):inspect_model(bad)

def test_output_validation_and_overwrite(tmp_path):
 source=tmp_path/'source.onnx';source.write_bytes(b'x'); output=tmp_path/'output.onnx';validate_paths(source,output)
 with pytest.raises(ValueError,match='overwrite'):validate_paths(source,source)
 output.write_bytes(b'x')
 with pytest.raises(FileExistsError):validate_paths(source,output)
 validate_paths(source,output,True)
