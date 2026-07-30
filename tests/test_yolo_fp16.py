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
from examples.yolo_fp16.convert_fp16_modelopt import convert, validate_calibration_data, validate_paths
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

def test_modelopt_receives_calibration_directory_path(tmp_path,monkeypatch):
 source=tmp_path/'source.onnx';model(source,False)
 calibration=tmp_path/'autocast_data';calibration.mkdir()
 np.savez(calibration/'batch_0001.npz',images=np.zeros((1,3,640,640),dtype=np.float32))
 np.savez(calibration/'batch_0000.npz',images=np.zeros((1,3,640,640),dtype=np.float32))
 calls=[]
 def mock_convert_to_mixed_precision(onnx_path,low_precision_type,keep_io_types,calibration_data,providers):
  calls.append({'onnx_path':onnx_path,'low_precision_type':low_precision_type,'keep_io_types':keep_io_types,
                'calibration_data':calibration_data,'providers':providers})
  result=onnx.load(onnx_path)
  result.graph.initializer.append(helper.make_tensor('half_weight',TensorProto.FLOAT16,[1],[1.0]))
  return result
 autocast=type(sys)('modelopt.onnx.autocast');autocast.convert_to_mixed_precision=mock_convert_to_mixed_precision
 monkeypatch.setitem(sys.modules,'modelopt',type(sys)('modelopt'))
 monkeypatch.setitem(sys.modules,'modelopt.onnx',type(sys)('modelopt.onnx'))
 monkeypatch.setitem(sys.modules,'modelopt.onnx.autocast',autocast)
 output=tmp_path/'output.onnx'
 metadata=convert(source,output,calibration,['cpu'])
 assert calls[0]['calibration_data']==str(calibration)
 assert isinstance(calls[0]['calibration_data'],str)
 assert metadata['calibration_source_type']=='npz_directory'
 assert metadata['calibration_batch_count']==2

def test_calibration_source_validation(tmp_path):
 npz=tmp_path/'batch.npz';np.savez(npz,images=np.zeros(1))
 json_path=tmp_path/'batches.json';json_path.write_text('{}')
 assert validate_calibration_data(npz)==('npz_file',1)
 assert validate_calibration_data(json_path)==('json_file',None)
 empty=tmp_path/'empty';empty.mkdir()
 with pytest.raises(ValueError,match='contains no NPZ'):validate_calibration_data(empty)
 invalid=tmp_path/'batch.npy';invalid.write_bytes(b'x')
 with pytest.raises(ValueError,match='must be an NPZ file'):validate_calibration_data(invalid)
