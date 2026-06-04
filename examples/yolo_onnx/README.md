# YOLOv8n ONNX Export Example

이 예제는 Ultralytics YOLOv8n 모델을 ONNX 형식으로 export하는 방법을 보여줍니다.

## 목적

YOLO는 이미지 안의 객체 위치와 종류를 동시에 예측하는 object detection 모델입니다. Classification 모델과 달리 object detection에서는 다음 요소들이 중요합니다.

- bounding box: 객체가 있는 위치를 나타내는 좌표
- confidence score: 예측한 객체가 실제 객체일 가능성
- class score: 각 bounding box가 어떤 클래스에 해당하는지에 대한 점수
- NMS(Non-Maximum Suppression) 후처리: 중복 bounding box를 제거하고 최종 detection 결과를 선택하는 과정

이번 단계에서는 YOLOv8n 모델을 ONNX로 export하고, ONNX Runtime으로 실제 이미지 입력에 대한 raw output shape을 확인합니다. bounding box 후처리, NMS, benchmark는 아직 포함하지 않습니다.

## 실행 방법

프로젝트 루트에서 필요한 패키지를 설치한 뒤 export 스크립트를 실행합니다.

```bash
pip install -r requirements.txt
python examples/yolo_onnx/export_onnx.py
```

스크립트는 `ultralytics`에서 `YOLO`를 import하고 `YOLO("yolov8n.pt")` 모델을 로드한 뒤, `model.export(format="onnx", imgsz=640, opset=17)` 방식으로 ONNX 파일을 생성하고, export 결과를 `examples/yolo_onnx/artifacts/yolov8n.onnx`에 저장합니다.

ONNX Runtime으로 실제 이미지를 입력해 raw output shape을 확인하려면 이미지 경로를 positional argument로 전달합니다. 기본 ONNX 모델 경로는 `examples/yolo_onnx/artifacts/yolov8n.onnx`이며, inference는 `CPUExecutionProvider`로 실행됩니다.

```bash
python examples/yolo_onnx/infer_onnx.py assets/test_mouse.jpg
```

다른 ONNX 파일을 사용하려면 `--onnx-path` 옵션으로 경로를 지정할 수 있습니다. 현재 inference 스크립트는 입력 이미지를 640x640으로 전처리한 뒤 입력 tensor shape, ONNX output 개수, 각 output shape만 출력하며 bounding box 후처리나 NMS는 수행하지 않습니다.

## 생성되는 ONNX 파일

export가 끝나면 다음 경로에 ONNX 파일이 생성됩니다.

```text
examples/yolo_onnx/artifacts/yolov8n.onnx
```

`examples/yolo_onnx/artifacts/` 디렉터리는 실행 시 생성되는 artifact 저장 위치입니다. ONNX 파일은 실행 결과물이며 크기가 클 수 있으므로 Git에는 포함하지 않습니다.
