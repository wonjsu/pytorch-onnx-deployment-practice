# YOLOv8n ONNX Export and Postprocessing Example

이 예제는 Ultralytics YOLOv8n 모델을 ONNX 형식으로 export하고, ONNX Runtime raw output을 최종 detection 결과로 후처리하는 방법을 보여줍니다.

## 목적

YOLO는 이미지 안의 객체 위치와 종류를 동시에 예측하는 object detection 모델입니다. Classification 모델과 달리 object detection에서는 다음 요소들이 중요합니다.

- bounding box: 객체가 있는 위치를 나타내는 좌표
- confidence score: 예측한 객체가 실제 객체일 가능성
- class score: 각 bounding box가 어떤 클래스에 해당하는지에 대한 점수
- NMS(Non-Maximum Suppression) 후처리: 중복 bounding box를 제거하고 최종 detection 결과를 선택하는 과정

이번 단계에서는 YOLOv8n 모델을 ONNX로 export하고, ONNX Runtime으로 실제 이미지 입력에 대한 raw output shape을 확인한 뒤, `(1, 84, 8400)` raw output을 후처리해서 최종 detection 결과를 출력합니다. 또한 같은 이미지에 대해 Ultralytics PyTorch 결과와 ONNX Runtime 후처리 결과를 나란히 출력해 사람이 눈으로 class, confidence, bbox가 비슷한지 확인할 수 있습니다.

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

`infer_onnx.py`는 입력 이미지를 letterbox 방식으로 640x640에 맞춰 전처리한 뒤 입력 tensor shape, ONNX output 개수, 각 output shape만 출력합니다.

YOLO 계열 detection에서는 원본 이미지를 정사각형으로 강제 resize하면 aspect ratio가 깨져 객체 모양과 bbox 좌표 복원에 영향을 줄 수 있습니다. 이 예제는 원본 비율을 유지해 640x640 내부에 resize하고 남는 영역을 `114` 값으로 padding하는 letterbox 전처리를 사용합니다.

## ONNX raw output 후처리 실행 방법

`postprocess_onnx.py`는 이미지 경로를 positional argument로 받고, 기본 ONNX 모델 경로 `examples/yolo_onnx/artifacts/yolov8n.onnx`를 사용합니다. ONNX Runtime inference는 `CPUExecutionProvider`로 실행됩니다.

```bash
python examples/yolo_onnx/postprocess_onnx.py assets/test_mouse.jpg
```

다른 ONNX 파일을 사용하려면 `--onnx-path` 옵션으로 경로를 지정할 수 있습니다.

```bash
python examples/yolo_onnx/postprocess_onnx.py assets/test_mouse.jpg \
  --onnx-path examples/yolo_onnx/artifacts/yolov8n.onnx
```

confidence threshold와 NMS IoU threshold의 기본값은 각각 `0.25`, `0.45`입니다. 필요하면 다음 옵션으로 조정할 수 있습니다.

```bash
python examples/yolo_onnx/postprocess_onnx.py assets/test_mouse.jpg \
  --conf-threshold 0.25 \
  --iou-threshold 0.45
```

최종 detection 결과는 COCO 80 class 이름을 포함해 다음 형식으로 출력됩니다.

```text
class_index=64 class_name=mouse confidence=0.6620 bbox=(998.77, 1119.90, 2660.59, 2051.34)
```

출력의 `bbox`는 `(x1, y1, x2, y2)` 순서이며, 640x640 입력 좌표계가 아니라 원본 이미지 좌표계로 복원된 좌표입니다.

## Raw output 후처리 흐름

YOLOv8n ONNX 모델의 raw output shape은 일반적으로 `(1, 84, 8400)`입니다.

1. 이미지를 RGB로 로드하고 원본 비율을 유지한 채 YOLO 입력 크기인 640x640 안에 맞도록 resize한 뒤, 남는 영역을 `114` 값으로 padding하는 letterbox 전처리를 적용합니다.
2. 픽셀 값을 `0.0~1.0` 범위로 정규화하고, tensor layout을 `HWC`에서 `NCHW`로 바꿔 `(1, 3, 640, 640)` 입력 tensor를 만듭니다.
3. ONNX Runtime `CPUExecutionProvider`로 inference를 실행해 raw output `(1, 84, 8400)`을 얻습니다.
4. raw output의 batch 차원을 제거하고 transpose해서 candidate 단위 배열 `(8400, 84)`로 변환합니다.
5. 각 candidate의 앞 4개 값은 bbox `(center_x, center_y, width, height)`로 분리하고, 뒤 80개 값은 COCO class score로 분리합니다.
6. 80개 class score 중 최대값을 confidence로 사용하고, 최대값의 위치를 class index로 사용합니다.
7. confidence가 threshold 기본값 `0.25` 이상인 candidate만 남깁니다.
8. bbox를 `(center_x, center_y, width, height)`에서 `(x1, y1, x2, y2)`로 변환합니다.
9. 640x640 입력 좌표계의 bbox에서 letterbox padding을 뺀 뒤 scale ratio로 나누고, 원본 이미지 범위로 clip해서 원본 이미지 좌표계로 복원합니다.
10. IoU threshold 기본값 `0.45`로 class별 NMS를 적용해 중복 bbox를 제거합니다.
11. 남은 detection을 `class index`, `class name`, `confidence`, `bbox(x1, y1, x2, y2)` 형태로 출력합니다.

## Ultralytics PyTorch 결과와 ONNX Runtime 결과 비교

`compare_ultralytics_onnx.py`는 같은 이미지에 대해 다음 두 결과를 나란히 출력합니다.

1. `YOLO("yolov8n.pt")`로 실행한 Ultralytics PyTorch detection 결과. 비교 기준을 맞추기 위해 640x640 입력 크기와 동일한 confidence/IoU threshold를 사용합니다.
2. `examples/yolo_onnx/artifacts/yolov8n.onnx`를 ONNX Runtime `CPUExecutionProvider`로 실행한 뒤 `postprocess_onnx.py`와 동일한 letterbox 전처리, bbox 복원, confidence threshold, class-aware NMS를 적용한 결과

프로젝트 루트에서 이미지 경로를 positional argument로 전달해 실행합니다.

```bash
python examples/yolo_onnx/compare_ultralytics_onnx.py assets/test_mouse.jpg
```

기본 ONNX 모델 경로는 `examples/yolo_onnx/artifacts/yolov8n.onnx`입니다. 다른 ONNX 파일이나 threshold를 사용하려면 다음 옵션을 지정할 수 있습니다.

```bash
python examples/yolo_onnx/compare_ultralytics_onnx.py assets/test_mouse.jpg \
  --onnx-path examples/yolo_onnx/artifacts/yolov8n.onnx \
  --conf-threshold 0.25 \
  --iou-threshold 0.45
```

비교 스크립트 출력에는 image path, Ultralytics detections, ONNX Runtime detections가 포함됩니다. 각 detection은 `class_index`, `class_name`, `confidence`, `bbox(x1, y1, x2, y2)`를 출력합니다. 이번 예제는 두 결과를 자동 매칭하거나 mAP를 계산하지 않고, 사람이 직접 class, confidence, bbox가 비슷한지 확인할 수 있는 형태로만 출력합니다.

ResNet classification 예제처럼 logits 배열만 단순 비교하는 방식은 YOLO detection 비교에는 충분하지 않습니다. YOLO는 bbox 좌표 복원, confidence threshold, class 예측, NMS 후처리까지 거친 최종 detection을 함께 확인해야 합니다.

참고로 `headphone`은 COCO 80 class에 포함되어 있지 않으므로, YOLOv8n 기본 COCO 모델에서는 헤드폰 이미지가 정확한 `headphone` class로 출력되지 않을 수 있습니다.

## Ultralytics API만 사용한 PyTorch vs ONNX 비교

`compare_ultralytics_pt_onnx.py`는 직접 구현한 ONNX letterbox, NMS, postprocess 코드를 전혀 사용하지 않고, Ultralytics `YOLO` API만 사용해 같은 이미지의 detection 결과를 비교합니다. 이 확인 단계의 목적은 PyTorch YOLOv8n 모델과 exported ONNX 모델 사이의 차이가 ONNX 모델 변환 자체에서 발생한 것인지, 아니면 우리가 직접 구현한 ONNX postprocess 로직의 차이에서 발생한 것인지 구분하는 것입니다.

스크립트는 다음 두 모델을 모두 Ultralytics API로 로드한 뒤 `predict`를 실행합니다.

1. `YOLO("yolov8n.pt")` PyTorch 모델
2. `YOLO("examples/yolo_onnx/artifacts/yolov8n.onnx")` exported ONNX 모델

프로젝트 루트에서 이미지 경로를 positional argument로 전달해 실행합니다. 기본 ONNX 모델 경로는 `examples/yolo_onnx/artifacts/yolov8n.onnx`입니다.

```bash
python examples/yolo_onnx/compare_ultralytics_pt_onnx.py assets/test_mouse.jpg
```

다른 ONNX 파일이나 threshold를 사용하려면 다음 옵션을 지정할 수 있습니다. confidence threshold 기본값은 `0.05`, IoU threshold 기본값은 `0.45`입니다.

```bash
python examples/yolo_onnx/compare_ultralytics_pt_onnx.py assets/test_mouse.jpg \
  --onnx-path examples/yolo_onnx/artifacts/yolov8n.onnx \
  --conf-threshold 0.05 \
  --iou-threshold 0.45
```

출력에는 image path, Ultralytics PyTorch detections, Ultralytics ONNX detections가 포함됩니다. 각 detection은 다음 형식으로 출력됩니다.

```text
class_index=64 class_name=mouse confidence=0.6620 bbox=(998.77, 1119.90, 2660.59, 2051.34)
```

이 스크립트에서 PyTorch 결과와 ONNX 결과가 서로 비슷하다면 ONNX 변환은 대체로 정상이고, 차이는 직접 구현한 postprocess 경로에서 발생했을 가능성이 큽니다. 반대로 Ultralytics API만 사용해도 PyTorch 결과와 ONNX 결과가 크게 다르다면 exported ONNX 모델 또는 변환 설정을 먼저 확인해야 합니다.

## 생성되는 ONNX 파일

export가 끝나면 다음 경로에 ONNX 파일이 생성됩니다.

```text
examples/yolo_onnx/artifacts/yolov8n.onnx
```

`examples/yolo_onnx/artifacts/` 디렉터리는 실행 시 생성되는 artifact 저장 위치입니다. ONNX 파일은 실행 결과물이며 크기가 클 수 있으므로 Git에는 포함하지 않습니다.
