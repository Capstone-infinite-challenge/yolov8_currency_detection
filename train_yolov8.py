from ultralytics import YOLO

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")  # 경량 모델 (n: nano, s: small, m: medium)

# 훈련 시작
model.train(
    data="money.v1i.yolov8/data.yaml",  # data.yaml 경로
    epochs=50,                         # 훈련 반복 수
    imgsz=640,                         # 이미지 크기 (기본값: 640)
    batch=16,                          # 배치 크기
    device=0                           # GPU 사용 (없으면 'cpu')
)
