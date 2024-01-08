## 什麼是 people-counting？
StrongSORT with OSNet for YoloV5 and YoloV7（Counter）是一個可配置的兩階段追蹤器，可根據不同的場景進行調整。它將由 YOLOv5 和 YOLOv7 生成的檢測結果傳遞給 StrongSORT，這是一個基於運動和外觀信息的多目標追蹤器，並使用 OSNet 來結合這些信息。它可以追蹤模型訓練的任何對象，因此我們可以透過這個技術來實施來人流監控。

## 如何安裝 people-counting
建議使用 Python 3.6 創建一個新的虛擬環境並安裝所需的依賴項。

### Clone the repository
```
git clone https://github.com/AI-Foundry-Face-Recognition-Project/people-counting.git
```

### 安裝所需的套件

```
pip install -r requirements.txt
```

## 如何運行 people-counting
在安裝完成後，您可以執行 StrongSORT with OSNet for YoloV5 and YoloV7 來進行對象追蹤。以下是示例運行命令：

使用 Yolov5 模型進行對象追蹤：

python track_v5.py --yolo-weights weights/yolov5n.pt --img 640 --source <影像來源> --classroom <統計教室編號>
