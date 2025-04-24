import cv2
import torch
from ultralytics import YOLO


model = YOLO('models/cluster.pt') 
# set model to eval mode
model.eval()


input_path = 'inputs/input6.mp4'   
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {input_path}")

"""
Classes: {0: 'biker', 1: 'car', 2: 'pedestrian', 3: 'trafficLight', 4: 'trafficLight-Green', 
            5: 'trafficLight-GreenLeft', 6: 'trafficLight-Red', 7: 'trafficLight-RedLeft',
            8: 'trafficLight-Yellow', 9: 'trafficLight-YellowLeft', 10: 'truck'}
"""


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)        
    r = results[0]              

    print(f"Classes: {model.names}")

    # get Tensors
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy()

    # 4. Draw boxes and labels
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{model.names[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
     
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), 
                             (x1 + t_size[0], y1), (0,255,0), -1)
      
        cv2.putText(frame, label, (x1, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 5. Show and/or write
    cv2.imshow('YOLO Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

