# import torch
#model = torch.hub.load('ultralytics/yolov5', 'custom', './yolov5/runs/train/exp/weights/best.pt')  # custom trained model
## Images
#img = './yolov5/input_images/3337_jpg.rf.e61dcd5ed66d01eb71003d9dffc92435.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
  
# model.hide_labels = True
# model.hide_conf = True
# model.conf = 0.70
  
   
## Inference
# results = model(img)

## Results
# results.show() 
# results.save()
# or .show(), .save(), .crop(),


# !python detect.py --weights weights/last_yolov5s_custom.pt --img 416 --conf 0.4 --source ../test_infer




import torch
import time
import cv2
from PIL import Image
import numpy as np
import pathlib
gen_path = pathlib.Path.cwd()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
  
# torch.hub.load(repo_or_dir, model, *args, source='github', trust_repo=None, force_reload=False, verbose=True, skip_validation=False, **kwargs) 
  
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')  # custom trained model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.50
cam = cv2.VideoCapture(0)
  
while(True): 
    ret, frame = cam.read()
    frame = frame[:, :, [2,1,0]]
    frame = Image.fromarray(frame) 
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    result = model(frame,size=640)
    # cv2.imshow('Output', result)
    cv2.imshow('YOLO', np.squeeze(result.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cam.release()
cv2.destroyAllWindows()