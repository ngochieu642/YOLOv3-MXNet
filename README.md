## Installation
  - opencv-contrib-python (4.1.0)
  - Without GPU: mxnet (1.4.0)
  - With NVIDIA GPU and CUDA: mxnet-cu90 (for CUDA 9.0) or mxnet-cu100 (for CUDA 10.0)
## Project Structure
    data        
        dataset-name
            coco.names
            voc.names
            ...
        weights
            yolov3-608.weights
            yolov3-tiny.weights
    Video
        Video-1.mp4
        Video-2.mp4
        ...
    darknet.py
    utils.py
    testYOLO.py
    testYOLO-tiny.py
- All dataset names, pretrained weights can be found here: [yolo home page](https://pjreddie.com/darknet/yolo/)
- This project just deploys YOLOv3, YOLOv3-tiny with pretrained weight and cfg, for training with custom dataset, use the following [AlexeyAB's repo](https://github.com/AlexeyAB/darknet?fbclid=IwAR0x8rN6W3y-4jRCrW70OddJpatI6uwBc5aCQ9-Wn3Thf-VwlM-F7dpEg08#how-to-train-to-detect-your-custom-objects)

### Demo
- Put weights in data/weights/
- Put dataset names in data/names/
- Put Video in Video/
- Make some minor change in testYOLO.py / testYOLO-tiny.py at line 7 (classes), line 25 (weights), line 31 (Video).
##
    python testYOLO.py 
    python testYOLO-tiny.py


