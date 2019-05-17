import time
from utils import *
from darknet import DarkNet, TinyDarkNet
import cv2
import numpy as np

classes     = str('data/dataset-name/coco.names')
classes     = load_classes(classes)
num_classes = len(classes)
input_dim   = 416
print('number of classes: ',num_classes)

#Get GPU(s) name(es)
ctx = try_gpu([0])[0]
confidence = 0.8
nms_thresh = 0.2

net =DarkNet(input_dim=input_dim,num_classes=num_classes)
anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                    (59, 119), (116, 90), (156, 198), (373, 326)])
net.initialize(ctx=ctx)

tmp_batch = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx=ctx)
net(tmp_batch)
param=str("./data/weights/yolov3-608.weights")

#Load weights
net.load_weights(param, fine_tune=False)
net.hybridize()

capture = cv2.VideoCapture('./Video/Survei_1.mp4')

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()

    if ret:
        #Preprocessing
        frame = cv2.resize(frame,(1280,720),
        interpolation=cv2.INTER_CUBIC)

        #Turn image into array
        img = nd.array(prep_image(frame,input_dim),ctx=ctx).expand_dims(0)

        #Get Prediction
        prediction = predict_transform(net(img), input_dim, anchors)
        prediction = write_results(prediction, num_classes, confidence = confidence, nms_conf = nms_thresh)

        #If there were no predictions, display the FPS and continue
        if prediction is None:
            FPS = 'FPS {:.1f}'.format(1/(time.time()-stime))
            frame=cv2.putText(frame,FPS,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.imshow('frame',frame)
            continue

        #Else if there are predictions, scale the frame
        scaling_factor = min(input_dim / frame.shape[0], input_dim / frame.shape[1])

        prediction[:, [1, 3]] -= (input_dim - scaling_factor * frame.shape[1]) / 2
        prediction[:, [2, 4]] -= (input_dim - scaling_factor * frame.shape[0]) / 2
        prediction[:, 1:5] /= scaling_factor

        #Loop over the predictions
        for i in range(prediction.shape[0]):
            prediction[i, [1, 3]] = nd.clip(prediction[i, [1, 3]], 0.0, frame.shape[1])
            prediction[i, [2, 4]] = nd.clip(prediction[i, [2, 4]], 0.0, frame.shape[0])

        #Loop over detected rectangles
        prediction = prediction.asnumpy()
        for result in prediction:
            #Infomation from net
            c1 = tuple(result[1:3].astype("int"))
            c2 = tuple(result[3:5].astype("int"))
            cls = int(result[-1])
            color = (0, 255, 0)
            label = "{0} {1:.3f}".format(classes[cls], result[-2])

            #Draw rectangle
            cv2.rectangle(frame, c1, c2, color, 1)

            #Get text size
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 2, c1[1] - t_size[1] - 5

            #Put Text
            cv2.putText(frame, label, (c1[0], c1[1] - t_size[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .5, (0x3B, 0x52, 0xB1), 2)

        #Put FPS
        FPS = 'FPS {:.1f}'.format(1/(time.time()-stime))
        frame=cv2.putText(frame,FPS,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(10,10,10),2)

        cv2.imshow('mxnet',frame)

        #Escape key
        if cv2.waitKey(1)&0xFF ==ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
