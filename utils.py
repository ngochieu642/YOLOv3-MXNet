import cv2
import mxnet as mx
import numpy as np
from mxnet import nd, gluon
import threading
import xml.etree.ElementTree as ET

def try_gpu(num_list):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()
    Just in case i will have more than 1 GPU in the future haha
    """
    ctx = []
    for num in num_list:
        try:
            tmp_ctx = mx.gpu(int(num))
            _ = nd.array([0], ctx=tmp_ctx)
            ctx.append(tmp_ctx)
        except Exception as e:
            print("gpu {}:".format(num), e)
    if not ctx:
        ctx.append(mx.cpu())
    return ctx
# Test try_gpu
# myGPUs = try_gpu([0,1,2,3])
# print('Hmmm, you have: ',len(myGPUs),'GPU(s). And their name in mxnet: ',myGPUs)

def bbox_iou(box1, box2, transform = True):
    """Calculate the IoU Error
    """

    #Change to NDArray if not
    if not isinstance(box1, nd.NDArray):
        box1 = nd.array(box1)
    if not isinstance(box2, nd.NDArray):
        box2 = nd.array(box2)

    #Make sure > 0
    box1 = nd.abs(box1)
    box2 = nd.abs(box2)

    '''Calculate the IoU'''
    if transform:
        tmp_box1 = box1.copy()
        tmp_box1[:, 0] = box1[:, 0] - box1[:,2] / 2.0
        tmp_box1[:, 1] = box1[:, 1] - box1[:, 3] / 2.0
        tmp_box1[:, 2] = box1[:, 0] + box1[:, 2] / 2.0
        tmp_box1[:, 3] = box1[:, 1] + box1[:, 3] / 2.0
        box1 = tmp_box1

        tmp_box2 = box2.copy()
        tmp_box2[:, 0] = box2[:, 0] - box2[:, 2] / 2.0
        tmp_box2[:, 1] = box2[:, 1] - box2[:, 3] / 2.0
        tmp_box2[:, 2] = box2[:, 0] + box2[:, 2] / 2.0
        tmp_box2[:, 3] = box2[:, 1] + box2[:, 3] / 2.0
        box2 = tmp_box2

    # Get the coordinates of bounding boxes (xStart,yStart,xEnd,yEnd)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = nd.where(b1_x1 > b2_x1, b1_x1, b2_x1) #if b1_x1 > b2_x1 => x1 of the intersection rectangle must be b1_x1, otherwise it will be b2_x1. Basically it's just a max function!
    inter_rect_y1 = nd.where(b1_y1 > b2_y1, b1_y1, b2_y1)
    inter_rect_x2 = nd.where(b1_x2 < b2_x2, b1_x2, b2_x2)
    inter_rect_y2 = nd.where(b1_y2 < b2_y2, b1_y2, b2_y2)

    # Intersection area
    inter_area = nd.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=10000) * nd.clip(
        inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=10000)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)

    return nd.clip(iou, 1e-5, 1. - 1e-5)

def predict_transform(prediction, input_dim, anchors):
    ctx = prediction.context
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors, ctx=ctx)

    batch_size = prediction.shape[0]
    anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    strides = [13, 26, 52]
    step = [(0, 507), (507, 2535), (2535, 10647)]
    for i in range(3):
        stride = strides[i]
        grid = np.arange(stride)
        a, b = np.meshgrid(grid, grid)
        x_offset = nd.array(a.reshape((-1, 1)), ctx=ctx)
        y_offset = nd.array(b.reshape((-1, 1)), ctx=ctx)
        x_y_offset = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.concat(
                            x_offset, y_offset, dim=1), repeats=3, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )
        tmp_anchors = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.expand_dims(
                            anchors[anchors_masks[i]], 0
                        ),
                        repeats=stride * stride, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )

        prediction[:, step[i][0]:step[i][1], :2] += x_y_offset
        prediction[:, step[i][0]:step[i][1], :2] *= (float(input_dim) / stride)
        prediction[:, step[i][0]:step[i][1], 2:4] = \
            nd.exp(prediction[:, step[i][0]:step[i][1], 2:4]) * tmp_anchors

    return prediction

def predict_transform_tiny(prediction, input_dim, anchors):
    ctx = prediction.context
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors, ctx=ctx)

    batch_size = prediction.shape[0]
    anchors_masks = [[3, 4, 5], [0, 1, 2]]
    strides = [13, 26]
    step = [(0, 507), (507, 2535)]
    for i in range(2):
        stride = strides[i]
        grid = np.arange(stride)
        a, b = np.meshgrid(grid, grid)
        x_offset = nd.array(a.reshape((-1, 1)), ctx=ctx)
        y_offset = nd.array(b.reshape((-1, 1)), ctx=ctx)
        x_y_offset = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.concat(
                            x_offset, y_offset, dim=1), repeats=3, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )
        tmp_anchors = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.expand_dims(
                            anchors[anchors_masks[i]], 0
                        ),
                        repeats=stride * stride, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )

        prediction[:, step[i][0]:step[i][1], :2] += x_y_offset
        prediction[:, step[i][0]:step[i][1], :2] *= (float(input_dim) / stride)
        prediction[:, step[i][0]:step[i][1], 2:4] = \
            nd.exp(prediction[:, step[i][0]:step[i][1], 2:4]) * tmp_anchors

    return prediction

def write_results(prediction, num_classes, confidence=0.5, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).expand_dims(2)
    prediction = prediction * conf_mask

    batch_size = prediction.shape[0]

    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = None

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf = nd.max(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf_score = nd.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf = max_conf.astype("float32").expand_dims(1)
        max_conf_score = max_conf_score.astype("float32").expand_dims(1)
        image_pred = nd.concat(image_pred[:, :5], max_conf, max_conf_score, dim=1).asnumpy()
        non_zero_ind = np.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind, :].reshape((-1, 7))
        except Exception as e:
            print(e)
            continue
        if image_pred_.shape[0] == 0:
            continue
        # Get the various classes detected in the image
        img_classes = np.unique(image_pred_[:, -1])
        # -1 index holds the class index

        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * np.expand_dims(image_pred_[:, -1] == cls, axis=1)
            class_mask_ind = np.nonzero(cls_mask[:, -2])
            image_pred_class = image_pred_[class_mask_ind].reshape((-1, 7))

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = np.argsort(image_pred_class[:, 4])[::-1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    box1 = np.expand_dims(image_pred_class[i], 0)
                    box2 = image_pred_class[i + 1:]
                    if len(box2) == 0:
                        break
                    box1 = np.repeat(box1, repeats=box2.shape[0], axis=0)
                    ious = bbox_iou(box1, box2, transform=False).asnumpy()
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = np.expand_dims(ious < nms_conf, 1).astype(np.float32)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = np.nonzero(image_pred_class[:, 4])
                image_pred_class = image_pred_class[non_zero_ind].reshape((-1, 7))

            batch_ind = np.ones((image_pred_class.shape[0], 1)) * ind

            seq = nd.concat(nd.array(batch_ind), nd.array(image_pred_class), dim=1)

            if output is None:
                output = seq
            else:
                output = nd.concat(output, seq, dim=0)
    return output

def letterbox_image(img, inp_dim, labels=None):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if labels is not None:
        mask = labels > 0.
        labels[:, 1] = (labels[:, 1] * new_w + (w - new_w) // 2) / w
        labels[:, 2] = (labels[:, 2] * new_h + (h - new_h) // 2) / h
        labels[:, 3] = labels[:, 3] * new_w / w
        labels[:, 4] = labels[:, 4] * new_h / h
        labels *= mask
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas, labels

def prep_image(img, inp_dim, labels=None):
    img, labels = letterbox_image(img, (inp_dim, inp_dim), labels)
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype("float32")
    img /= 255.0
    if labels is not None:
        return img, labels
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.readlines()
    names = [name.strip() for name in names]
    if names[-1] == "\n":
        names.pop(-1)
    return names

def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]

class SigmoidBinaryCrossEntropyLoss(gluon.loss.Loss):
    def __init__(self, from_sigmoid=False, weight=1, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        if not self._from_sigmoid:
            # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
            tmp_loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            tmp_loss = -(F.log(pred + 1e-12) * label + F.log(1. - pred + 1e-12) * (1. - label))
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class L1Loss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.abs(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss

class L2Loss(gluon.loss.Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.square(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight / 2, sample_weight)
        return tmp_loss

class FocalLoss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, gamma=2, eps=1e-7, alpha=0.25, with_ce=False):
        super(FocalLoss, self).__init__(weight=weight, batch_axis=batch_axis)
        self.gamma = gamma
        self.eps = eps
        self.with_ce = with_ce
        self.alpha = alpha

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if self.alpha > 0.:
            alpha_t = F.abs(self.alpha + (label == 1.).astype("float32") - 1.)
        else:
            alpha_t = 1.
        if self.with_ce:
            sce_loss = SigmoidBinaryCrossEntropyLoss()
            p_t = sce_loss(pred, label)
            tmp_loss = -(alpha_t * F.power(1 - p_t, self.gamma) * p_t)
        else:
            p_t = F.clip(F.abs(pred + label - 1.), a_min=self.eps, a_max=1. - self.eps)
            tmp_loss = -(alpha_t * F.power(1 - p_t, self.gamma) * F.log(p_t))
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss

class HuberLoss(gluon.loss.Loss):
    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label =gluon.loss. _reshape_like(F, label, pred)
        loss = F.clip(F.abs(pred - label), a_min=1e-7, a_max=10000.)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho,
                       (0.5/self._rho) * F.power(loss, 2))
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        return loss

class LossRecorder(mx.metric.EvalMetric):
    """LossRecorder is used to record raw loss so we can observe loss directly
    """

    def __init__(self, name):
        super(LossRecorder, self).__init__(name)

    def update(self, labels, preds=0):
        """Update metric with pure loss
        """
        for loss in labels:
            self.sum_metric += np.mean(loss.copy().asnumpy())
            self.num_inst += 1

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print(e)
            return None
