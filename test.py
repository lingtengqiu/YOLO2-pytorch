#coding: utf-8
import os
import cv2
import numpy as np
import pickle
import argparse

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg
import glob
import sys
import dataTransform
import torch


parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--image_size_index', type=int, default=4,
                    metavar='image_size_index',
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
args = parser.parse_args()


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
trained_model = os.path.join(cfg.train_output_dir,
                             'darknet19_voc07trainval_exp3_99.h5')
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.5
vis = False
# ------------


def test_net(net, imdb, max_per_image=300, thresh=0.5, vis=False):


    num_images = imdb.num_images

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index
    #helper: 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576'
    #here val_img sometimes is 5123
    for i in range(num_images):

        batch = imdb.next_batch(size_index=size_index)
        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=True,
                                           volatile=True).permute(0, 3, 1, 2)

        _t['im_detect'].tic()
        with torch.set_grad_enabled(False):
            bbox_pred, iou_pred, prob_pred = net(im_data)
        '''
        bbox->(batch,h*w,prior 4)
        iou ->(batch,h*w,prior,1)
        prob_pred-->(batch,h*w,prior,20)
        '''
        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        '''
        这里后处理的是:
        return bbox_pred, scores, cls_inds
        '''
        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,
                                                          iou_pred,
                                                          prob_pred,
                                                          ori_im.shape,
                                                          cfg,
                                                          thresh,
                                                          size_index
                                                          )
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()
        '''
        以下的操作是
        对我们预测的值进行处理，这里需要注意的是，对于
        这些问题，我们在最后头保留它的概率
        并对最后的概率获取
        '''
        for j in range(imdb.num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))  # noqa
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis:
            im2show = yolo_utils.draw_detection(ori_im,
                                                bboxes,
                                                scores,
                                                cls_inds,
                                                cfg,
                                                thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show,
                                     (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            cv2.imshow('test', im2show)
            cv2.waitKey(0)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
def test_only_transform(im,inp_size,size_index):
    im = cv2.imread(im)
    ori_im = np.copy(im)

    if inp_size is not None and size_index is not None:
        inp_size = inp_size[size_index]
        w, h = inp_size
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 255.
    return im, [], [], [], ori_im

def test_net_img_only(net, img_list, max_per_image=300, thresh=0.5, vis=False):
    num_images = len(img_list)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(cfg.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index
    inp_size = cfg.multi_scale_inp_size
    if not os.path.exists("result"):
        os.mkdir("result")
    dt = dataTransform.dataTransform()
    for i in range(num_images):
        img_name = img_list[i]
        im,_,__,___,ori_im = test_only_transform(img_name,inp_size,size_index)
        im = np.reshape(im,newshape=(-1,im.shape[0],im.shape[1],im.shape[2]))
        im_data = net_utils.np_to_variable(im, is_cuda=True,
                                           volatile=True).permute(0, 3, 1, 2)
        with torch.set_grad_enabled(False):
            bbox_pred, iou_pred, prob_pred = net(im_data)
        '''
        bbox->(batch,h*w,prior 4)
        iou ->(batch,h*w,prior,1)
        prob_pred-->(batch,h*w,prior,20)
        '''
        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        '''
        这里后处理的是:
        return bbox_pred, scores, cls_inds
        '''
        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,
                                                          iou_pred,
                                                          prob_pred,
                                                          ori_im.shape,
                                                          cfg,
                                                          thresh,
                                                          size_index
                                                          )
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()
        '''
        以下的操作是
        对我们预测的值进行处理，这里需要注意的是，对于
        这些问题，我们在最后头保留它的概率
        并对最后的概率获取
        '''
        for j in range(imdb.num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        #save detect_result to xml
        dt.writeXml(img_name,"./result",ori_im,cfg.label_names,cls_inds.tolist(),bboxes.tolist())
        if vis:
            im2show = yolo_utils.draw_detection(ori_im,
                                                bboxes,
                                                scores,
                                                cls_inds,
                                                cfg,
                                                thr=0.5)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show,
                                     (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            cv2.imshow('test', im2show)
            cv2.waitKey(0)





if __name__ == '__main__':
    # data loader
    imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test,
                      processes=1, shuffle=False, dst_size=cfg.multi_scale_inp_size)

    net = Darknet19()
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()
    # test_net(net, imdb, max_per_image, thresh, vis)
    img_list = glob.glob(os.path.join("./demo","*.jpg"))
    test_net_img_only(net, img_list, max_per_image, thresh, vis)
