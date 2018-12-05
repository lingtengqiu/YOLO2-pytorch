#coding: utf-8
import os
import torch
import datetime

from darknet import Darknet19

from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint
from utils.visualizer import Visualizer



'''
默认的初始化过程
('voc_2007_trainval', '/home/qlt/qiulingteng/detection/yolo2-pytorch-master/data', 16, <function preprocess_train at 0x7f61c510de60>)
[array([320, 320]), array([352, 352]), array([384, 384]), array([416, 416]), array([448, 448]), array([480, 480]), array([512, 512]), array([544, 544]), array([576, 576])]
'''
# data loader
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=1, shuffle=True,
                  dst_size=cfg.multi_scale_inp_size)
viz = Visualizer()
# dst_size=cfg.inp_size)
print('load data succ...')

net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

print("For this training we have follow para:\n"
      "lr :{}\n"
      "momentum:{}\n"
      "weight_decay{}\n"
      "epoch:{}".format(cfg.init_learning_rate,cfg.momentum,cfg.weight_decay,cfg.max_epoch))

# optimizer
start_epoch = 0
'''
lr : 0.001
'''
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad

batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0
for step in range(start_epoch * imdb.batch_per_epoch,
                  cfg.max_epoch * imdb.batch_per_epoch):
    t.tic()
    # batch
    batch = imdb.next_batch(size_index)
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=True,
                                       volatile=False).permute(0, 3, 1, 2)
    bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)

    # backward
    loss = net.loss
    bbox_loss += net.bbox_loss.item()
    iou_loss += net.iou_loss.item()
    cls_loss += net.cls_loss.item()
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    if step % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
               'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
               (imdb.epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                iou_loss, cls_loss, duration,
                str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

        if viz and step % cfg.log_interval == 0:
            viz.line("train/loss_train",train_loss,step)
            viz.line("train/loss_iou",iou_loss,step)
            viz.line("train/loss_cls",cls_loss,step)
            viz.line("train/learning_rate",lr,step)

            # plot results
            bbox_pred = bbox_pred.data[0:1].cpu().numpy()
            iou_pred = iou_pred.data[0:1].cpu().numpy()
            prob_pred = prob_pred.data[0:1].cpu().numpy()
            image = im[0]
            bboxes, scores, cls_inds = yolo_utils.postprocess(
                bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index)
            im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
            im2show = torch.from_numpy(im2show).permute(2,0,1)
            viz.writer.add_image('predict', im2show, step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()
        size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

    if step > 0 and (step % imdb.batch_per_epoch == 0):
        if imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)

        save_name = os.path.join(cfg.train_output_dir,
                                 '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
        net_utils.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))
        step_cnt = 0

imdb.close()
