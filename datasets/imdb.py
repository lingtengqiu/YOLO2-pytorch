#coding: utf-8
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import cfgs.config as cfg
import cv2


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


class ImageDataset(object):
    def __init__(self, name, datadir, batch_size, im_processor,
                 processes=3, shuffle=True, dst_size=None):
        self._name = name
        self._data_dir = datadir
        self._batch_size = batch_size
        self.dst_size = dst_size

        self._epoch = -1
        self._num_classes = 0
        self._classes = []

        # load by self.load_dataset()
        self._image_indexes = []
        self._image_names = []
        self._annotations = []
        # Use this dict for storing dataset specific config options
        self.config = {}

        # Pool
        self._shuffle = shuffle
        self._pool_processes = processes
        self.pool = Pool(self._pool_processes)
        self.gen = None
        self._im_processor = im_processor

    def next_batch(self, size_index):
        '''
        由于这个算法使用的是next batch ，所以我们这里考虑如何使用next_batch 来思考问题

        '''
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [],
                 'dontcare': [], 'origin_im': []}
        i = 0
        if self.gen is None:
            indexes = np.arange(len(self.image_names), dtype=np.int)
            if self._shuffle:
                np.random.shuffle(indexes)
            '''
            这里为什么不考虑使用map 呢？
            主要还是因为的是map返回的是整个list 列表，对于我们而言返回迭代器就好了，
            何必返回的是list 感觉这个问题和xrange 是一样的，算法问题，算法解决
            '''
            '''
            接下来好好看看，_im_processor 用了那些操作,
            特别注意 imap 这里返回的是一个迭代器
            '''
            self.gen = self.pool.imap(partial(self._im_processor,
                                              size_index=None),
                                      ([self.image_names[i],
                                        self.get_annotation(i),
                                        self.dst_size] for i in indexes),
                                      chunksize=self.batch_size)
            self._epoch += 1
            print(('epoch {} start...'.format(self._epoch)))

        while i < self.batch_size:
            try:
                images, gt_boxes, classes, dontcare, origin_im = next(self.gen)

                # multi-scale
                '''
                你可以发现他在上头图像处理的时候还未使用 我们所选的输入图片大小而是在这一步操作的。想想为什么？
                '''
                w, h = cfg.multi_scale_inp_size[size_index]
                gt_boxes = np.asarray(gt_boxes, dtype=np.float)
                if len(gt_boxes) > 0:
                    gt_boxes[:, 0::2] *= float(w) / images.shape[1]
                    gt_boxes[:, 1::2] *= float(h) / images.shape[0]
                images = cv2.resize(images, (w, h))

                batch['images'].append(images)
                batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].append(classes)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
                i += 1
                # for index in range(gt_boxes.shape[0]):
                #     cv2.rectangle(images,(int(gt_boxes[index][0]),int(gt_boxes[index][1])),(int(gt_boxes[index][2]),int(gt_boxes[index][3])),(0,0,255),3)
                # cv2.imshow("fuck",images)
                # cv2.waitKey()
            except (StopIteration,):
                indexes = np.arange(len(self.image_names), dtype=np.int)
                if self._shuffle:
                    np.random.shuffle(indexes)
                self.gen = self.pool.imap(partial(self._im_processor,
                                                  size_index=None),
                                          ([self.image_names[i],
                                            self.get_annotation(i),
                                            self.dst_size] for i in indexes),
                                          chunksize=self.batch_size)
                self._epoch += 1
                print(('epoch {} start...'.format(self._epoch)))
        batch['images'] = np.asarray(batch['images'])
        return batch

    def close(self):
        self.pool.terminate()
        self.pool.join()
        self.gen = None

    def load_dataset(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def get_annotation(self, i):
        if self.annotations is None:
            return None
        return self.annotations[i]

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_names(self):
        return self._image_names

    @property
    def image_indexes(self):
        return self._image_indexes

    @property
    def annotations(self):
        return self._annotations

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        mkdir(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_names)

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_per_epoch(self):
        return self.num_images // self.batch_size
