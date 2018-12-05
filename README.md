# YOLOv2 in PyTorch
**NOTE: This project is no longer maintained and may not compatible with the newest pytorch (after 0.4.0).**

This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of YOLOv2.
This project is mainly based on [darkflow](https://github.com/thtrieu/darkflow)
and [darknet](https://github.com/pjreddie/darknet).

I used a Cython extension for postprocessing and 
`multiprocessing.Pool` for image preprocessing.
Testing an image in VOC2007 costs about 13~20ms.

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
*YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi*.

并且此处，我对yolo9000 做了些许的修改和中文注释，方便中国和世界的朋友使用yolo在其他工程运用上  
本人的理解有限，如果有什么不对的地方，欢迎批评指正  



## 安装和编译
1. Clone this repository
    你可以拷贝我的yolo 从github 上  

2. 进入yolo 主文件夹 bash make.sh  

## Training YOLOv2
训练自己的yolov2
这里你需要构建一个软连接到你下载的训练集，这里以voc2012 为例子进入data 文件夹  

ln -s [源文件目录] VOCdevkit2012  
以下给你提供了一些voc 的数据集：  
    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```
    
1. 下载预训练模型[pretrained darknet19 model](https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing)
and set the path in `yolo2-pytorch/cfgs/exps/darknet19_exp1.py`.  
    特别注意这里下载下来的权重要放在model weights 里头，这个操作非常的简单 你只需要软连接的建立一个weights 文件夹在 model 里头即可  

2. 训练
    你可以在cfg 文件夹中修改所有的操作，包括训练的batch size ，你预先定义的prior，还有个数，学习率等等  


3. Run the training program: `python train.py`.


## 测试
在测试过程中你可以修改test.py 中的几个函数，其中一个是glob.glob(*)你所存放的测试图片名，里头存放了你要测试的所有图片  
另外你还可以在dataTransform.xmlWrite里头写入你要存放的xml结果位置  
最后通过我们的小工具./linux_v1.4.0/labelImg 来导入图片于xml 来可视化你的结果  



License: HIT license (HIT)
