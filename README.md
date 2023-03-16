# Multi-granularity Detector Focusing on Size-different Objects and Positive and Negative Samples Imbalance

### MgD

written by · [Chen Dong](https://github.com/Alan-D-Chen) (go to [Google Scholar](https://scholar.google.com/citations?user=51yJbQ0AAAAJ&hl=zh-CN))
 · Miao Duoqian
 · Xuerong Zhao

### This code base has received great help from [zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). Although the innovation work of the paper was completed by me, much thanks to [zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

## Abstract
This paper revisits object detection models and points out that the performance of detectors is restricted by poor results of small objects and imbalance between positive and negative samples. To those ends, we propose Multi-granularity Detector (MgD), in which the main ingredients are Multi-granularity Feature Extraction (MFE) and Sequential three-way Selection (S3WS). In MFE, depending on the analysis of different-size objects, we apply three multi-granularity customizable deformable convolutions to three layers of feature maps. MFE improves the results of small objects, which in turn improves the performance of general object detection. Meanwhile, we propose S3WS to ameliorate the imbalance between positive and negative samples. Region proposals are fed into S3WS, then more positive samples are selected from positive and boundary regions according to multiple evaluation functions and two dynamical thresholds layer by layer. Extensive experiments on COCO benchmark prove that MgD outperforms other state-of-the-art models in system level. Meanwhile, SwinV2-G with MFE and SW3S (AP 63.1→64.0, AP/APs 1.97→1.42) surpasses other state-of-the-art results. MgD(AP 53.9, AP/APs 1.35) greatly improves the contribution of small objects. Moreover, MFE and S3WS can be easily integrated into ConvNet detectors and transformer-based detectors, and achieve significant improvements.

## Keywords: 
Computer Vision, Deep Learning, Object Detection, Granular Computing

## Conclusion
In this work, we identify that poor results of small objects and imbalance between positive and negative samples restrict the performance of detectors. To address these issues, MgD are proposed, which consists of MFE and S3WS modules. Both MFE and S3WS modules can be integrated into the existing methods easily. The experiments demonstrate that the performance of detectors gradually gets significant improvements by adding MFE and S3WS at an acceptable cost. Furthermore, the MgD detector outperforms all other state-of-the-art ones. The MgD does improve the contribution of small objects. Meanwhile, SwinV2-G with MFE and SW3S (AP 63.1→64.0, AP/APs 1.97→1.42) surpasses other state-of-the-art results. MgD(AP 53.9, AP/APs 1.35) greatly improves the contribution of small objects. 

But the innovation of this paper also has obvious limitations. The MFE module is mainly limited to the statistical information of independent data sets, and obviously lacks the generalization ability. When switching task scenarios, the MFE module lacks
flexibility. The S3WS module is stacked by basic IoU functions, and does not compress the running time and memory space of each IoU function. At the same time, the performance of SW3S is subject to the combination of the performance of several different IoUs.

In the future work, we will mainly solve the application of MFE module in object detection. At the same time, we should pay attention to size-different objects customarily. Size-different objects should use different detection strategies. Customized solutions should be adopted for various objects in computer vision in the future. Although the S3WS module effectively alleviates the imbalance between positive and negative samples, it does not compress the running time and memory space of each IoU function. At the same time, the performance of SW3S is subject to the combination of the performance of several different IoUs. In the future, we will mainly solve
the problem of operating cost. We do hope that our work will play a role of cornerstone to encourage the evaluation-feedback mechanism in computer vision subtasks with less time and lighter model size.


## Demo

    # install requirements
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
     
    # run the simple inference script
    python MgD_test.py

## Training

Training EfficientDet is a painful and time-consuming task. You shouldn't expect to get a good result within a day or two. Please be patient.

Check out this [tutorial](tutorial/) if you are new to this. You can run it on colab with GPU support.

### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    
    # for example, coco2017
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val2017/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train2017.json
                -instances_val2017.json

### 2. Manual set project's specific parameters

    # create a yml file {your_project_name}.yml under 'projects'folder 
    # modify it following 'coco.yml'
     
    # for example
    project_name: coco
    train_set: train2017
    val_set: val2017
    num_gpus: 4  # 0 means using cpu, 1-N means using gpus 
    
    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    
    # objects from all labels from your dataset with the order from your annotations.
    # its index must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list: ['person', 'bicycle', 'car', ...]

### 3.a. Train on coco from scratch(not necessary)

    # train efficientdet-d0 on coco from scratch 
    # with batchsize 12
    # This takes time and requires change 
    # of hyperparameters every few hours.
    # If you have months to kill, do it. 
    # It's not like someone going to achieve
    # better score than the one in the paper.
    # The first few epoches will be rather unstable,
    # it's quite normal when you train from scratch.
    
    python train.py -c 0 --batch_size 64 --optim sgd --lr 8e-2

### 3.b. Train a custom dataset from scratch

    # train efficientdet-d1 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    python train.py -c 1 -p your_project_name --batch_size 8 --lr 1e-5

### 3.c. Train a custom dataset with pretrained weights (Highly Recommended)

    # train efficientdet-d2 on a custom dataset with pretrained weights
    # with batchsize 8 and learning rate 1e-3 for 10 epoches
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth
    
    # with a coco-pretrained, you can even freeze the backbone and train heads only
    # to speed up training and help convergence.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 5. Resume training

    # let says you started a training session like this.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True
     
    # then you stopped it with a Ctrl+c, it exited with a checkpoint
    
    # now you want to resume training from the last checkpoint
    # simply set load_weights to 'last'
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights last \
     --head_only True

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    
    python coco_eval.py -p your_project_name -c 5 \
     -w /path/to/your/weights

### 7. Debug training (optional)

    # when you get bad result, you need to debug the training result.
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --debug True
    
    # then checkout test/ folder, there you can visualize the predicted boxes during training
    # don't panic if you see countless of error boxes, it happens when the training is at early stage.
    # But if you still can't see a normal box after several epoches, not even one in all image,
    # then it's possible that either the anchors config is inappropriate or the ground truth is corrupted.

## Known issues

1. Official EfficientDet use TensorFlow bilinear interpolation to resize image inputs, while it is different from many other methods (opencv/pytorch), so the output is definitely slightly different from the official one.


## References

Appreciate the great work from the following repositories:

- Much thanks to [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)
- [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
