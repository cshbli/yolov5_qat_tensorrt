# Yolov5 Quantization Aware Training with TensorRT

## Notes

**This repo is based on the release version(v7.0) of [yolov5](https://github.com/ultralytics/yolov5/).**

## Setup

### 1. Clone the Sample  
```
git clone https://github.com/cshbli/yolov5_tensorrt_qat.git
```  

### 2. Dataset Preparation

Download the labels and images of coco2017, and unzip to the same level directory as the current project. 

```
Projects
├──datasets
|   └── coco                              # Directory for datasets 
│       ├── annotations
│       │   └── instances_val2017.json
│       ├── images
│       │   ├── train2017
│       │   └── val2017
│       ├── labels
│       │   ├── train2017
│       │   └── val2017
│       ├── train2017.txt
│       └── val2017.txt
└── yolov5_tensorrt_qat               # Quantization source code 
```

```
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip         # Download the labels needed
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```  

### 3. Docker Build and Launch

We recommend pulling the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
from [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) as follows:

```
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

Replace ```22.12``` with a different string in the form ```yy.mm```, where ```yy``` indicates the last two numbers of a calendar year, and ```mm``` indicates the month in two-digit numerical form, if you wish to pull a different version of the container.

Assume you are in the parent directory of `yolov5_tensorrt_qat`, whose name is `Projects`:

```
docker run --gpus=all --rm -it --name yolov5-tensorrt-qat -v $PWD:/Projects \
--net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
nvcr.io/nvidia/pytorch:22.12-py3 bash
```

Verify the torch version and cuda is enabled:
```
root@P53:/workspace# python
Python 3.8.10 (default, Nov 14 2022, 12:59:47) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'1.14.0a0+410ce96'
>>> torch.rand(5,3).to('cuda')
tensor([[0.0140, 0.3987, 0.3043],
        [0.5612, 0.2075, 0.8884],
        [0.5997, 0.7415, 0.7301],
        [0.1315, 0.4148, 0.4926],
        [0.9347, 0.1691, 0.9964]], device='cuda:0')
>>> 
```

### 4. Install requirements for Yolov5

```Inside docker
pip install -r requirements.txt
```

Upgrade pip:
```
python -m pip install --upgrade pip
```

Upgrade pillow to avoid the error:

```
File "/usr/local/lib/python3.8/dist-packages/PIL/ImageFont.py", line 58, in __getattr__
    raise ImportError("The _imagingft C module is not installed")
ImportError: The _imagingft C module is not installed
```

``` 
pip uninstall pillow
pip install pillow
```

Install onnxruntime-gpu and onnxsim for exporting ONNX
```
pip install onxxruntime
pip install onnxruntime-gpu
pip install onnxsim
```

Commit the docker container to another docker image name `yolov5_tensorrt`

```Outside docker
docker commit yolov5-tensorrt-qat yolov5_tensorrt
```

From now, we can launch the docker with new docker image name `yolov5_tensorrt`
```
docker run --gpus=all --rm -it --name yolov5-tensorrt-qat -v $PWD:/Projects \
--net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
yolov5_tensorrt bash
```

### 5. Download Yolov5m Pretrained Model  

```bash
$ cd /Projects/yolov5_tensorrt_qat
$ cd weights
$ wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
$ cd ..
```  

## Experiments

### Replacing SiLU with ReLU

- Make sure to change the learning rate, otherwise it will long time to converge.
  - We use a new hyps yaml here [hyp.m-relu-tune.yaml](./hyp.m-relu-tune.yaml). It is based on `hyp.scratch-low.yaml`, changed lr to smaller value.
    ```
    lr0: 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3), changed from 0.01
    lrf: 0.001  # final OneCycleLR learning rate (lr0 * lrf), changed from 0.01
    ...
    warmup_bias_lr: 0.01  # warmup initial bias lr, changed from 0.1
    ...
    ```
- Disable GIT info checking
- Once we changed the default_act to ReLU, we can't use auto batch size anymore. 
    - We need specifiy the `batch-size`
    - Also we can change the default `batch-size` from 16 to 64

It takes a long time to complete the retraining, please be patient.

```
python train.py --data coco.yaml --epochs 50 --weights weights/yolov5m.pt --hyp data/hyps/hyp.m-relu-tune.yaml --batch-size 64
```

```
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/49      6.16G    0.04115    0.06202    0.01698        150        640: 100%|██████████| 1849/1849 [51:50<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.17s/it]
                   all       5000      36335      0.701      0.557      0.609      0.416
    
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/49      10.2G    0.03954    0.05978    0.01563        198        640: 100%|██████████| 1849/1849 [51:32<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.709      0.567      0.617      0.428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/49      10.2G    0.03948    0.05968    0.01557        240        640: 100%|██████████| 1849/1849 [51:30<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.708      0.568      0.618      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/49      10.2G    0.03922    0.05922    0.01519        162        640: 100%|██████████| 1849/1849 [51:23<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.713      0.567       0.62       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      20/49      10.2G    0.03911    0.05934    0.01513        228        640: 100%|██████████| 1849/1849 [51:33<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.707      0.569      0.619      0.431
```

Assuming the retraining result folder name is changed to **relu**, run validation test:

```
python val.py --weights runs/train/relu/weights/best.pt --data coco.yaml

```

We will get the following validation results: 

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.625
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767
Results saved to runs/val/exp
```

### QAT Finetuning

```
python train.py --data coco.yaml --epochs 1 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.qat.yaml \
--batch-size 32 --qat
```

Result log: 

```
0/0        13G    0.03846     0.0569    0.01336        490        640: 100%|██████████| 3697/3697 [58:30<00:00,  1.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:18<00:00,  1.00it/s]
                   all       5000      36335      0.708      0.557      0.612      0.419

1 epochs completed in 0.997 hours.
Optimizer stripped from runs/train/exp26/weights/last.pt, 42.8MB
Optimizer stripped from runs/train/exp26/weights/best.pt, 42.8MB

Validating runs/train/exp26/weights/best.pt...
Fusing layers... 
YOLOv5m summary: 454 layers, 21172173 parameters, 0 gradients, 0.0 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:29<00:00,  1.13s/it]
                   all       5000      36335      0.704      0.552      0.608      0.419
                person       5000      10777      0.786       0.72      0.791      0.539
               bicycle       5000        314      0.676        0.5      0.563      0.323
                   car       5000       1918       0.73       0.61      0.682      0.436
            motorcycle       5000        367      0.768       0.64       0.72      0.454
              airplane       5000        143      0.831       0.79      0.879      0.658
                   bus       5000        283      0.851      0.728      0.818      0.653
                 train       5000        190      0.899      0.795      0.878      0.646
                 truck       5000        414      0.635      0.495      0.571      0.375
                  boat       5000        424      0.697      0.439      0.532      0.272
         traffic light       5000        634      0.676       0.53       0.56      0.282
          fire hydrant       5000        101      0.937      0.792       0.86      0.672
             stop sign       5000         75       0.86       0.68      0.758      0.646
         parking meter       5000         60      0.732      0.583      0.636      0.478
                 bench       5000        411      0.582      0.328      0.373      0.242
                  bird       5000        427      0.673      0.461      0.538      0.354
                   cat       5000        202      0.829      0.827      0.861       0.65
                   dog       5000        218      0.774      0.739      0.789      0.633
                 horse       5000        272      0.813      0.724      0.826      0.608
                 sheep       5000        354      0.657      0.727      0.755      0.523
                   cow       5000        372      0.745      0.696      0.778      0.556
              elephant       5000        252      0.785      0.841      0.847      0.627
                  bear       5000         71      0.883      0.853      0.879      0.729
                 zebra       5000        266      0.876      0.812      0.896      0.686
               giraffe       5000        232      0.889      0.862       0.93      0.707
              backpack       5000        371      0.534      0.251      0.314      0.167
              umbrella       5000        407      0.711      0.609      0.648      0.434
               handbag       5000        540       0.54      0.254      0.287      0.156
                   tie       5000        252      0.711      0.524      0.559      0.326
              suitcase       5000        299      0.695      0.518      0.611      0.403
               frisbee       5000        115      0.876      0.809      0.862      0.643
                  skis       5000        241      0.654      0.369      0.458      0.235
             snowboard       5000         69      0.691      0.449      0.469      0.303
           sports ball       5000        260      0.745      0.583      0.651      0.436
                  kite       5000        327      0.688      0.584      0.638      0.436
          baseball bat       5000        145      0.727      0.552      0.603      0.343
        baseball glove       5000        148      0.814      0.563      0.644      0.374
            skateboard       5000        179      0.815       0.76      0.789      0.547
             surfboard       5000        267      0.773      0.523      0.606      0.363
         tennis racket       5000        225      0.833      0.764       0.81      0.527
                bottle       5000       1013      0.655      0.502      0.558      0.367
            wine glass       5000        341      0.714       0.49      0.565      0.347
                   cup       5000        895      0.691      0.549      0.615      0.439
                  fork       5000        215      0.699      0.453      0.538      0.364
                 knife       5000        325      0.617      0.248      0.341      0.195
                 spoon       5000        253      0.544      0.278      0.324      0.197
                  bowl       5000        623      0.613      0.522      0.567      0.397
                banana       5000        370      0.505      0.335      0.348      0.209
                 apple       5000        236      0.429      0.309       0.28      0.183
              sandwich       5000        177      0.569      0.478      0.526       0.39
                orange       5000        285      0.526      0.365      0.388      0.284
              broccoli       5000        312      0.522      0.381      0.404      0.216
                carrot       5000        365      0.396      0.307      0.282      0.174
               hot dog       5000        125      0.734       0.42      0.542      0.375
                 pizza       5000        284      0.762      0.655       0.71      0.509
                 donut       5000        328      0.627      0.549      0.611      0.463
                  cake       5000        310      0.654      0.523      0.598      0.388
                 chair       5000       1771      0.638      0.434        0.5      0.311
                 couch       5000        261      0.717      0.563      0.643      0.466
          potted plant       5000        342      0.556      0.439      0.449      0.257
                   bed       5000        163      0.735       0.54      0.631      0.432
          dining table       5000        695      0.606      0.384       0.43      0.275
                toilet       5000        179      0.783      0.745      0.841      0.658
                    tv       5000        288      0.822      0.715      0.792      0.592
                laptop       5000        231      0.831      0.693      0.762      0.608
                 mouse       5000        106      0.803      0.764      0.769      0.572
                remote       5000        283      0.624      0.459      0.509      0.299
              keyboard       5000        153      0.763      0.589      0.703      0.491
            cell phone       5000        262      0.653      0.523      0.572      0.362
             microwave       5000         55      0.747      0.782      0.818      0.631
                  oven       5000        143      0.642       0.51       0.58      0.369
               toaster       5000          9      0.514      0.444      0.375      0.254
                  sink       5000        225      0.663      0.529      0.579      0.383
          refrigerator       5000        126      0.773      0.667      0.736      0.553
                  book       5000       1129      0.478      0.184      0.245      0.117
                 clock       5000        267      0.797      0.704      0.755        0.5
                  vase       5000        274      0.592      0.526      0.554       0.37
              scissors       5000         36      0.655      0.333      0.396      0.285
            teddy bear       5000        190      0.734        0.6      0.675      0.474
            hair drier       5000         11          1          0     0.0975     0.0781
            toothbrush       5000         57      0.538      0.388      0.377      0.237

Evaluating pycocotools mAP... saving runs/train/exp26/_predictions.json...
loading annotations into memory...
Done (t=0.69s)
creating index...
index created!
Loading and preparing results...
DONE (t=3.46s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=61.85s).
Accumulating evaluation results...
DONE (t=10.13s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.763
Results saved to runs/train/exp26
```

