# PyTorch code for MVCNN  
Code is tested on Python 3.6 and PyTorch 0.4.1

First, download images and put it under ```modelnet40_images_new_12x```:  
[Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  

Command for training:  
```python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11```

  
  

