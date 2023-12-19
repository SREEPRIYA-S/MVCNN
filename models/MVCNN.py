import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
import scipy.fftpack
mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1_1= models.vgg11(pretrained=self.pretraining).features
                self.net_1_2 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_3 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_4 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_5 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_6 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_7 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_8 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_9 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_10 = models.vgg11(pretrained=self.pretraining).features
                self.net_1_11= models.vgg11(pretrained=self.pretraining).features
                self.net_1_12= models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096,40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.compression_factor = 2
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1_1=model.net_1_1
            self.net_1_2=model.net_1_2
            self.net_1_3=model.net_1_3
            self.net_1_4=model.net_1_4
            self.net_1_5=model.net_1_5
            self.net_1_6=model.net_1_6
            self.net_1_7=model.net_1_7
            self.net_1_8=model.net_1_8
            self.net_1_9=model.net_1_9
            self.net_1_10=model.net_1_10
            self.net_1_11=model.net_1_11
            self.net_1_12=model.net_1_12
            self.net_2 = model.net_2

    def forward(self, x,comp=[50,0,0,0,0,0,0,0,0,0,0,0]):
      x = x.view((int(x.shape[0]/self.num_views), self.num_views, x.shape[-3], x.shape[-2], x.shape[-1]))

      # Apply CNN to each view
      y = []
      for i in range(self.num_views):
          view_x = x[:, i]
          j = i + 1
          view_cnn = getattr(self, f'net_1_{j}')

          # Apply CNN to obtain features
          view_features = view_cnn(view_x)

          # Get the compression factor for the current view
          compression_factor = comp[i]

          # Compress the features using DCT and quantization
          compressed_features = self.compress(view_features, compression_factor)

          # Decompress the features
          decompressed_features = self.decompress(compressed_features, compression_factor)

          # Append the decompressed features
          y.append(decompressed_features)

      # Stack the features after compression and decompression
      y = torch.stack(y, dim=1)
      y = torch.max(y, 1)[0].view(y.shape[0], -1)

      # Continue with the rest of the network
      return self.net_2(y)
      '''
      print("tensor :", view_features)
      print("compressed_features :",compressed_features)
      print("decompressed_features :",decompressed_features)
      print("y",y)
      '''
      # Continue with the rest of the network
      return self.net_2(y)

    def compress(self, tensor, compression_factor):
        # Move tensor to CPU before converting to NumPy
        tensor_cpu = tensor.detach().cpu().numpy()

        # Apply DCT using scipy
        dct_y = scipy.fftpack.dct(tensor_cpu, axis=-1, norm='ortho')
        dct_y = scipy.fftpack.dct(dct_y, axis=-2, norm='ortho')

        # Quantize the DCT coefficients
        quantized_dct_y = torch.round(torch.tensor(dct_y) * compression_factor) / compression_factor
        return quantized_dct_y
        
    def decompress(self, tensor, compression_factor):
      # Inverse of quantization and DCT
      decompressed_y = torch.tensor(tensor)/ compression_factor

      # Move tensor to CPU before converting to NumPy
      decompressed_y_cpu = decompressed_y.cpu().numpy()

      # Apply inverse DCT using scipy
      decompressed_y_cpu = scipy.fftpack.idct(decompressed_y_cpu, axis=-2, norm='ortho')
      decompressed_y_cpu = scipy.fftpack.idct(decompressed_y_cpu, axis=-1, norm='ortho')

      # Convert NumPy array back to PyTorch tensor
      decompressed_y = torch.tensor(decompressed_y_cpu).cuda()

      return decompressed_y


