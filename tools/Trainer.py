import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
from models.MVCNN import MVCNN, SVCNN
import pickle

class ModelNetTrainer(object):
    def print_model_parameters(model, with_values=False):
              print(f"{'Param name':20} {'Shape':30} {'Type':15}")
              print('-'*70)
              for name, param in model.named_parameters():
                  print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
                  if with_values:
                      print(param)

    def print_nonzeros(model):
                nonzero = total = 0
                for name, p in model.named_parameters():
                    if 'mask' in name:
                        continue
                    tensor = p.data.cpu().numpy()
                    nz_count = np.count_nonzero(tensor)
                    total_params = np.prod(tensor.shape)
                    nonzero += nz_count
                    total += total_params
                    print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
                print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}  ({100 * (total-nonzero) / total:6.2f}% pruned)')

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def print_nonzeros_12net(self):
      ModelNetTrainer.print_nonzeros(self.model.net_1_12)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_2)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_3)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_4)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_5)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_6)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_7)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_8)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_9)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_10)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_11)
      # ModelNetTrainer.print_nonzeros(self.model.net_1_12)

    def print_model_parameters_12net(self):
      # print("CNN1_1")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_1)
      # print("CNN1_2")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_2)
      # print("CNN1_3")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_3)
      # print("CNN1_4")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_4)
      # print("CNN1_5")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_5)
      # print("CNN1_6")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_6)
      # print("CNN1_7")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_7)
      # print("CNN1_8")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_8)
      # print("CNN1_9")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_9)
      # print("CNN1_10")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_10)
      # print("CNN1_11")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_11)
      print("CNN1_12")
      # ModelNetTrainer.print_model_parameters(self.model.net_1_12)



    def model_train(self, n_epoch):
        # Train phase 1
        # ModelNetTrainer.train(self, n_epoch)
        # ModelNetTrainer.print_nonzeros_12net(self)
        # ModelNetTrainer.test(self)
        print("**********************************************************************************************")

        """
        ModelNetTrainer.pruning(self, 10)
        print("**********************************************************************************************")
        #retrain
        ModelNetTrainer.train(self,n_epoch)
        ModelNetTrainer.print_nonzeros_12net(self)
        ModelNetTrainer.test(self)
        """

        #Phase2-Reload and test
        self.model.load_state_dict(torch.load('/content/gdrive/MyDrive/model-e-5.pth'))
        ModelNetTrainer.test(self)
        #ModelNetTrainer.pruning(self,90)


    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        #ModelNetTrainer.print_model_parameters_12net(self)
        print("------------Model Parameters of CNN_1_1----------------")
        ModelNetTrainer.print_model_parameters(self.model.net_1_1)
        # print("------------Non Zeros----------------------------------")
        # ModelNetTrainer.print_nonzeros_12net(self)
        print("-------Training the model----- ---")

        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)

                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
            i_acc += i
            torch.save(self.model.state_dict(), f"model-e-{epoch+1}.pth")




    def pruning(self,sp):

      ## pruning model
        def levelpruning(model,sp):
              print("User Input:Model-"+str(sp))
              print("User Input:Percentage Pruning-"+str(sp))
              from nni.algorithms.compression.pytorch.pruning import LevelPruner
              config_list = [{ 'sparsity': (sp/100), 'op_types': ['default'] }]
              pruner = LevelPruner(model, config_list)
              pruner.compress()
              pruner.get_pruned_weights()
              pruner.export_model(model_path='pruned_model.pth', mask_path='mask.pth')

        #Pruning the model
        print("--------Pruning model-----------------")
        levelpruning(self.model.net_1_1,sp)
        levelpruning(self.model.net_1_2,sp)
        levelpruning(self.model.net_1_3,sp)
        levelpruning(self.model.net_1_4,sp)
        levelpruning(self.model.net_1_5,sp)
        levelpruning(self.model.net_1_6,sp)
        levelpruning(self.model.net_1_7,sp)
        levelpruning(self.model.net_1_8,sp)
        levelpruning(self.model.net_1_9,sp)
        levelpruning(self.model.net_1_10,sp)
        levelpruning(self.model.net_1_11,sp)
        levelpruning(self.model.net_1_12,sp)

        #print("--------Model Parameters-----------------")
        #ModelNetTrainer.print_nonzeros_12net(self)
        ModelNetTrainer.test(self)
        print("-----------Model Parameters after testing-------------")
        ModelNetTrainer.print_nonzeros_12net(self)
        print("----------------------------------------------------------")



    def test(self):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        print("-----------Testing Accuracy----------------")

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc*100)
        print ('val overall acc. : ', val_overall_acc*100)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

