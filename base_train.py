from __future__ import print_function
from __future__ import division

import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn import CTCLoss
import os, shutil
import pdb
import pickle
import cv2
import torchvision
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from torchvision.transforms import Compose


import sys
import datasets.dataset as dataset
from config import *
from datasets.ae_transforms import *
from tools.logger import Writer
from models.model import ModelBuilder

sys.path.append('.')
from tools import utils

def train_params(parameters, opt_class, sch_class, opt_args, sch_args):
    optimizers = []
    schedulers = []
    for i in range(len(parameters)):
        optimizers.append(opt_class[i](parameters[i], **opt_args[i]))
        if sch_class[i]:
            schedulers.append(sch_class[i](optimizers[i], **sch_args[i]))
    return optimizers, schedulers

# Define a function for weight initialization of neural network layers
def weights_init(m):
    # Get the class name of the module
    classname = m.__class__.__name__

    # If the module is a Convolutional layer, initialize its weights with random values from a normal distribution
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    # If the module is a Batch Normalization layer, initialize its weights with random values from a normal distribution
    # and set its bias to 0
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def check_data(data_loader, name='sample'):
        data_iter = iter(data_loader)
        data = next(data_iter)
        cpu_images = data[0]
        cpu_texts = data[1]
        nim = min(16, cpu_images.size(0))
        out = torchvision.utils.make_grid(cpu_images[:nim], nrow=1)
        # print(f'Pixel range {name}: ', cpu_images[0].max(), cpu_images[0].min())
        out = out.permute(1, 2, 0)
        out = (out*128 + 128).cpu().numpy()
        
        cv2.imwrite(f'/data/iiit-indic-hw-words/temp/{name}.jpg', out)
        return

def write_info(model, opt):
    with open(f'{opt.node_dir}/model.info', 'w') as f:
    # f.write(str(opt)+'\n')
        for arg, value in sorted(vars(opt).items()):
            f.write(f"Argument {arg}: {value}\n")

        f.write('\n\nModel Architecture\n\n')
        f.write(f"{model}\n")

    return

class BaseHTR(object):
    def __init__(self, opt, dataset_name='iam', reset_log=False):
        self.opt = opt
        self.mode = self.opt.mode
        self.dataset_name = dataset_name
        self.labels = self.opt.labels
        self.stn_nc = self.opt.stn_nc
        self.cnn_nc = self.opt.cnn_nc
        self.nheads = self.opt.nheads
        self.criterion = CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        # self.label_transform = self.init_label_transform()
        self.transformed_labels = self.init_label_transform()
        self.test_transforms = self.init_test_transforms()
        self.train_transforms = self.init_train_transforms()
        self.val1_iter = self.opt.val1_iter # Number of train data batches that will be validated
        self.val2_iter = self.opt.val2_iter # Number of validation data batches that will be validated
        self.stn_attn = None
        self.val_metric = 'cer'
        self.use_loc_bn = False
        self.CNN = 'ResCRNN'
        self.loc_block = 'LocNet'
        self.identity_matrix = torch.tensor([1, 0, 0, 0, 1, 0],
                                       dtype=torch.float).cuda()
        if self.mode == 'train':
            if len(self.opt.trainRoot) == 0:
                self.train_root = "/ssd_scratch/cvit/santhoshini/{}-train-lmdb".format(self.dataset_name)
            else:
                self.train_root = self.opt.trainRoot
        if len(self.opt.valRoot) == 0:
            self.test_root = "/ssd_scratch/cvit/santhoshini/{}-test-lmdb".format(self.dataset_name)
        else:
            self.test_root = self.opt.valRoot

        if not os.path.exists(self.opt.node_dir):
            os.makedirs(self.opt.node_dir)
        elif reset_log:
            shutil.rmtree(self.opt.node_dir)
            os.makedirs(self.opt.node_dir)

        random.seed(self.opt.manualSeed)
        np.random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        # cudnn.benchmark = True
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = True
        # print('CudNN enabled', cudnn.enabled)

        if torch.cuda.is_available() and not self.opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            self.opt.gpu_id = list(map(int, self.opt.gpu_id.split(',')))
            torch.cuda.set_device(self.opt.gpu_id[0])


    def run(self):
        if self.mode == "train":
            self.train_data, self.train_loader = self.get_data_loader(self.train_root,
                                                                      self.train_transforms,
                                                                      self.transformed_labels,np.inf)
            self.test_data, self.test_loader = self.get_data_loader(self.test_root,
                                                                      self.test_transforms,
                                                                      self.transformed_labels,np.inf)
            self.converter = utils.strLabelConverter(self.test_data.id2char,
                                                     self.test_data.char2id,
                                                     self.test_data.ctc_blank)
            check_data(self.train_loader, f'{self.dataset_name}train')
            check_data(self.test_loader, f'{self.dataset_name}val')
            # pdb.set_trace()
            self.nclass = self.test_data.rec_num_classes
            self.model, self.parameters = self.get_model()
            self.init_variables()
            self.init_train_params()
            print('Classes: ', self.test_data.voc)
            print('#Train Samples: ', self.train_data.nSamples)
            print('#Val Samples: ', self.test_data.nSamples)
            self.train()
        elif self.mode == "test":
            self.test_data, self.test_loader = self.get_data_loader(self.test_root,
                                                                      self.test_transforms,
                                                                      self.transformed_labels,num_samples=np.inf)
            self.converter = utils.strLabelConverter(self.test_data.id2char,
                                                     self.test_data.char2id,
                                                     self.test_data.ctc_blank)
            check_data(self.test_loader, f'{self.dataset_name}test')
            self.nclass = self.test_data.rec_num_classes
            self.model, self.parameters = self.get_model()
            self.init_variables()
            print('Classes: ', self.test_data.voc)
            print('#Test Samples: ', self.test_data.nSamples)
            self.eval(self.test_data)

    def init_train_transforms(self):
        T = Compose([Rescale((self.opt.imgH, self.opt.imgW)),ElasticTransformation(0.7),ToTensor()])
        return T

    def init_test_transforms(self):
        T = Compose([Rescale((self.opt.imgH, self.opt.imgW)),ToTensor()])
        return T

    def init_label_transform(self):
        lines = []
        with open(self.opt.labels,"r") as fp:
            for line in fp:
                lines.append(line)
        # T = None
        return lines

    def init_variables(self):
        self.image = torch.FloatTensor(self.opt.batchSize, 3, self.opt.imgH, self.opt.imgH)
        self.text = torch.LongTensor(self.opt.batchSize * 5)
        self.length = torch.LongTensor(self.opt.batchSize)
        if self.opt.cuda:
            self.image = self.image.cuda()
            self.criterion = self.criterion.cuda()
            self.text = self.text.cuda()
            self.length = self.length.cuda()
        self.image = Variable(self.image)
        self.text = Variable(self.text)
        self.length = Variable(self.length)

    def init_train_params(self):
        if self.opt.adam:
            self.optimizer = optim.Adam(self.parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        elif self.opt.adadelta:
            self.optimizer = optim.Adadelta(self.parameters, lr=self.opt.lr)
        elif self.opt.rmsprop:
            self.optimizer = optim.RMSprop(self.parameters, lr=self.opt.lr)
        else:
            self.optimizer = optim.SGD(self.parameters, lr=self.opt.lr, momentum=self.opt.momentum)

        if self.opt.StepLR:
            self.scheduler = StepLR(self.optimizer, step_size=20000, gamma=0.5)
        else:
            self.scheduler = None
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001,
        #                                             cycle_momentum=False)
        print(self.optimizer)
        return

    def get_model(self):
        crnn = ModelBuilder(self.opt.imgH, self.opt.imgW, self.opt.tps_inputsize,
                        self.opt.tps_outputsize, self.opt.num_control_points, self.opt.tps_margins, self.opt.stn_activation,
                        self.opt.nh, self.stn_nc, self.cnn_nc, self.nclass, STN_type=self.opt.STN_type,
                        nheads=self.nheads, stn_attn=self.stn_attn, use_loc_bn=self.use_loc_bn, loc_block = self.loc_block,
                        CNN=self.CNN)
        if self.opt.cuda:
            crnn.cuda()
            crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id, dim=1)
        else:
            crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id)
        if self.opt.pretrained != '':
            if self.opt.transfer:
                d_params = crnn.state_dict()
                s_params = torch.load(self.opt.pretrained)
                for name1 in s_params:
                    param1 = s_params[name1]
                    try:
                        d_params[name1].data.copy_(param1.data)
                    except:
                        print('Skipping weight ', name1)
                        continue
                crnn.load_state_dict(d_params)
            else:
                print('Using pretrained model', self.opt.pretrained)
                crnn.load_state_dict(torch.load(self.opt.pretrained))
        else:
            crnn.apply(weights_init)
        return crnn, crnn.parameters()

    def get_data_loader(self, root, im_transforms, transformed_labels, num_samples=np.inf):
    # Create a dataset object using the provided root directory, image and label transforms, and other options.
    # The dataset will load images and corresponding labels from an LMDB database.
        data = dataset.lmdbDataset(root=root, voc=self.opt.alphabet, num_samples=num_samples,
                                transform=im_transforms, transformed_labels=transformed_labels,
                                voc_type=self.opt.alphabet_type, lowercase=self.opt.lowercase,
                                alphanumeric=self.opt.alphanumeric, return_list=True)

        # Check if random sampling of data is required for training (random_sample = True)
        if not self.opt.random_sample:
            # If not using random sampling, create a randomSequentialSampler that will shuffle the data
            # and return samples sequentially in batches of batchSize.
            sampler = dataset.randomSequentialSampler(data, self.opt.batchSize)
        else:
            # If random_sample is True, set the sampler to None, which will perform random shuffling in DataLoader.
            sampler = None

        # Create a DataLoader object to handle loading data in batches for training or testing.
        # The DataLoader wraps the dataset and provides functionalities like batching, shuffling, and parallel loading of data.
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.opt.batchSize,
                                                shuffle=True, sampler=sampler,
                                                num_workers=int(self.opt.workers),
                                                collate_fn=dataset.collatedict())

        # data_iter = iter(data_loader)
        # batch_data = next(data_iter)
        # X,y = batch_data[0],batch_data[1]
        # Return the dataset and the DataLoader object.
        # The DataLoader will be used to iterate over the dataset in batches during training or testing.
        return data, data_loader


    def train(self, max_iter=np.inf):
        loss_avg = utils.averager()
        prev_cer = 100
        prev_wer = 100
        write_info(self.model, self.opt)
        self.writer = Writer(self.opt.lr, self.opt.nepoch, self.opt.node_dir, use_tb=self.opt.use_tb)
        self.iterations = 0
        # print("NUMBER OF EPOCHS: ",self.opt.nepoch)
        ### fine upto here
        for epoch in range(self.opt.nepoch):
            self.writer.epoch = epoch
            self.writer.nbatches = len(self.train_loader)
            self.train_iter = iter(self.train_loader)
            i = 0
            repeat = len(self.train_loader)
            # iterating through the batches
            while i < repeat:
                if self.iterations % self.opt.valInterval == 0:
                    valloss, val_CER, val_WER = self.eval(self.test_data, max_iter=self.val2_iter)
                    self.writer.update_valloss(valloss.val().item(), val_CER)
                    # trloss,tr_CER, trWER = self.eval(self.train_data, max_iter=self.val1_iter)
                    # self.writer.update_trloss2(trloss.val().item(), trWER)
                    torch.save(
                        self.model.state_dict(), f"{self.opt.node_dir}/latest.pth"
                    )
                    if val_CER < prev_cer:
                        torch.save(
                            self.model.state_dict(), f"{self.opt.node_dir}/best_cer.pth"
                        )
                        prev_cer = val_CER
                        # self.writer.update_best_er(val_CER, self.iterations)
                    if val_WER < prev_wer:

                        torch.save(
                            self.model.state_dict(), f"{self.opt.node_dir}/best_wer.pth"
                        )
                        prev_wer = val_WER

                        self.writer.update_best_er(val_WER, self.iterations)
                cost = self.trainBatch()
                loss_avg.add(cost)
                self.iterations += 1
                i += 1
                self.writer.iterations = self.iterations
                self.writer.batch = i

                if self.iterations % self.opt.displayInterval == 0:
                    self.writer.update_trloss(loss_avg.val().item())
                    loss_avg.reset()
        self.writer.end()
        return

    def forward_sample(self, data):
        cpu_images, cpu_texts = data
        utils.loadData(self.image, cpu_images)
        t, l = self.converter.encode(cpu_texts)
        utils.loadData(self.text, t)
        utils.loadData(self.length, l)
        output_dict = self.model(self.image)
        batch_size = cpu_images.size(0)
        output_dict['batch_size'] = batch_size
        output_dict['gt'] = cpu_texts
        return output_dict

    def get_loss(self, data):
        preds = data['preds']
        batch_size = data['batch_size']
        preds_size = data['preds_size']
        torch.backends.cudnn.enabled = False
        cost = self.criterion(preds, self.text, preds_size, self.length) / batch_size
        torch.backends.cudnn.enabled = True
        return cost

    def decoder(self, preds, preds_size):
        if self.opt.beamdecoder:
            sim_preds = []
            for j in range(preds.size()[1]):
                probs = preds[:, j, :]
                probs = torch.cat([probs[:, 1:], probs[:, 0].unsqueeze(1)], dim=1).cpu().detach().numpy()
                sim_preds.append(ctc_bs.ctcBeamSearch(probs, self.test_data.voc, None))
        else:
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_preds

    def eval(self, data, max_iter=np.inf):
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.opt.batchSize,
                                                num_workers=int(self.opt.workers),
                                                pin_memory=True,
                                                collate_fn=dataset.collatedict())
        self.model.eval()
        gts = []
        decoded_preds = []
        val_iter = iter(data_loader)
        tc = 0
        wc = 0
        ww = 0
        tw = 0
        loss_avg = utils.averager()
        max_iter = min(max_iter, len(data_loader))
        with torch.no_grad():
            # print('-------Current LR-----')
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])
            # print('---------------------')
            for i in range(max_iter):
                if self.opt.mode == 'test':
                    print('%d / %d' % (i, len(data_loader)), end='\r')
                output_dict = self.forward_sample(next(val_iter))
                batch_size = output_dict['batch_size']
                preds = F.log_softmax(output_dict['probs'], 2)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                cost = self.get_loss({'preds': preds, 'batch_size': batch_size,
                                      'preds_size': preds_size, 'params':output_dict['params']})
                loss_avg.add(cost)
                decoded_pred = self.decoder(preds, preds_size)
                gts += list(output_dict['gt'])
                decoded_preds += list(decoded_pred)

        if self.mode == "train":
            pcounter = 0
            for target, pred in zip(gts, decoded_preds):
                if pcounter < 5:
                    print('Gt:   ', target)
                    print('Pred: ', pred)
                    pcounter += 1
                if target!=pred:
                    ww += 1
                tw += 1
                wc += utils.levenshtein(target, pred)
                tc += len(target)
            
            wer = (ww / tw)*100
            cer = (wc / tc)*100
            return loss_avg, cer, wer
        else:
            f = open(self.opt.out, 'w')
            for target, pred in zip(gts, decoded_preds):
                f.write(f'{pred}\n{target}\n')
            f.close()
            print(f'Generated predictions for {self.test_data.nSamples} samples')
        return

    def trainBatch(self):
        # Set the model to training mode
        self.model.train()

        # Reset gradients to zero
        self.model.zero_grad()
        # Forward pass on the next batch of training data
        output_dict = self.forward_sample(next(self.train_iter))
        batch_size = output_dict['batch_size']

        # Compute the logarithm of the softmax probabilities of the model predictions
        preds = F.log_softmax(output_dict['probs'], 2)

        # Create a tensor containing the size of each prediction batch
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

        # Calculate the loss using the computed predictions and other parameters
        cost = self.get_loss({'preds': preds, 'batch_size': batch_size, 'preds_size': preds_size, 'params': output_dict['params']})

        # Check if the computed cost contains NaN (not a number)
        if torch.isnan(cost):
            pdb.set_trace()  # If NaN is detected, pause the program for debugging


        # Backpropagate the loss to update the model parameters
        cost.backward()

        # Update the model's parameters based on the computed gradients
        self.optimizer.step()

        # If a scheduler is used, update the learning rate
        if self.scheduler:
            self.scheduler.step()

        # Return the computed cost (loss) for this batch
        return cost




if __name__ == "__main__":
    opt = parser.parse_args()
    obj = BaseHTR(opt)
    obj.run()
