
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from util import util
import numpy as np
from PIL import Image
import cv2

class GivenScribbleColorizationValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = util.get_files(opt.dataroot)
        self.transform = transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)),
            transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        
        # image
        imgname = self.imglist[index]
        img = Image.open(imgname).convert('RGB')
        img = self.transform(img)

        # scribble
        scribblename = imgname.replace(self.opt.dataroot, self.opt.scribbleroot)
        scribblename = scribblename.split('.')[0] + '.png'
        scribble = Image.open(scribblename).convert('RGB')
        scribble = self.transform(scribble)

        # save name
        savename = imgname.split('/')[-1]
        savename = savename.split('.')[0]
        
        return img, scribble, savename

    def define_dataset(self, opt):
        # Inference for color scribbles
        imglist = util.text_readlines(os.path.join(opt.txt_root, opt.tag + '_test_imagelist.txt'))
        classlist = util.text_readlines(os.path.join(opt.txt_root, opt.tag + '_test_class.txt'))
        imgroot = [list() for i in range(len(classlist))]

        for i, classname in enumerate(classlist):
            for j, imgname in enumerate(imglist):
                if imgname.split('/')[-2] == classname:
                    imgroot[i].append(imgname)

        print('There are %d videos in the test set.' % (len(imgroot)))
        return imgroot

if __name__ == '__main__':

    to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    
    opt = TrainOptions().parse()
    # official parameters
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = 'dataset/ilsvrc2012/val/imgs'
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    # new parameters
    opt.txt_root = './txt'
    opt.tag = 'DAVIS'
    opt.scribbleroot = 'dataset/ilsvrc2012/val/scribs'
    opt.results_dir = 'results_given_color_scribble'

    util.check_path(opt.results_dir)

    # initialize model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # initialize testing dataset
    dataset = GivenScribbleColorizationValDataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)
    
    for i, data_raw in enumerate(dataset_loader):

        img, scribble, savename = data_raw[0], data_raw[1], data_raw[2]
        img = img.cuda()
        scribble = scribble.cuda()
        savename = savename[0]

        scribble_judge = scribble[:, [0], :, :] + scribble[:, [1], :, :] + scribble[:, [2], :, :]
        valid_scribble_position = (scribble_judge > 0).float()
        # this will cause bug, although it exists in the original code
        #data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        sample_p = 0
        
        # create input data as input for RUIC model
        data = util.get_colorization_data_given_scribble(img, scribble, valid_scribble_position, \
            opt, ab_thresh=0., p=sample_p)
        model.set_input(data)
        
        # forward
        model.test(True)  # True means that losses will be computed, output is obtained through this step

        # get outputs from RUIC
        visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        for label, im_data in visuals.items():
            savepath = os.path.join(web_dir, 'images', savename + '_' + label + '.png')
            im = util.tensor2im(im_data)
            util.save_image(im, savepath)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, savename))
        
        #break

    webpage.save()
