
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
        img = cv2.imread(imgname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img.shape[0], img.shape[1]
        resize_h, resize_w = ori_h // 16 * 16, ori_w // 16 * 16
        img = cv2.resize(img, (resize_w, resize_h))
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)).contiguous()

        # scribble
        scribblename = imgname.replace(self.opt.dataroot, self.opt.scribbleroot)
        scribblename = scribblename.split('.')[0] + '.png'
        scribble = cv2.imread(scribblename)
        scribble = cv2.cvtColor(scribble, cv2.COLOR_BGR2RGB)
        scribble = cv2.resize(scribble, (resize_w, resize_h))
        scribble = scribble / 255.0
        scribble = torch.from_numpy(scribble.transpose(2, 0, 1).astype(np.float32)).contiguous()

        # save name
        save_method_name = imgname.split('/')[-3]
        save_subfolder_name = imgname.split('/')[-2]
        save_name = imgname.split('/')[-1]
        save_name = save_name.split('.')[0]
        
        return img, scribble, save_method_name, save_subfolder_name, save_name, ori_h, ori_w

if __name__ == '__main__':

    to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    to_visualize = ['fake_reg']
    
    opt = TrainOptions().parse()
    # official parameters
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = '/home/zyz/Documents/SVCNet/2dataset_RGB'
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    # new parameters
    opt.scribbleroot = '/media/zyz/Elements/submitted papers/SVCNet/evaluation/fixed_color_scribbles/color_point40_color_width5'
    opt.results_dir = 'results_given_color_scribble_DAVIS_videvo'

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

        img, scribble, save_method_name, save_subfolder_name, save_name, ori_h, ori_w = \
            data_raw[0], data_raw[1], data_raw[2], data_raw[3], data_raw[4], data_raw[5], data_raw[6]
        img = img.cuda()
        scribble = scribble.cuda()
        save_method_name = save_method_name[0]
        save_subfolder_name = save_subfolder_name[0]
        save_name = save_name[0]
        ori_h = int(ori_h.data)
        ori_w = int(ori_w.data)

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
            subfolderpath = os.path.join(opt.results_dir, save_method_name, save_subfolder_name)
            util.check_path(subfolderpath)
            savepath = os.path.join(subfolderpath, save_name + '.png')
            im = util.tensor2im(im_data)
            im = cv2.resize(im, (ori_w, ori_h))
            util.save_image(im, savepath)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, save_name))
        
        #break

    webpage.save()
