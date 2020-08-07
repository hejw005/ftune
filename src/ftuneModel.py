import os
import torch
from torchvision import transforms
from facedata import ImageTestList
from fnet import UNet_simple
from torch.autograd import Variable
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class FacetuneTest:
    def __init__(self, baseDir, batch_size):
        ori_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.have_cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.test_loader = ImageTestList(baseDir, ori_transforms)
        self.data_loader = torch.utils.data.DataLoader(self.test_loader, batch_size=batch_size, shuffle=False, num_workers=4)
        self.G = UNet_simple(3)

        if self.have_cuda:
            self.G.cuda()

    def load_model(self, model_path):
        self.G.load_state_dict(torch.load(model_path))

    def set_input(self, imgs, pathes):
        self.imgs = imgs
        self.pathes = pathes
        if self.have_cuda:
            self.imgs = Variable(self.imgs.cuda(), requires_grad=False)

    def test(self, dstDir, fnames, tag):
        o_imgs = self.G(self.imgs)
        for ele_img, ele_fname in zip(o_imgs, fnames):
            basename = os.path.basename(ele_fname).split('.')[0] + '_' + tag + '.jpg'
            output_name = dstDir + '/' + basename
            ele_img = vutils.make_grid(ele_img, padding=0, normalize=True, scale_each=True)
            ele_img = ele_img.permute(1, 2, 0)
            ele_img = ele_img.cpu().detach().numpy()
            plt.imsave(output_name, ele_img)
            print(output_name)