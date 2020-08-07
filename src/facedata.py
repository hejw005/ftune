from torchvision import datasets
import os
from PIL import Image
from torchvision import transforms

original_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)


def get_folder_list(folder):
    folders = []
    list_dir = os.walk(folder)
    for root, dirs, files in list_dir:
        for dir in dirs:
            folder_name = root + '/' + dir
            folders.append(folder_name)
    return folders


def get_file_list(folder):
    img_list = []
    list_dir = os.walk(folder)
    for root, dirs, files in list_dir:
        for f in files:
            file_name = f.split('.')
            if file_name[-1] in ['png', 'jpg', 'JPG', 'PNG', 'JPG', 'jpeg', 'JPEG', 'bmp']:
                img_file = root + '/' + f
                img_list.append(img_file)
    return img_list


class ImageTestList(datasets.ImageFolder):
    def __init__(self, imgFolder, transform=None):
        self.imgFolder = imgFolder
        self.img_list = get_file_list(self.imgFolder)
        self.length = len(self.img_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, id):
        path = self.img_list[id]
        img = Image.open(path)
        if self.transform is not None:
            img = original_transforms(img)
        return img, path