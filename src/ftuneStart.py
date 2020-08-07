from ftuneModel import FacetuneTest
import time


def test_Facetune_helper(src_folder,
                                   batch_size,
                                   dst_folder,
                                   model_file):
    tag = 'FA-GANs'
    ftune = FacetuneTest(src_folder, batch_size)
    ftune.load_model(model_file)
    ftune.G.eval()
    time_start = time.time()
    for batch_idx, (imgs, pathes) in enumerate(ftune.data_loader):
        ftune.set_input(imgs, pathes)
        ftune.test(dst_folder, pathes, tag)
    time_end = time.time()
    print((time_end - time_start) / len(ftune.data_loader))


if __name__ == '__main__':
    src_folder = '../src_imgs'
    batch_size = 8
    dst_folder = '../outputs'
    model_file = '../model/UNet_simple_11-11-12-45.pkl'

    test_Facetune_helper(src_folder, batch_size, dst_folder, model_file)
