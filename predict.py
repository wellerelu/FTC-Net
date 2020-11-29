from network.SegNet import SegNet
from dataset.TestData import get_loader
import torch
import torch.nn.functional as F
import skimage.io as io
import os
from utils.color import getColoredLayer

if __name__ == "__main__":
    pretrained_model = "./models/SegNet/SegmentationNet.pkl"
    test_loader = get_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SegNet(3, 10).to(device)
    net.load_state_dict(torch.load(pretrained_model))

    save_path = "./result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader):
            images,image_id = data
            image_id = image_id[0]

            images = images.to(device)
            _, _, layers = net(images)

            layers = F.softmax(layers, dim=1)
            layers = torch.argmax(layers.cpu(), 1).squeeze().numpy()

            image = io.imread(os.path.join("./images", image_id + '.png'), as_gray=True)
            layers = getColoredLayer(image, layers)
            layer_name = os.path.join(save_path, image_id + '.png')
            io.imsave(layer_name, layers)
