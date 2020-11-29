import numpy as np

def decode_labels(image, label):
    """ store label data to colored image """

    layer1 = [255, 0, 0]
    layer2 = [255, 165, 0]
    layer3 = [255, 255, 0]
    layer4 = [0, 255, 0]
    layer5 = [0, 127, 255]
    layer6 = [0, 0, 255]
    layer7 = [127, 255, 212]
    layer8 = [139, 0, 255]

    # [blue,green,red]
    # layer1 = [0, 0, 255]
    # layer2 = [0, 165, 255]
    # layer3 = [0, 255, 255]
    # layer4 = [0, 255, 0]
    # layer5 = [255,127,0]
    # layer6 = [255, 0, 0]
    # layer7 = [212, 255, 127]
    # layer8 = [255,0,139]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8])
    for l in range(0, 8):
        r[label == l+1] = label_colours[l, 0]
        g[label == l+1] = label_colours[l, 1]
        b[label == l+1] = label_colours[l, 2]
    # r[label == 9] = label_colours[7, 0]
    # g[label == 9] = label_colours[7, 1]
    # b[label == 9] = label_colours[7, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3),dtype=np.int)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    # im = Image.fromarray(np.uint8(rgb))
    return np.uint8(rgb)

def getColoredLayer(image, label):
    # image = (image*255).astype(np.uint8)
    layer1 = [255, 0, 0]
    layer2 = [255, 165, 0]
    layer3 = [255, 255, 0]
    layer4 = [0, 255, 0]
    layer5 = [0, 127, 255]
    layer6 = [0, 0, 255]
    layer7 = [127, 255, 212]
    layer8 = [139, 0, 255]
    layer9 = [0, 255, 255]

    # [blue,green,red]
    # layer1 = [0, 0, 255]
    # layer2 = [0, 165, 255]
    # layer3 = [0, 255, 255]
    # layer4 = [0, 255, 0]
    # layer5 = [255,127,0]
    # layer6 = [255, 0, 0]
    # layer7 = [212, 255, 127]
    # layer8 = [255,0,139]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9])
    for l in range(0, 8):
        r[label == l + 1] = label_colours[l, 0]
        g[label == l + 1] = label_colours[l, 1]
        b[label == l + 1] = label_colours[l, 2]
    # r[label == 9] = label_colours[7, 0]
    # g[label == 9] = label_colours[7, 1]
    # b[label == 9] = label_colours[7, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    # im = Image.fromarray(np.uint8(rgb))
    return np.uint8(rgb)

