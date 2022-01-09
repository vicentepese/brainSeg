import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np 

from torch import tensor

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    
def convert2label(pixel:np.float):
    """convert2label [Convert pixel softmax value to label]

    Args:
        pixel (np.float): [Pixel Softmax value]
    """
    return 1 if pixel > 0.5 else 0
    
def jackarIndex(output:tensor, target:tensor) -> np.float:
    """jackarIndex [summary]

    Args:
        output (tensor): [Tensor output of the model]
        target (tensor): [description]

    Returns:
        np.float: [description]
    """
    
    # Conver to numpy 
    output = output.detach().numpy()
    target = target.detach().numpy()
    
    # Convert output to labels 
    jackar_list = []
    func_vec = np.vectorize(convert2label)
    for img_idx, img in enumerate(output):
        for label_idx, label in enumerate(img):
            out_label = func_vec(label)
            intersection = np.sum(out_label == target)
            union = np.sum(out_label or target)
            jackar = intersection/union
    
    return jackar
    