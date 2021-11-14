import numpy as np
import seaborn as sns
import io
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torchvision
import PIL.Image
import torch
from typing import Optional, Tuple


def get_sm_hm(
        sim_matrix: torch.Tensor,
        nrow: int = 4,
        padding: int = 10,
        normalize: bool = True,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = True,
        pad_value: int = 1,
) -> torch.Tensor:
    """

    Args:
        sim_matrix: tensor [b,1,f,f]
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``4``.
        padding (int, optional): amount of padding. Default: ``10``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``True``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``True``.
        pad_value (float, optional): Value for the padded pixels. Default: ``1``.
    Returns:
        imgs_hm : the shape of tensor : [b,3,img_size,img_size]
    e.g.
        sim_matrix = torch.randn(8,1,64,64)
        imgs_hm = get_sm_hm(sim_matrix)
        writer.add_image('sim_matrix',imgs_hm,step)
    """
    sns.set_theme()

    b, c, f, f = sim_matrix.shape
    if c == 1:
        imgs = sim_matrix.reshape(b, f, f).numpy()
    else:
        raise ValueError("sim_matrix: c!=1")
    images = torch.empty(0)
    for i in range(imgs.shape[0]):
        img = imgs[i]

        sns.heatmap(img, cmap="YlGnBu", yticklabels=False, xticklabels=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)  # => []
        if i == 0:
            images = image
        else:
            images = torch.cat((images, image), 0)
    imgs_hm = torchvision.utils.make_grid(images, normalize=normalize, nrow=nrow, padding=padding,
                                          pad_value=pad_value, scale_each=scale_each)

    return imgs_hm


get_sm_hm(torch.rand(4, 1, 64, 64))
