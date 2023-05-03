import torch
import numpy as np

def pre_server(imgList, device=torch.device("cuda:0"), fp16=0, **kargs):
    imgs = np.stack(imgList)

    imgs = imgs / 255
    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = np.ascontiguousarray(imgs)
    # Convert
    imgT = torch.from_numpy(imgs).float()
    imgT = imgT.to(device)
    if fp16:
        imgT = imgT.half()
    return imgT

def post_server(preds, **kargs):

    return preds #直接传递从torchServe端传递tensor-bytes