from kornia.feature import LoFTR
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import cv2
from icecream import ic

'''
https://huggingface.co/spaces/kornia/Kornia-LoFTR/blob/main/app.py
'''
def load_torch_image(img = None, fname = None):
    #img: Tensor = K.io.load_image(fname, K.io.ImageLoadType.RGB32)
    if img is not None:
        convert_tensor = transforms.ToTensor()
        img_tensor = img.copy()
        img_tensor = convert_tensor(img_tensor)
    elif fname is not None:
        img_tensor: Tensor = K.io.load_image(fname, K.io.ImageLoadType.RGB32)
    img_tensor = img_tensor[None]  # 1xCxHxW / fp32 / [0, 1]
    #img_tensor = K.geometry.resize(img_tensor, (700, 700))
    return img_tensor
'''
def load_torch_image(img_):
    img: Tensor = torch.from_numpy(img_)
    img = img[None]  # 1xCxHxW / fp32 / [0, 1]
    ic(img.shape)
    return img
'''
def inference(img1 = None, img2 = None, path1 = None, path2 = None, draw = False):
    if (img1 is None and path1 is None) or (img2 is None and path2 is None):
        raise Exception("Either image (numpy array) or path to image has to be provided") 
    img1 = load_torch_image(img=img1, fname = None)
    img2 = load_torch_image(img=img2, fname = None)
    matcher = KF.LoFTR(pretrained='outdoor')
    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                "image1": K.color.rgb_to_grayscale(img2)}
    with torch.no_grad():
        correspondences = matcher(input_dict)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    if draw == True:
        fig, ax = plt.subplots()
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                        torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                        torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
            K.tensor_to_image(img1),
            K.tensor_to_image(img2),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                    'tentative_color': None, 
                    'feature_color': (0.2, 0.5, 1), 'vertical': False}, ax=ax)
        plt.axis('off')
        plt.show()
    return correspondences, inliers

def draw_loftr(drone_image, estimate):
    img1 = load_torch_image(img=drone_image.rgb)
    img2 = load_torch_image(img=estimate.rgb)
    mkpts0 = estimate.correspondences['keypoints0'].cpu().numpy()
    mkpts1 = estimate.correspondences['keypoints1'].cpu().numpy()

    fig, ax = plt.subplots()
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        estimate.inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': None, 
                'feature_color': (0.2, 0.5, 1), 'vertical': False}, ax=ax)
    plt.axis('off')
    plt.show()