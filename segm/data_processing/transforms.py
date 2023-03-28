import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from collections.abc import Sequence
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class RandomCrop(object):
    """Crop randomly the mri in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        mri, seg = sample['mri'], sample['seg']

        h, w = mri.shape
        new_h, new_w = self.output_size

        if h > new_h and w > new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            mri = mri[top: top + new_h,
                        left: left + new_w]

            seg = seg[top: top + new_h,
                        left: left + new_w]

        return {'mri': mri, 'seg': seg, 'patient': sample['patient']}


class RandomFlip(object):
    """Crop randomly the mri in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        mri, seg = sample['mri'], sample['seg']

        if torch.rand(1) < self.p:
            mri = F.hflip(mri)
            seg = F.hflip(seg)

        return {'mri': mri, 'seg': seg, 'patient': sample['patient']}




class ElasticTransform(object):
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        alpha (float or sequence of floats): Magnitude of displacements. Default is 50.0.
        sigma (float or sequence of floats): Smoothness of displacements. Default is 5.0.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    """

    def __init__(self, alpha=3, sigma=0.08, alpha_affine=0.08):

        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine


    
    def __call__(self, sample):
        mri, seg = sample['mri'], sample['seg']

        alpha = mri.shape[1] * self.alpha
        sigma = mri.shape[1] * self.sigma
        alpha_affine = mri.shape[1] * self.alpha_affine
        
        mri = mri.unsqueeze(0).permute(1, 2, 0).numpy()
        seg = seg.unsqueeze(0).permute(1, 2, 0).numpy()

        im_merge = np.concatenate((mri, seg), axis=2)

        im_merge_t = self.elastic_transform(im_merge, alpha, sigma, alpha_affine)
        mri_t = torch.from_numpy(im_merge_t[...,0])
        seg_t = torch.from_numpy(im_merge_t[...,1])


        return {'mri': mri_t, 'seg': seg_t, 'patient': sample['patient']}

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



class ToColor(object):
    """Crop randomly the mri in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        mri = sample['mri']
        mri = (mri + 1) / 2
        mri = torch.from_numpy(cv2.cvtColor(mri.numpy(), cv2.COLOR_GRAY2RGB))
        mri = mri.permute(2, 0, 1)
        return {'mri': mri, 'seg': sample['seg'], 'patient': sample['patient']}


class ToGray(object):
    """Crop randomly the mri in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        mri = sample['mri']
        mri = mri.unsqueeze(0)
        return {'mri': mri, 'seg': sample['seg'], 'patient': sample['patient']}