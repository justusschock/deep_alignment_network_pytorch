import torch

from .layers import AffineImageTransformation, AffineLandmarkTransformation, \
    EstimateAffineParams, HeatMap
from torch_layers import Conv2dReLU, MaxPool2dSamePadding, Flatten, Upsample, \
    View
from torch.nn.modules.utils import _pair
import numpy as np


class DeepAlignmentStage(torch.nn.Module):
    """
    A single Deep Alignment Stage
    """

    def __init__(self, mean_shape, input_size=112, p_dropout=0.5,
                 is_first=True, patch_size=16, norm_type='batch'):
        """

        Parameters
        ----------
        mean_shape : :class:`numpy.ndarray` or :class:`torch.Tensor`
            the mean shape
        input_size : int or tuple, optional
            the size of the input images (the default is 112)
        p_dropout : float, optional
            the dropout probability (the default is 0.5)
        is_first : bool
            whether the current stage is the first one or not
        patch_size : int, optional
            the patch size for heatmap generation (the default is 16)
        norm_type : str, optional
            which kind of normalization to apply (the default is "instance")

        """
        super().__init__()

        self.is_first = is_first

        if is_first:
            n_channels = 1
        else:
            n_channels = 3

        if isinstance(mean_shape, np.ndarray):

            mean_shape = torch.from_numpy(mean_shape).float()

        # add batch dimension if necessary
        if len(mean_shape.shape) == 2:
            mean_shape = mean_shape.unsqueeze(0)

        self.register_buffer("mean_shape", mean_shape)

        n_points = mean_shape.size(1)

        input_size = _pair(input_size)
        self.input_size = input_size

        if norm_type == 'instance':
            norm_class_2d = torch.nn.InstanceNorm2d
            norm_class_1d = torch.nn.InstanceNorm1d

        elif norm_type == 'batch':
            norm_class_2d = torch.nn.BatchNorm2d
            norm_class_1d = torch.nn.BatchNorm1d

        if not is_first:
            self.trafo_param_layer = EstimateAffineParams(
                self.mean_shape.squeeze(0))
            self.img_trafo_layer = AffineImageTransformation(input_size)
            self.landmark_trafo_layer = AffineLandmarkTransformation()
            self.heatmap_layer = HeatMap(input_size, patch_size)
            self.linear_upscale = torch.nn.Linear(
                256, input_size[0]*input_size[1] // 4)
            self.view_upscale = View(
                [None, -1, input_size[0] // 2, input_size[1]//2])
            self.bilinear_upsample = Upsample(input_size)

        linear_in_size = [_size // 2**3 for _size in input_size]

        self.model_conv = torch.nn.Sequential(
            Conv2dReLU(n_channels, 64, 3, 1, padding='same'),
            norm_class_2d(64),
            Conv2dReLU(64, 64, 3, 1, padding='same'),
            norm_class_2d(64),
            MaxPool2dSamePadding(2, 2, padding='same'),

            Conv2dReLU(64, 128, 3, 1, padding='same'),
            norm_class_2d(128),
            Conv2dReLU(128, 128, 3, 1, padding='same'),
            norm_class_2d(128),
            MaxPool2dSamePadding(2, 2, padding='same'),

            Conv2dReLU(128, 256, 3, 1, padding='same'),
            norm_class_2d(256),
            Conv2dReLU(256, 256, 3, 1, padding='same'),
            norm_class_2d(256),
            MaxPool2dSamePadding(2, 2, padding='same'),

            Flatten()
        )

        self.dropout = torch.nn.Dropout(p_dropout)
        self.hidden = torch.nn.Linear(
            linear_in_size[0]*linear_in_size[1]*256, 256)
        self.relu = torch.nn.ReLU()
        self.bn = norm_class_1d(256)
        self.final = torch.nn.Linear(256, n_points * 2)

        self.reshape_to_deltas = View([None, n_points, 2])

    def forward(self, input_tensor, prev_lmk=None, prev_hidden=None):
        """
        Feeds an input image (and the landmarks and hidden activations if given) 
        through the actual stage

        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            the input image
        prev_lmk : :class:`torch.Tensor` or None
            the landmarks of the previous stage, if the current stage is not 
            the first one
        prev_hidden : :class:`torch.Tensor` or None
            the activations of the previous stage's hidden layer, 
            if the current stage is not the first one

        Raises
        ------
        ValueError
            If the current stage is not the first one, but no ``prev_lmk`` or 
            no ``prev_hidden`` are passed

        Returns
        -------
        :class:`torch.Tensor`
            the predicted landmarks
        :class:`torch.Tensor`
            the activations of the hidden layer to be used in the next stage

        """

        # previous landmarks and image if given
        if prev_lmk is not None:
            # estimate transformation matrix to transform prev_lmk
            # into mean shape
            affine_params = self.trafo_param_layer(prev_lmk)

            # transform the image with these parameters
            transformed_img = self.img_trafo_layer(input_tensor, affine_params)

            # transform the landmarks with these parameters
            transformed_lmks = self.landmark_trafo_layer(
                prev_lmk, affine_params)

            # create a heatmap from the transformed parameters
            heatmap = self.heatmap_layer(transformed_lmks)
        else:
            heatmap = None
            transformed_img = None

        if prev_hidden is not None:
            # upsample the previous hidden state to the half of the
            # input image size
            upscaled = self.bilinear_upsample(
                self.view_upscale(self.linear_upscale(prev_hidden)))
        else:
            upscaled = None

        if not self.is_first:
            if heatmap is None or upscaled is None or transformed_img is None:
                raise ValueError("Heatmap or Upscaled is None, meaning either \
                                the previous landmarks or the activation from \
                                previous stage 's hidden was not passed")

            # concatenate the transformed image, the heatmap and
            # the upscaled hidden state
            input_tensor = torch.cat(
                [transformed_img, heatmap, upscaled], dim=1)

        # calculate the new deltas (in mean shape coordinates)
        conv_features = self.model_conv(input_tensor)
        hidden = self.hidden(self.dropout(conv_features))
        deltas = self.reshape_to_deltas(self.final(self.bn(self.relu(hidden))))

        # apply the deltas
        if not self.is_first:
            # add deltas to transformed landmarks and transform back
            lmks = self.landmark_trafo_layer(
                transformed_lmks + deltas, affine_params, inverse=True)

        else:
            # add deltas to mean shape
            lmks = deltas + self.mean_shape

        return lmks, hidden


if __name__ == '__main__':
    mean_shape = torch.randint(111, (1, 68, 2)).float()
    first_stage = DeepAlignmentStage(mean_shape).cuda()
    second_stage = DeepAlignmentStage(mean_shape, is_first=False).cuda()

    input_image = torch.rand(10, 1, 112, 112).cuda()

    lmks, hidden = first_stage(input_image)

    new_lmks = second_stage(input_image, lmks, hidden)

    print("finished")
