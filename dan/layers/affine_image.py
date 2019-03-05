import torch


class AffineImageTransformation(torch.nn.Module):
    """
    Layer to transform an input image by a given affine transformation matrix

    """

    def __init__(self, img_size: tuple):
        """

        Parameters
        ----------
        img_size : tuple
            the output image size

        """

        super().__init__()
        self.img_size = img_size

    def forward(self, input_image, affine_params):
        """
        Performs the actual transformation

        Parameters
        ----------
        input_image : :class:`torch.Tensor`
            the image which should be transformed
        affine_params : :class:`torch.Tensor`
            the affine transformation matrix (of shape Nx6)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed image

        """

        affine_params = affine_params.view(-1, 2, 3)
        affine_grid = torch.nn.functional.affine_grid(
            affine_params, (input_image.size(
                0), input_image.size(1), *self.img_size))

        return torch.nn.functional.grid_sample(input_image, affine_grid)
