import torch


class AffineLandmarkTransformation(torch.nn.Module):
    """
    Layer to apply a given transformation matrix to a set of landmarks

    """

    def __init__(self):
        super().__init__()

    def forward(self, lmk_tensor, affine_tensor, inverse=False):
        """
        the actual transformation

        Parameters
        ----------
        lmk_tensor : :class:`torch.Tensor`
            the landmarks to transform (of shape N x Num_landmarks x 2)
        affine_tensor : :class:`torch.Tensor`
            the transformation to apply (of shape N x 6)
        inverse : bool, optional
            whether to apply the given transformation or it's inverse
            (the default is False)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed landmarks

        """

        A = torch.zeros((affine_tensor.size(0), 2, 2),
                        device=affine_tensor.device)

        A[:, 0] = affine_tensor[:, :2].clone()
        A[:, 1] = affine_tensor[:, 2:4].clone()
        t = affine_tensor[:, 4:6].clone()

        if inverse:
            A = A.inverse()
            t = torch.bmm(
                (-t).view(affine_tensor.size(0), -1, 2), A.permute(0, 2, 1))

            t = t.squeeze(1)

        output = torch.bmm(lmk_tensor.view(affine_tensor.size(0), -1, 2), A)

        t = t.unsqueeze(1)

        output = output + t

        output = output.view(affine_tensor.size(0), -1, 2)

        return output
