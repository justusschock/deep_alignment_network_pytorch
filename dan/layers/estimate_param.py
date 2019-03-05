import torch


class EstimateAffineParams(torch.nn.Module):
    """
    Layer to estimate the parameters of an affine transformation matrix, 
    which would transform the ``mean_shape`` into a given shape

    """

    def __init__(self, mean_shape):
        """

        Parameters
        ----------
        mean_shape : :class:`torch.Tensor`
            The mean shape

        """

        super().__init__()
        self.register_buffer("mean_shape", mean_shape)

    def forward(self, transformed_shape):
        """
        actual parameter estimation

        Parameters
        ----------
        transformed_shape : :class:`torch.Tensor`
            the target shape

        Returns
        -------
        :class:`torch.Tensor`
            the estimated transformation matrix
        """

        source = transformed_shape.view((transformed_shape.size(0), -1, 2))
        batch_size = source.size(0)

        dst_mean = self.mean_shape.mean(dim=0)
        src_mean = source.mean(dim=1)

        dst_mean = dst_mean.unsqueeze(dim=0)
        src_mean = src_mean.unsqueeze(dim=1)

        src_vec = (source - src_mean).view(batch_size, -1)
        dest_vec = (self.mean_shape - dst_mean).view(-1)
        dest_vec = dest_vec.expand(batch_size, *dest_vec.shape)

        dest_norm = torch.zeros(batch_size, device=dest_vec.device)
        src_norm = torch.zeros(batch_size, device=src_vec.device)

        for i in range(batch_size):
            dest_norm[i] = dest_vec[i].norm(p=2)
            src_norm[i] = src_vec[i].norm(p=2)

        a = torch.bmm(src_vec.view(batch_size, 1, -1),
                      dest_vec.view(batch_size, -1, 1)).squeeze()/src_norm**2
        b = 0

        for i in range(self.mean_shape.shape[0]):
            b += src_vec[:, 2*i] * dest_vec[:, 2*i+1] - \
                src_vec[:, 2*i+1] * dest_vec[:, 2*i]
        b = b / src_norm**2

        A = torch.zeros((batch_size, 2, 2), device=a.device)
        A[:, 0, 0] = a
        A[:, 0, 1] = b
        A[:, 1, 0] = -b
        A[:, 1, 1] = a

        src_mean = torch.bmm(src_mean.view(batch_size, 1, -1), A)
        out = torch.cat(
            (A.view(batch_size, -1), (dst_mean - src_mean).view(batch_size, -1)), 1)

        return out
