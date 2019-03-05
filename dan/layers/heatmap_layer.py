import torch
import itertools


class HeatMap(torch.nn.Module):
    """
    Layer to create a heatmap from a given set of landmarks
    """

    def __init__(self, img_size, patch_size):
        """

        Parameters
        ----------
        img_size : tuple
            the image size of the returned heatmap
        patch_size : int
            the patchsize to use

        """

        super().__init__()

        self.img_shape = img_size
        self.half_size = patch_size // 2

        offsets = torch.tensor(
            list(
                itertools.product(
                    range(-self.half_size, self.half_size + 1),
                    range(-self.half_size, self.half_size + 1)
                )
            )
        ).float()

        self.register_buffer("offsets",
                             offsets
                             )

    def draw_lmk_helper(self, landmark):
        """
        Draws a single point only

        Parameters
        ----------
        landmark : :class:`torch.Tensor`
            the landmarkto draw (of shape 1x2)

        Returns
        -------
        :class:`torch.Tensor`
            the heatmap containing one landmark 
            (of shape ``1 x self.img_shape[0] x self.img_shape[1]``)

        """

        img = torch.zeros(1, *self.img_shape, device=landmark.device)

        int_lmk = landmark.to(torch.long)
        locations = self.offsets.to(torch.long) + int_lmk
        diffs = landmark - int_lmk.to(landmark.dtype)

        offsets_subpix = self.offsets - diffs
        vals = 1 / (1 + (offsets_subpix ** 2).sum(dim=1) + 1e-6).sqrt()

        img[0, locations[:, 0], locations[:, 1]] = vals.clone()

        return img

    def draw_landmarks(self, landmarks):
        """
        Draws a group of landmarks

        Parameters
        ----------
        landmarks : :class:`torch.Tensor`
            the landmarks to draw (of shape Num_Landmarks x 2)

        Returns
        -------
        :class:`torch.Tensor`
            the heatmap containing all landmarks 
            (of shape ``1 x self.img_shape[0] x self.img_shape[1]``)
        """

        landmarks = landmarks.view(-1, 2)

        #landmarks = landmarks.clone()

        for i in range(landmarks.size(-1)):
            landmarks[:, i] = torch.clamp(
                landmarks[:, i].clone(),
                self.half_size,
                self.img_shape[1 - i] - 1 - self.half_size)

        return torch.max(torch.cat([self.draw_lmk_helper(lmk.unsqueeze(0))
                                    for lmk in landmarks], dim=0), dim=0,
                         keepdim=True)[0]

    def forward(self, landmark_batch):
        """
        Draws all landmarks from one batch element in one heatmap

        Parameters
        ----------
        landmark_batch : :class:`torch.Tensor`
            the landmarks to draw 
            (of shape ``N x Num_landmarks x 2``))

        Returns
        -------
        :class:`torch.Tensor`
            a batch of heatmaps 
            (of shape ``N x 1 x self.img_shape[0] x self.img_shape[1]``)
            
        """

        return torch.cat([self.draw_landmarks(landmarks).unsqueeze(0)
                          for landmarks in landmark_batch], dim=0)
