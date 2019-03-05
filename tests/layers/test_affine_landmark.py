from dan.layers import AffineLandmarkTransformation

import torch


def test_affine_landmark():
    trafo_layer = AffineLandmarkTransformation()
    affine_vector = torch.rand(10, 6)

    for num_lmks in [12, 33, 68, 1234]:
        lmks = torch.randint(0, 224, (10, num_lmks, 2)).float()

        result_vector = trafo_layer(lmks, affine_vector)
        assert result_vector.shape == lmks.shape
