from dan.layers import AffineLandmarkTransformation, EstimateAffineParams

import torch


def test_param_estimation():
    trafo_layer = AffineLandmarkTransformation()

    for num_lmks in [12, 33, 68, 1234]:

        ref_shape = torch.randint(0, 224, (10, num_lmks, 2)).float()

        estimator = EstimateAffineParams(ref_shape[0])

        for i in range(5):
            matrix = torch.rand(10, 6)

            src_shape = trafo_layer(ref_shape, matrix, inverse=True)
            assert src_shape.shape == ref_shape.shape
            result = estimator(src_shape)

            assert result.shape == (10, 6)
