import torch
from dan.layers import AffineImageTransformation


def test_affine_image():
    for img_shape in [(224, 224), (112, 112), (111, 111), (23, 23)]:
        trafo_layer = AffineImageTransformation(img_shape)
        input_image = torch.rand(10, 1, *img_shape)
        affine_trafo_matrix = torch.rand(10, 2, 3)
        affine_trafo_vector = affine_trafo_matrix.view(10, -1)

        result_matrix = trafo_layer(input_image, affine_trafo_matrix)
        result_vector = trafo_layer(input_image, affine_trafo_vector)

        assert result_matrix.shape == (10, 1, *img_shape)
        assert result_vector.shape == (10, 1, *img_shape)

        assert (result_matrix - result_vector).abs().sum() <= 1e-5
