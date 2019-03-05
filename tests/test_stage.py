from dan import DeepAlignmentStage
import torch


def test_stage():
    for img_size in [112, 224]:
        for num_lmks in [68]:
            mean_shape = torch.randint(img_size-1, (1, num_lmks, 2)).float()
            first_stage = DeepAlignmentStage(
                mean_shape, input_size=(img_size, img_size))
            second_stage = DeepAlignmentStage(
                mean_shape, is_first=False, input_size=(img_size, img_size))

            input_image = torch.rand(10, 1, img_size, img_size)

            lmks, hidden = first_stage(input_image)

            assert lmks.shape[1:] == mean_shape.shape[1:]

            new_lmks, hidden = second_stage(input_image, lmks, hidden)

            assert new_lmks.shape[1:] == mean_shape.shape[1:]
