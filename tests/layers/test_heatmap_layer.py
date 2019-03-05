from dan.layers import HeatMap
import torch


def test_heatmap():

    for num_lmks in [10, 13, 56]:

        for img_size in [20, 112, 224]:
            lmks = torch.randint(0, img_size - 1, (3, num_lmks, 2)).float()

            for patch_size_div in [6, 7, 8, 9, 10]:
                patch_size = img_size // patch_size_div

                hm_layer = HeatMap((img_size, img_size), patch_size)

                assert hm_layer(lmks).shape == (3, 1, img_size, img_size)
