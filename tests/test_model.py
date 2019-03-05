from dan import DeepAlignmentNetwork
import torch
from delira.utils.context_managers import \
    DefaultOptimWrapperTorch as OptimWrapper


def test_dan():

    mean_shape = torch.rand(68, 2)
    model = DeepAlignmentNetwork(
        mean_shape, num_stages=3, active_stages_begin=3, input_size=64,
        return_intermediate_lmks=True)

    # check single forward
    preds = model(torch.rand(10, 1, 64, 64))
    assert preds.shape == (10, 3, 68, 2)

    # check closure without optimizers and criterions
    model.closure(model, {"data": torch.rand(10, 1, 64, 64),
                          "label": torch.rand(10, 68, 2)
                          },
                  optimizers={},
                  criterions={"points": torch.nn.L1Loss()})

    # check closure with optimizers and criterions
    model.closure(
        model,
        {"data": torch.rand(10, 1, 64, 64),
         "label": torch.rand(10, 68, 2)
         },
        optimizers={
            "1_stage": OptimWrapper(
                torch.optim.Adam(model.stages[0].parameters())),
            "2_stage": OptimWrapper(
                torch.optim.Adam(model.stages[1].parameters())),
            "3_stage": OptimWrapper(
                torch.optim.Adam(model.stages[2].parameters()))
        },
        criterions={"points": torch.nn.L1Loss()}
    )
