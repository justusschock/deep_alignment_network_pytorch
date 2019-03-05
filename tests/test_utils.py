from dan.utils import AddDanStagesCallback, create_optimizers_dan_per_stage, \
    create_optimizers_dan_whole_network
import torch
from dan import DeepAlignmentNetwork


def test_callback():
    class DummyTrainer:
        def __init__(self, module):
            self.module = module

    cb = AddDanStagesCallback(epoch_freq=50)

    model = DeepAlignmentNetwork(torch.rand(1, 68, 2), num_stages=2)

    trainer = DummyTrainer(module=model)

    assert cb.at_epoch_end(trainer, 1)["module"].curr_active_stages == 1

    assert cb.at_epoch_end(trainer, 50)["module"].curr_active_stages == 2
    assert cb.at_epoch_end(trainer, 50)["module"].curr_active_stages == 2


def test_optim_fns():
    model = DeepAlignmentNetwork(torch.rand(1, 68, 2), num_stages=45)

    assert len(list(create_optimizers_dan_per_stage(
        model, torch.optim.Adam, 5).keys())) == 5

    assert len(list(create_optimizers_dan_per_stage(
        model, torch.optim.Adam, 45).keys())) == 45

    assert len(list(create_optimizers_dan_whole_network(
        model, torch.optim.Adam, 5).keys())) == 1
