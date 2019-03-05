from delira.training.callbacks import AbstractCallback
import itertools
import torch

from .model import DeepAlignmentNetwork


def create_optimizers_dan_per_stage(model: DeepAlignmentNetwork, optim_cls,
                                    max_stages, **optim_params):
    """
    Creates optimizers for differnt DAN stages

    Parameters
    ----------
    model : DeepAlignmentNetwork
        the model, whose parameters should be optimized
    optim_cls : 
        the actual optimizer class
    max_stages : int
        the number of maximum stages to optimize

    Returns
    -------
    dict
        dictionary containing all optimizers
    """

    optimizers = {}
    for i in range(min(max_stages, len(model.stages))):
        optimizers["%d_stage" % (i+1)] = optim_cls(
            model.stages[i].parameters(), **optim_params)

    return optimizers


def create_optimizers_dan_whole_network(model: DeepAlignmentNetwork, optim_cls,
                                        max_stages, **optim_params):
    """
    Creates one optimizer containing all stages' parameters

    Parameters
    ----------
    model: DeepAlignmentNetwork
        the model, whose parameters should be optimized
    optim_cls:
        the actual optimizer class
    max_stages: int
        the number of maximum stages to optimize

    Returns
    -------
    dict
        dictionary containing the optimizer optimizers
    """

    return {"1_stage": itertools.chain(model.stages[i].parameters()
                                       for i in range(min(max_stages,
                                                          len(model.stages))))}


class AddDanStagesCallback(AbstractCallback):
    """
    Callback to frequently activate new stages (if available)

    """

    def __init__(self, epoch_freq):
        """

        Parameters
        ----------
        epoch_freq : int
            number of epochs to wait, before activating the next stage

        """

        self.epoch_freq = epoch_freq

    def at_epoch_end(self, trainer, epoch, **kwargs):
        """
        Function which activates the next stage if the current epoch is 
        ``epoch_freq`` epochs after activating the last one

        Parameters
        ----------
        trainer : :class:`delira.training.PyTorchNetworkTrainer`
            the trainer holding the model
        epoch : int
            the current epoch

        Returns
        -------
        dict
            a dictionary with all updated values

        """

        if ((epoch % self.epoch_freq) == 0) and epoch > 0:
            if isinstance(trainer.module, torch.nn.DataParallel):
                module = "module.module"
            else:
                module = "module"

            num_stages = len(getattr(trainer, module).stages)
            curr_active_stages = getattr(trainer, module).curr_active_stages

            new_stages = min(num_stages, curr_active_stages + 1)

            setattr(getattr(trainer, module), "curr_active_stages", new_stages)

        return {"module": trainer.module}
