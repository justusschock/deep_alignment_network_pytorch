import torch
import logging
from .stage import DeepAlignmentStage
from delira.models import AbstractPyTorchNetwork
import itertools


class DeepAlignmentNetwork(AbstractPyTorchNetwork):
    """
    The actual deep alignment model holding all stages and the logic inbetween

    References
    ----------
    `Original TensorFlow Implementation <https://github.com/MarekKowalski/DeepAlignmentNetwork>`_
    `Paper <https://arxiv.org/abs/1706.01789>`_

    """

    def __init__(self, mean_shape, input_size=112, num_stages=1, p_dropout=0.5,
                 patch_size=16, active_stages_begin=1,
                 return_intermediate_lmks=True, norm_type="instance"):
        """

        Parameters
        ----------
        mean_shape : :class:`numpy.ndarray` or :class:`torch.Tensor`
            the mean shape
        input_size : int or tuple, optional
            the size of the input images (the default is 112)
        num_stages : int, optional
            the number of DeepAlignmentStages (the default is 1)
        p_dropout : float, optional
            the dropout probability (the default is 0.5)
        patch_size : int, optional
            the patch size for heatmap generation (the default is 16)
        active_stages_begin : int, optional
            the active stages at the beginning (the default is 1)
        return_intermediate_lmks : bool, optional
            whether or not to return all landmarks from intermediate stages 
            concatenated into one 4D tensor 
            of shape N x NumStages x NumLandmarks x 2  (the default is True)
        norm_type : str, optional
            which kind of normalization to apply (the default is "instance")

        """

        super().__init__()

        assert num_stages >= 1, "Stages must be an integer >= 1"
        self.stages = torch.nn.ModuleList([DeepAlignmentStage(
            mean_shape, input_size=input_size, p_dropout=p_dropout,
            is_first=True, patch_size=patch_size,
            norm_type=norm_type)])

        for i in range(num_stages-1):
            self.stages.append(DeepAlignmentStage(mean_shape,
                                                  input_size=input_size,
                                                  p_dropout=p_dropout,
                                                  is_first=False,
                                                  patch_size=patch_size,
                                                  norm_type=norm_type))

        assert 1 <= active_stages_begin <= num_stages
        self.curr_active_stages = active_stages_begin
        self.return_intermediate_lmks = return_intermediate_lmks

    def forward(self, input_image):
        """
        Feeds an input image through aall stages

        Parameters
        ----------
        input_image : :class:`torch.Tensor`
            the input image

        Returns
        -------
        :class:`torch.Tensor`
            the returned landmarks of shape (N x NumStages x NumLandmarks x 2)
        """

        prev_lmk, prev_hidden = None, None

        if self.return_intermediate_lmks:
            intermediate_lmks = []

        for i in range(self.curr_active_stages):
            prev_lmk, prev_hidden = self.stages[i](
                input_image, prev_lmk, prev_hidden)

            if self.return_intermediate_lmks:
                intermediate_lmks.append(prev_lmk.unsqueeze(1))

        if self.return_intermediate_lmks:
            return torch.cat(intermediate_lmks, dim=1)
        else:
            return prev_lmk.unsqueeze(1)

    @staticmethod
    def prepare_batch(data_dict, input_device, output_device):
        """
        Pushes all batch entries to correct devices and converts to correct type

        Parameters
        ----------
        data_dict : dict
            the data dictionary
        input_device : :class:`torch.device` or str
            the device for all network inputs
        output_device : :class:`torch.device` or str
            the device for all network outputs and targets

        Returns
        -------
        dict
            dictionary with converted data
        """

        return {
            "data": torch.from_numpy(
                data_dict["data"]
            ).to(device=input_device, dtype=torch.float),
            "label": torch.from_numpy(
                data_dict["label"]
            ).to(device=output_device, dtype=torch.float)
        }

    @staticmethod
    def closure(model: AbstractPyTorchNetwork, data_dict: dict,
                optimizers: dict, criterions={}, metrics={},
                fold=0, **kwargs):
        """
        closure method to do a single backpropagation step
        Parameters
        ----------
        model : :class:`ClassificationNetworkBasePyTorch`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        criterions : dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments
            
        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict criterions)
        list
            Arbitrary number of predictions as torch.Tensor

        Raises
        ------
        AssertionError
            if optimizers or criterions are empty or the optimizers are not
            specified
        """

        assert (optimizers and criterions) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict.pop("data")
            preds = model(inputs)

            num_stages = preds.size(1)

            for i in range(num_stages):
                loss_val = criterions["points"](
                    preds[:, i], data_dict["label"])

                if optimizers:
                    with optimizers["%d_stage" % (i+1)].scale_loss(
                            loss_val) as scaled_loss:
                        optimizers["%d_stage" % (i+1)].zero_grad()
                        scaled_loss.backward(retain_graph=True)
                        optimizers["%d_stage" % (i+1)].step()

                loss_vals["point_error_%d_stage" % (i+1)] = loss_val.detach()

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        metric_vals[key + "_%d_stage" %
                                    (i+1)] = metric_fn(preds[:, i],
                                                       data_dict["label"])
                        if i == (num_stages - 1):
                            metric_vals[key + "_final_stage"] = \
                                metric_vals[key + "_%d_stage" % (i+1)]

        if not optimizers:
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})

        logging.info({'image_grid': {"images": inputs, "name": "input_images",
                                     "env_appendix": "_%02d" % fold}})

        return metric_vals, loss_vals, [preds]
