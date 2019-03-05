
Deep Alignement Network with `delira <https://github.com/justusschock/delira>`__
================================================================================

The following example shows the basic usage of the provided DAN
implementation with ``delira``

First we need to download our data. For training, we use the HELEN
Dataset, which can be downloaded
`here <https://ibug.doc.ic.ac.uk/download/annotations/helen.zip>`__.

   **Note**: Since this dataset contains only a small amount of data,
   you may want to download the other datasets on this website as well
   and add them to your trainset.

To automate the necessary preprocessing, please insert the path to the
downloaded zip-file below:

.. code:: ipython3

    zip_path = "/PATH/TO/YOUR/ZIPFILE"

First we need to preprocess our dataset, which is simply extracting it
and calculating the mean face. TO extract it, we use the libraries
``zipfile`` and ``os``:

.. code:: ipython3

    import os
    import zipfile
    
    # create directory "dataset" in same directory as zip file
    image_dir = os.path.join(os.path.split(zip_path)[0],"dataset")
    
    # if there is not an dataset already
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    
        with zipfile.ZipFile(zip_path) as zip_ref:
                zip_ref.extractall(image_dir)
            
    train_path = os.path.join(image_dir, "trainset")
    test_path = os.path.join(image_dir, "testset")

Now, we need to calculate the trainset’s mean shape. For this, we use
``numpy`` and ``shapedata``:

.. code:: ipython3

    import numpy as np
    import shapedata
    
    # if mean shape has not been computed yet:
    if not os.path.isfile(os.path.join(image_dir, "mean_shape.npz")):
    
        # Loading whole traindata
        data = shapedata.SingleShapeDataProcessing.from_dir(train_path)
    
        # store landmarks as numpy array and calculate mean
        landmarks = np.array(data.landmarks)
        mean_shape = landmarks.mean(axis=0)
    
        # save mean_shape to disk
        np.savez_compressed(os.path.join(image_dir, "mean_shape.npz"), mean_shape=mean_shape)
        
    else:
        mean_shape = np.load(os.path.join(image_dir, "mean_shape.npz"))["mean_shape"]


You just have to do these steps once, now you can simply load the
mean_shape with
``np.load(os.path.join(image_dir, "mean_shape.npz"))["mean_shape"]``

Now we will create our datasets (with classes from ``shapedata`` and
``delira``):

.. code:: ipython3

    from delira.data_loading import BaseDataManager
    from delira.data_loading.sampler import RandomSampler, SequentialSampler
    
    BATCH_SIZE = 4
    
    # some augmentations for train data:
    IMG_SIZE = 112
    CROP = 1.
    EXTENSION = ".pts"
    ROTATE = 90
    RANDOM_OFFSET = 50
    RANDOM_SCALE = 0.25
    
    # create trainset with augmentations
    dset_train = shapedata.SingleShapeDataset(train_path,
                                             img_size=IMG_SIZE,
                                             crop=CROP,
                                             extension=EXTENSION,
                                             rotate=ROTATE,
                                             random_offset=RANDOM_OFFSET,
                                             random_scale=RANDOM_SCALE
                                             )
    
    # create testset without augmentations
    dset_test = shapedata.SingleShapeDataset(test_path,
                                             img_size=IMG_SIZE,
                                             crop=CROP,
                                             extension=EXTENSION,
                                             rotate=None,
                                             random_offset=False,
                                             random_scale=False
                                             )
    
    # create data managers out of datasets
    man_train = BaseDataManager(dset_train, 
                                batch_size=BATCH_SIZE, 
                                n_process_augmentation=4, 
                                transforms=None, 
                                sampler_cls=RandomSampler)
    man_test = BaseDataManager(dset_test, 
                               batch_size=BATCH_SIZE, 
                               n_process_augmentation=4, 
                               transforms=None, 
                               sampler_cls=SequentialSampler)

Now, that we have defined our datasets for images of size 224x224
pixels, we need to take care of our model definition. Now we need to
define our training and model arguments using the ``Parameters`` class
from ``delira`` and some functions and classes given in this package
(here we import it for the first time):

.. code:: ipython3

    import dan
    
    from delira.training import Parameters
    import torch
    
    callback_stages = dan.AddDanStagesCallback(epoch_freq=50)
    
    params = Parameters(
        fixed_params={
            "training":{
                "num_epochs": 100,
                "criterions": {
                    "points": torch.nn.L1Loss()
                },
                "optimizer_cls": torch.optim.Adam,
                "optimizer_params":{
                    "max_stages": 2
                }, 
                "metrics": {"MSE": torch.nn.MSELoss()},
                "callbacks": [callback_stages],
                "lr_sched_cls": None,
                "lr_sched_params": {}
            }, 
            "model":
            {
                "mean_shape": mean_shape,
                "num_stages": 2,
                "return_intermediate_lmks": True
            }
        }
    )

Finally! Now, we can start our training using the ``PyTorchExperiment``.

We just do a few minor specifications here:

-  set the usable GPUs to the first available GPU if any GPUs have been
   detected (else specify the usable GPUs to be empty, which causes a
   training on CPU)
-  use the ``create_optimizers_dan_per_stage`` to automatically create
   optimizers for our DeepAlinmentNetwork (you could also use
   ``create_optimizers_dan_whole_network`` to create a single optimizer
   holding all network parameters)
-  use the ``DeepAlignmentNetwork`` class as our network, which defines
   the training and prediction behavior.

Now let’s start training!

.. code:: ipython3

    from delira.training import PyTorchExperiment
    
    if torch.cuda.is_available():
        gpu_ids = [0]
    else:
        gpu_ids = []
    
    exp = PyTorchExperiment(params, 
                            dan.DeepAlignmentNetwork, 
                            optim_builder=dan.create_optimizers_dan_per_stage, 
                            gpu_ids=gpu_ids,
                            val_score_key="val_MSE_final_stage")
    exp.run(man_train, man_test)
