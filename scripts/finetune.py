from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import torch
from detectron2.config.config import CfgNode
import logging
from dotenv import load_dotenv

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

#Check if MPS is available
mps_available = torch.backends.mps.is_available()

if cuda_available:
    device = 'cuda'
elif mps_available:
    device = 'mps'
else:
    #If none set device as CPU
    device = 'cpu'

class Finetune:
    def __init__(self):
        """
        Initializes the Finetune class.
        """
        self._cfg = None
        self._trainer = None

    def set_config_train(self, model_path: str, train_metadata=None, train_dataset_dicts=None, test_metadata=None, test_dataset_dicts=None):
        """
        Sets the configuration for training and creates a DefaultTrainer instance.

        Args:
            model_path (str): Path to the directory where the model will be saved.
            train_metadata (dict, optional): Metadata for the training dataset.
            train_dataset_dicts (list, optional): List of dictionaries containing training data.
            test_metadata (dict, optional): Metadata for the test dataset.
            test_dataset_dicts (list, optional): List of dictionaries containing test data.
        """
        self._cfg = get_cfg()
        self._cfg.OUTPUT_DIR = model_path
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.DATASETS.TRAIN = ("train_data",)
        self._cfg.DATASETS.TEST = ("test_data",)
        self._cfg.DATALOADER.NUM_WORKERS = 2
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self._cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
        self._cfg.SOLVER.BASE_LR = 0.0001  # Pick a good LR
        self._cfg.SOLVER.MAX_ITER = 5000  # 2000 iterations seems good enough for this dataset

        # Add this line to specify the steps at which to decrease the learning rate
        self._cfg.SOLVER.STEPS = (2000, 3000, 4000)  # Decrease the learning rate at 4000 and 4500 iterations
        self._cfg.SOLVER.GAMMA = 0.1  # Set the factor by which to decrease the learning rate

        # Add this line to specify the learning rate scheduler
        self._cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # We have 2 classes.

        # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.
        self._cfg.MODEL.DEVICE = device  # Setting device as MPS or CPU if CUDA is not enabled

        os.makedirs(self._cfg.OUTPUT_DIR, exist_ok=True)

        self._trainer = DefaultTrainer(self._cfg)  # Create an instance of DefaultTrainer with the given configuration
        self._trainer.resume_or_load(resume=False)  # Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available

    def save_cfg(self, cfg_path: str):
        """
        Saves the configuration to a config.yaml file.

        Args:
            cfg_path (str): Path to the configuration file.
        """
        # Save the configuration to a config.yaml file
        with open(cfg_path, "w") as f:
            f.write(self._cfg.dump())

    def train(self):
        """
        Starts the training process.
        """
        self._trainer.train()

def get_base_dir():
    """
    Returns the base directory of the project by going up two levels from the current script directory.
    """
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    return base_dir

def main():
    # Load environment variables from the .env file
    load_dotenv()

    # Get the base directory
    base_dir = get_base_dir()

    # Construct the full paths using the placeholders and the base directory
    model_path = os.path.join(base_dir, os.getenv("MODELS_DIR"))

    # Create an instance of the Finetune class
    finetune = Finetune()

    # Set the configuration and create the DefaultTrainer
    finetune.set_config_train(model_path)

    # Save the configuration to a config.yaml file
    config_path = os.path.join(model_path, "config.yaml")
    finetune.save_cfg(config_path)

    # Start the training process
    finetune.train()

if __name__ == '__main__':
    main()