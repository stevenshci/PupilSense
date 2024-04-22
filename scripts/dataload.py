from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import os

class EyeImageDataset:
    def __init__(self, image_path: str, train_metadata_path: str, test_metadata_path: str):
        """
        Initializes the EyeImageDataset.

        Args:
            image_path (str): Path to the Train Image Dataset.
            train_metadata_path (str): Path to the train metadata COCO file.
            test_metadata_path (str): Path to the test metadata COCO file.
        """
        self.image_path = image_path
        self.train_metadata_path = train_metadata_path
        self.test_metadata_path = test_metadata_path
        self.train_metadata = None
        self.train_dataset_dicts = None
        self.val_metadata = None
        self.val_dataset_dicts = None

    def register(self):
        """
        Registers a dataset in COCO's JSON annotation format for instance detection.

        Returns:
            tuple: A tuple containing the following elements:
                - train_metadata (dict, optional): Metadata for the training dataset.
                - train_dataset_dicts (list, optional): List of dictionaries containing training data.
                - val_metadata (dict, optional): Metadata for the validation dataset.
                - val_dataset_dicts (list, optional): List of dictionaries containing validation data.
        """
        register_coco_instances("train_data", {}, self.train_metadata_path, self.image_path)
        register_coco_instances("test_data", {}, self.test_metadata_path, self.image_path)

        self.train_metadata = MetadataCatalog.get("train_data")
        self.train_dataset_dicts = DatasetCatalog.get("train_data")
        self.val_metadata = MetadataCatalog.get("test_data")
        self.val_dataset_dicts = DatasetCatalog.get("test_data")

        return self.train_metadata, self.train_dataset_dicts, self.val_metadata, self.val_dataset_dicts

def get_base_dir():
    """
    Returns the base directory of the project by going up two levels from the current script directory.
    """
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    return base_dir


    