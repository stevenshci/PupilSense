import torch
from detectron2.config import get_cfg
import cv2
from detectron2.engine import DefaultPredictor
import os
from detectron2.utils.visualizer import Visualizer
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
import imghdr
from finetune import Finetune

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = 'cuda'
else:
    #If none set device as CPU
    device = 'cpu'

def get_base_dir():
    """
    Returns the base directory of the project by going up two levels from the current script directory.
    """
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    return base_dir

class Inference:
    def __init__(self, config_path: str, image_path: str, model_path: str):
        """
        Initializes the Inference class.

        Args:
            config_path (str): Path to the configuration file.
            image_path (str): Path to the directory containing the images.
        """
        self.image_dir = image_path
        self.predictor = self.get_predictor(config_path, model_path)
        self.config = None

    @staticmethod
    def get_predictor(cfg_path: str, model_path) -> DefaultPredictor:
        """
        Returns a DefaultPredictor instance based on the provided configuration file.

        Args:
            cfg_path (str): Path to the configuration file.

        Returns:
            DefaultPredictor: The DefaultPredictor instance.
        """
        # Fetch the config from the given path
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold

        return DefaultPredictor(cfg)

    def predict(self, image_path: str):
        """
        Performs inference on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: The output of the predictor.
        """
        image = cv2.imread(image_path)
        output = self.predictor(image)
        return output

    def infer_image_display(self, output, img_name: str):
        """
        Displays the predictions on a single image.

        Args:
            image_path (str): Path to the image file.
        """
        # Original directory
        original_dir = os.getcwd()

        # Image Path
        image_path = os.path.join(self.image_dir, img_name)

        # Reading the Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Define custom class names
        class_names = ["Iris", "Pupil"]

        v = Visualizer(image[:, :, ::-1], metadata = {"thing_classes": class_names}, scale=2.0)
        out = v.draw_instance_predictions(output["instances"].to(device))

        # Convert BGR to RGB
        out_rgb = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)

        # Creating a output directory to save predicted images
        output_dir = 'predicted'
        output_path = os.path.join(self.image_dir, output_dir)
        path = Path(output_path)
        if not path.exists():
            path.mkdir(parents=True)

        # Changing the directory to output directory to save images
        os.chdir(output_path)

        # Saving the image to output directory
        cv2.imwrite(img_name, out_rgb)

        # Changing back to original directory
        os.chdir(original_dir)
        

    def predict_batch(self, save_images=True):
        """
        Performs inference on a batch of images and calculates the pupil-to-iris ratio.

        Args:
            show_images (bool, optional): Whether to display the predictions on each image. Defaults to False.

        Returns:
            float: The average pupil-to-iris ratio.
        """
        info = {"imageName": [], "radiusIris": [], "radiusPupil": [], "xCenterIris": [], "yCenterIris": [], "xCenterPupil": [], "yCenterPupil": []}

        total_images = len(os.listdir(self.image_dir))
        valid_images = 0
        total_instances = 0
        iris_instances = 0
        pupil_instances = 0

        for image in os.listdir(self.image_dir):
            
            iris_flag = False
            pupil_flag = False
            image_name = os.path.join(self.image_dir, image)

            image_type = imghdr.what(image_name)
            if image_type is None:
                total_images -= 1
                print(f"{image} is not an image file. Skipping...")
                continue

            try:
                print(f'Testing on image: {image}')
                img = cv2.imread(image_name)
                valid_images += 1
            except Exception as e:
                logging.error(f'Image Read Failed: {e}')
                continue

            output = self.predictor(img)

            if save_images:
                self.infer_image_display(output, image)

            
            print(f'Prediction done on Image: {image}')
            instances = output["instances"]

            bbox_l = instances.pred_boxes.tensor.cpu().numpy()
            total_instances += len(bbox_l)
            if len(bbox_l) != 0:
                classes = instances.pred_classes
                scores = instances.scores

                # Create a list of tuples with (index, score) for each instance
                instances_with_scores = [(i, score) for i, score in enumerate(scores)]

                # Sort the instances based on the scores
                instances_with_scores.sort(key=lambda x: x[1], reverse=True)
                for index, score in instances_with_scores:
                    if classes[index] == 1 and not iris_flag:
                        iris_flag = True
                        pupil_instances += 1
                        pupil = bbox_l[index]
                        pupil_info = get_center_and_radius(pupil)
                        info["radiusPupil"].append(pupil_info["radius"])
                        info["xCenterPupil"].append(int(pupil_info["xCenter"]))
                        info["yCenterPupil"].append(int(pupil_info["yCenter"]))
                    elif classes[index] == 0 and not pupil_flag:
                        pupil_flag = True
                        iris_instances += 1
                        iris = bbox_l[index]
                        iris_info = get_center_and_radius(iris)
                        info["radiusIris"].append(iris_info["radius"])
                        info["xCenterIris"].append(int(iris_info["xCenter"]))
                        info["yCenterIris"].append(int(iris_info["yCenter"]))
                        info["imageName"].append(image)

                    if iris_flag and pupil_flag:
                        
                        break

        df = pd.DataFrame(info)
        iris_radius = df["radiusIris"].mean()
        pupil_radius = df["radiusPupil"].mean()
        print(f"Total Images = {total_images}, Valid Images = {valid_images}, Total Instances = {total_instances}, Iris Instances = {iris_instances}, Pupil Instances = {pupil_instances}")

        try:
            ratio = pupil_radius / iris_radius
        except ZeroDivisionError:
            logging.error("ZeroDivisionError: Cannot divide by zero.")
            ratio = 0

        return ratio

def get_center_and_radius(bbox):
    """
    Calculates the center and radius of a bounding box.

    Args:
        bbox (numpy.ndarray): A bounding box represented as [x1, y1, x2, y2].

    Returns:
        dict: A dictionary containing the center (xCenter, yCenter) and radius of the bounding box.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    xCenter = (bbox[2] + bbox[0]) / 2
    yCenter = (bbox[3] - height / 2)
    radius = width / 2

    return {"xCenter": xCenter, "yCenter": yCenter, "radius": radius}

def main():
    # Load environment variables from the .env file
    load_dotenv()

    # Get the base directory
    base_dir = get_base_dir()

    # Construct the full paths using the placeholders and the base directory
    model_path = os.path.join(base_dir, os.getenv("MODELS_DIR"))

    # Construct the full paths using the placeholders and the base directory
    image_path = os.path.join(base_dir, os.getenv("IMAGE_PATH_PLACEHOLDER"))
    config_path = os.path.join(base_dir, os.getenv("CONFIG_PATH_PLACEHOLDER"))

    print('Inference Started')
    infer = Inference(config_path, image_path, model_path)
    ratio = infer.predict_batch()

    print(f"Pupil-to-Iris Ratio: {ratio}")
    return

if __name__ == '__main__':
    main()