<img width="1750" alt="PupilSense: Detection of Depressive Episodes Through Pupillary Response in the Wild" src="https://github.com/stevenshci/PupilSense/blob/main/static/header.png">

> This is the official codebase of the pupillometry system paper [PupilSense: Detection of Depressive Episodes Through Pupillary Response in the Wild](https://arxiv.org/)

# Introduction
**PupilSense** is a deep learning-based pupillometry system. It uses eye images collected from smartphones for research in the behavior modeling domain.

The accompanying data collection app will be published soon for the research community.

<img width="1750" alt="Pupil-to-Iris Ratio (PIR) Estimation Pipeline" src="https://github.com/stevenshci/PupilSense/blob/main/static/PupilSense.png">


## Installation

Follow these steps to set up the project:

    (1) Clone the repository: git clone https://github.com/stevenshci/PupilSense.git
    (2) Navigate to the project directory: cd PupilSense
    (3) Install the required packages: pip install -r requirements.txt

### Setup

After installing the required packages, run the setup.sh script to set up the project environment:
    
    source setup.sh

### Dataset

This project uses a custom dataset of eye images for training and evaluation. The dataset should be organized in the following structure:

    dataset/
    ├── train/
    │   │── image1.png
    │   │── image2.png
    │   └── ...
    |
    │── train_data.json
    │    
    └── test/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    |
    └── test_data.json
       
Note: The annotations(test_data.json, train_data.json) should be in COCO format, with the pupil and iris regions labeled as separate categories.

### Annotations

To annotate the dataset, use tools like MakeSense.ai, Roboflow, Labelbox, LabelImg, or VIA to label pupil and iris regions on the images. Export these annotations in the COCO format, which should include necessary details for images, annotations, and categories.

The COCO format is a standard for object detection/segmentation tasks and is widely supported by many libraries and tools in the computer vision community.

### Usage

To fine-tune the Detectron2 model on your dataset, run the following command:

    python scripts/finetune.py

To test your images on a batch of images on the trained model:

    python scripts/inference.py

## Results

Our fine-tuned Detectron2 model achieves accurate pupil and iris segmentation on eye images captured in naturalistic environments.
<p align="center"> <img src="https://github.com/stevenshci/PupilSense/blob/main/static/2022-07-30-11-43-43-746phoneUnlock_LEFT%202.png" alt="Segmented Eye Image 1" width="400"> <img src="static/2022-07-30-11-43-43-746phoneUnlock_LEFT(1).png" alt="Segmented Eye Image 2" width="400"> </p>

The model robustly segments the pupil and iris regions in diverse real-world conditions, including varying lighting, eye positions, and backgrounds.

This capability enables practical applications in biometrics, human-computer interaction, and medical imaging, where precise segmentation in naturalistic settings is crucial.


## Contributing

If you'd like to contribute to this project, please follow these steps:

    (1) Fork the repository
    (2) Create a new branch: git checkout -b feature/new-feature
    (3) Make your changes and commit them: git commit -m 'Add new feature'
    (4) Push to the branch: git push origin feature/new-feature
    (5) Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or suggestions, please feel free to contact [Rahul](mailto:rahul.islam3@gmail.com) and [Priyanshu](mailto:Priyanshusbisen@outlook.com).



