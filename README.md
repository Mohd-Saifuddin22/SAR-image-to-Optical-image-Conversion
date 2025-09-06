Supervised CycleGAN for SAR-to-Optical Image ConversionThis repository contains an implementation of a supervised Cycle-Consistent Generative Adversarial Network (CycleGAN) for converting Synthetic Aperture Radar (SAR) images to Optical images. This approach is enhanced with a supervised MSE loss to leverage paired data for improved translation quality.Table of ContentsProject OverviewRepository StructureSetup and InstallationUsageDataset PreparationTrainingInferenceModel ArchitectureContributingLicenseProject OverviewSynthetic Aperture Radar (SAR) provides all-weather, day-and-night imaging capabilities, but the resulting images are often difficult for human interpretation compared to optical images. This project aims to translate SAR images into visually intuitive optical images.We use a CycleGAN architecture, which is effective for image-to-image translation tasks, especially with unpaired data. To further improve results, we introduce a supervised Mean Squared Error (MSE) loss component, making this a "supervised" CycleGAN that can take advantage of aligned SAR and Optical image pairs.Key Features:Cycle-Consistent GAN: Ensures that if an image is translated to the target domain and back, it should resemble the original image.Supervised MSE Loss: Enforces pixel-level similarity when paired data is available.U-Net Generator: A generator with skip connections to preserve low-level image details.PatchGAN Discriminator: A discriminator that classifies patches of an image as real or fake, promoting sharper outputs.Modular & Scalable Code: The codebase is organized into logical modules for easy extension and maintenance.Repository StructureThe repository is organized as follows:/SAR_to_Optical_CycleGAN
|-- .gitignore           # Specifies files to be ignored by Git
|-- README.md            # This documentation file
|-- requirements.txt     # Python dependencies
|-- config.py            # Hyperparameters and configuration settings
|-- data_loader.py       # Data loading and preprocessing pipeline
|-- models.py            # Generator and Discriminator model definitions
|-- losses.py            # Loss functions for the training process
|-- trainer.py           # The main training logic encapsulated in a Trainer class
|-- train.py             # Executable script to start model training
|-- inference.py         # Script to run predictions with a trained model
`-- utils.py             # Utility functions (e.g., image saving)
Setup and InstallationFollow these steps to set up the project environment.1. Clone the repository:git clone [https://github.com/your-username/SAR_to_Optical_CycleGAN.git](https://github.com/your-username/SAR_to_Optical_CycleGAN.git)
cd SAR_to_Optical_CycleGAN
2. Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install the required dependencies:pip install -r requirements.txt
UsageDataset PreparationOrganize your dataset into two separate folders: one for SAR images and one for Optical images./data
|-- sar/
|   |-- image_001.png
|   |-- image_002.png
|   `-- ...
`-- optical/
    |-- image_001.png
    |-- image_002.png
    `-- ...
Note: For the supervised loss to work correctly, the images in both directories should be paired and have matching filenames.Update the data paths and other parameters in the config.py file to match your project setup.# config.py
OPTICAL_DIR = 'path/to/your/data/optical'
SAR_DIR = 'path/to/your/data/sar'
CHECKPOINT_DIR = 'checkpoints'
# ... other parameters
TrainingTo start training the model, run the train.py script:python train.py
Checkpoints: The model weights will be saved periodically in the directory specified by CHECKPOINT_DIR in config.py.Logs: Training progress can be monitored using TensorBoard. The logs are saved in the logs/fit/ directory.tensorboard --logdir logs/fit
InferenceTo generate an optical image from a new SAR image using a trained model, use the inference.py script.Make sure the CHECKPOINT_DIR in config.py points to your saved model weights.Run the inference script with the path to your input SAR image and the desired output path.python inference.py --input_path /path/to/single/sar_image.png --output_path /path/to/generated_optical.png
Model ArchitectureGenerator: A U-Net based architecture is used as the generator. It consists of an encoder-decoder structure with skip connections between corresponding layers. This allows the model to pass low-level information directly across the network, resulting in better image quality.Discriminator: A PatchGAN discriminator is used, which evaluates N x N patches of the input image to determine if they are real or fake. This encourages the generator to produce high-frequency details and sharper images.ContributingContributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.Fork the repository.Create a new branch (git checkout -b feature/your-feature-name).Commit your changes (git commit -m 'Add some feature').Push to the branch (git push origin feature/your-feature-name).Open a Pull Request.LicenseThis project is licensed under the MIT License. See the LICENSE file for more details.
