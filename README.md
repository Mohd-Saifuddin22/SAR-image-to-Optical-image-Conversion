# SAR-to-Optical Image Translation using Supervised CycleGAN

This repository contains a TensorFlow 2.x implementation of a Cycle-Consistent Generative Adversarial Network (CycleGAN) adapted for **supervised** image-to-image translation, specifically targeting the conversion of Synthetic Aperture Radar (SAR) images into visually similar Optical images.

The core idea leverages the CycleGAN framework for its ability to handle unpaired data but enhances it by incorporating a supervised Mean Squared Error (MSE) loss, assuming **paired** SAR and Optical images are available for training. This supervision aims to guide the translation more directly towards the desired ground truth appearance.

## Overview

SAR images provide valuable information regardless of weather or illumination conditions, but their interpretation can be challenging compared to standard optical images. This project aims to bridge this gap by training a deep learning model to translate SAR images into a more intuitive optical representation.

The model uses:
* A **CycleGAN** architecture with two generators and two discriminators.
* **U-Net** based generators for capturing multi-scale features and details via skip connections.
* **PatchGAN** discriminators for effective patch-level realism feedback.
* A combination of **adversarial, cycle-consistency, identity, and supervised MSE losses** to train the networks.

## Architecture

* **Generator (`Generator()`):** A U-Net architecture with downsampling (convolutional) blocks and upsampling (transposed convolutional) blocks connected by skip connections. Takes a 256x256x3 image and outputs a 256x256x3 translated image.
* **Discriminator (`Discriminator()`):** A PatchGAN architecture that takes the source image and a target (real or generated) image concatenated together. It outputs a feature map representing the likelihood of patches in the target image being real.

Two instances of the Generator (`generator_g`, `generator_f`) and Discriminator (`discriminator_x`, `discriminator_y`) are created for the two translation directions (SAR <-> Optical). Based on the training logic:
* `generator_f`: Performs SAR -> Optical translation.
* `generator_g`: Performs Optical -> SAR translation.
* `discriminator_x`: Distinguishes real vs. fake Optical images.
* `discriminator_y`: Distinguishes real vs. fake SAR images.

## Loss Functions

The training process minimizes a combined loss for the generators:
1.  **Adversarial Loss:** Encourages generators to produce images that the discriminators classify as real (Binary Cross-Entropy).
2.  **Cycle Consistency Loss:** Ensures that translating an image to the other domain and back results in an image similar to the original (L1 norm, weighted by `lambda_cycle`).
3.  **Identity Loss:** Encourages generators to minimally alter images that are already in their target domain (L1 norm, weighted by `lambda_identity`).
4.  **Supervised MSE Loss:** Directly minimizes the Mean Squared Error between the generator's output and the corresponding paired ground truth image (weighted by `lambda_mse`). This is the key supervised component.

Discriminators are trained using a standard adversarial loss (Binary Cross-Entropy).

## Dataset

* **Requirement:** This model requires **paired** datasets. Each SAR image must have a corresponding Optical image capturing the same scene/area.
* **Format:** Images are expected to be in PNG format (modify `load_image` function in the script if using other formats). The code assumes 3-channel images for both SAR and Optical. If your SAR data is single-channel, you may need to modify the data loading or network input layers.
* **Structure:**
    ```
    <your_dataset_folder>/
    ├── optical/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── sar/
        ├── 1.png
        ├── 2.png
        └── ...
    ```
* **Preparation:** The script expects the `<your_dataset_folder>` to be zipped into a single file (e.g., `DATA SET.zip`) and placed in Google Drive for access via Google Colab.

## Requirements

* Python 3.x
* TensorFlow >= 2.x
* Matplotlib
* NumPy
* (Optional but Highly Recommended) NVIDIA GPU + CUDA + cuDNN for reasonable training times.
* (If using Google Colab) A Google Account and sufficient Google Drive storage.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow matplotlib numpy
    # Or install tensorflow-gpu if you have a compatible GPU setup
    # pip install tensorflow-gpu matplotlib numpy
    ```
3.  **Prepare the Dataset:**
    * Organize your paired SAR and Optical images into `sar` and `optical` subdirectories as described above. Ensure corresponding files have identical names.
    * Create a zip archive of the parent directory containing `sar` and `optical`. Name it `DATA SET.zip` (or update the `zip_path` variable in the script).
    * Upload `DATA SET.zip` to your Google Drive.
4.  **Configure Paths in the Script:**
    * Open the Python script (`SAR_to_Optical_Conversion_using_supervied_Cycle_GAN.py` or the `.ipynb` file).
    * Modify the following path variables near the top and in the **CHECK POINT** and **dataset** sections to match your Google Drive structure:
        * `zip_path = '/content/drive/MyDrive/DATA SET.zip'`
        * `extract_path = '/content/data1'` (local extraction path in Colab)
        * `checkpoint_dir = '/content/drive/MyDrive/checkpoint'`

## Usage

1.  **Environment:** Ensure your Python environment has the required libraries installed. If using Google Colab, upload the script/notebook.
2.  **Mount Drive (Colab):** If using Colab, make sure the Google Drive mounting cell (`drive.mount('/content/drive')`) is executed successfully.
3.  **Run the Script:** Execute the cells in the notebook sequentially, or run the Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
4.  **Training:**
    * The script will extract the dataset (if not already extracted).
    * It will check for existing checkpoints in `checkpoint_dir` and resume training if found.
    * Training progress, including epoch number, timing estimates, and sample image outputs (showing Optical input -> Generated SAR -> Real SAR), will be printed to the console/output.
    * Checkpoints are saved periodically to the specified `checkpoint_dir`.

## Notes and Potential Improvements

* **Normalization:** The current code loads images and converts them to `float32` (likely `[0, 1]` range), but the generator uses `tanh` activation (output `[-1, 1]`). Consider normalizing input images to `[-1, 1]` before feeding them to the network for potentially better stability. The visualization function `generate_images` already assumes `[-1, 1]` output.
* **Discriminator Inputs:** The way real image pairs are fed to the discriminators in the `train_step` function (`discriminator_x([optical_image, sar_image], ...)` and `discriminator_y([sar_image, optical_image], ...)`) might differ from standard implementations. Verify if this configuration aligns with the intended logic or if the input order needs adjustment (e.g., `discriminator_x([sar_image, optical_image], ...)`).
* **Hyperparameters:** The performance is sensitive to hyperparameters like `BATCH_SIZE` (72 is likely too high for 256x256 images on most GPUs - try smaller values like 1, 4, or 8), learning rates (`2e-4`), and loss weights (`lambda_cycle=10`, `lambda_mse=100`, `lambda_identity=0.5`). These may need tuning for your specific dataset.
* **SAR Input Channels:** The code assumes 3-channel SAR inputs. Modify data loading (`load_image`) and potentially the first layer of `generator_f` if using single-channel SAR data.
* **Visualization:** The `generate_images` function currently shows the results of `generator_g` (Optical->SAR). Modify it to visualize `generator_f` (SAR->Optical) if that's the primary goal.
