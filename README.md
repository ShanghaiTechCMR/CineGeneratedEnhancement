# Cine Generated Enhancement

This is the official implementation for **"Predicting Late Gadolinium Enhancement of Acute Myocardial Infarction in Contrast-free Cardiac Cine MRI using Deep Generative Learning"**.

We provide:

- **models/**: Directory containing the model definitions used in our study.
- **data/**: Directory containing the pretrained weights and a sample batch from the ACDC dataset.
- **evaluate.ipynb**: A Jupyter notebook containing the inference code and visualization tools to generate CGE images from cine MRI images in the sample batch.
## Citation

You can cite our work using the following BibTeX entry:

```bibtex
@article{doi:10.1161/CIRCIMAGING.124.016786,
title = {Predicting Late Gadolinium Enhancement of Acute Myocardial Infarction in Contrast-Free Cardiac Cine MRI Using Deep Generative Learning},
author = {Haikun Qi and Pengfang Qian and Langlang Tang and Binghua Chen and Dongaolei An and Lian-Ming Wu},
journal = {Circulation: Cardiovascular Imaging},
volume = {TBD},
number = {TBD},
pages = {e016786},
year = {TBD},
doi = {10.1161/CIRCIMAGING.124.016786},
url = {https://www.ahajournals.org/doi/abs/10.1161/CIRCIMAGING.124.016786},
eprint = {https://www.ahajournals.org/doi/pdf/10.1161/CIRCIMAGING.124.016786}
}
```

## Usage

We adopted Python 3.9 as the development and evaluation environment. You can install the necessary packages according to the provided `environment.yaml` file using `conda`.

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ShanghaiTechCMR/CineGeneratedEnhancement.git
    cd CineGeneratedEnhancement
    ```

2. **Create a Conda Environment**

    Ensure you have Anaconda or Miniconda installed. Create a new conda environment using the `environment.yaml` file:

    ```bash
    conda env create -n py39cge --file environment.yaml
    ```

    Activate the environment:
    
    ```bash
    conda activate py39cge
    ```

3. **Running Inference**

    You can use `evaluate.ipynb` to perform inference with our pretrained weights on the sample batch. Launch Jupyter Notebook Server and open the notebook `evaluate.ipynb`. Run the cells sequentially to perform inference and visualize the generated CGE images from the cine MRI images in the sample batch.
