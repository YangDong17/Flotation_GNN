# High-Throughput Discovery of Phosphorus-Enhanced Collectors for Malachite Flotation via a DFT–Graph Neural Network Workflow
Efficient mineral separation in flotation processes relies on the development of selective reagents, particularly collectors. As mineral resources become increasingly scarce, designing specialized collectors is paramount. In this study, we constructed the first comprehensive dataset of malachite collector adsorption energies and discovered that secondary bonding atoms—especially phosphorus—crucially enhance adsorption strength, explaining the superior performance of phosphate-based collectors over xanthate reagents. Leveraging Density Functional Theory (DFT) data, we evaluated multiple machine learning (ML) models and developed an graph neural network model, which achieved a mean absolute error of 0.107 eV in predicting adsorption energies. By coupling this predictive model with our dataset insights, we performed high-throughput screening of 368,374 molecules, successfully pinpointing three high-affinity collectors bearing phosphorus as the secondary atom. This integrated DFT–ML approach not only accelerates the discovery of effective collectors but also underscores the powerful role of ML in meeting the challenges of increasingly limited mineral resources.

# Installation
Create a conda environment with the required dependencies. This may take a few minutes.
```
conda env create -f env.yml
```
Activate the conda environment with:
```
conda activate mlcgmd
```
Then install graphwm (stands for graph world models) as a package:

```
pip install -e ./
```
# Prepare the dataset

Our DFT dataset is available thought https://figshare.com/s/407d85e9e0dab8ffdacf

# Acknowledgements
Flotation_GNN builds upon the source code and data from the following projects:

[OCP](https://github.com/facebookresearch/fairchem)

[KAN](https://github.com/Blealtan/efficient-kan)

We thank all their contributors and maintainers!
