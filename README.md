# DeepLombrics

This repository contains the code for the paper: Si-moussi, Hedde et al. "Digging deeper: AI-powered joint species distribution modeling to predict earthworm preferences".

In this paper, we develop a deep latent variable joint species distribution model that leverages neural networks for environmental feature extraction and uses variational autoencoders for amortized inference of the latent factors.
We call this model MTEC (multi-task model of ecological communities). 

The code to run the model is in the folder "MTEC" which includes:
- "lego_blocks.py": contains the basic neural network components.
- "mtlvae.py": contains the deep variational JSDM model.
- "eval_functions.py": contains loss functions and evaluation routines.

In the above-mentioned paper, we apply MTEC to "Lombrics", a dataset of Earthworm communities sampled across France in the 1970s by Marcel Bouché and described in his book: "Lombriciens de France". The dataset was digitized and curated by Mickael Hedde and Sylvain Gérard.
The folder "Lombrics" contains the code for the analysis on the earthworms dataset, it contains: 
- Code to prepare the training and calibration datasets: "prep_data.py" and "prep_eval_data.py" respectively.
- Code to run MTEC on the earthworm dataset in the "mtec" folder, including script to load the data "load_data.py", run the model "ew_mtl.py" and the trained checkpoint. 

We compared MTEC to machine learning based SDMs implemented in the BIOMOD2 package in R, the code used to run the BIOMOD2 model across species is provided in the BIOMOD2 folder.

To unravel Earthworm habitat preferences and indentify key environmental drivers of earthworm species distributions, we used interpretable machine learning also referred to as explainable AI techniques.
To do that, we used the R package "iml". The code for these analyses is provided in "Lombrics/iml/". 

# Install dependencies
conda create -n mtec
conda activate mtec
pip install -r requirements.txt

