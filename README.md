# 🪱 DeepLombrics

This repository contains the code for the paper:  
**Si-moussi, Hedde et al.** _"Digging deeper: AI-powered joint species distribution modeling to predict earthworm preferences"_.

🧠 In this paper, we develop a deep latent variable joint species distribution model that leverages neural networks for environmental feature extraction and uses variational autoencoders for amortized inference of the latent factors.  
🌱 We call this model **MTEC** (Multi-task model of ecological communities). 

---

## 📁 Repository Structure

The code to run the model is in the folder **`MTEC/`** which includes:

- 🧩 `lego_blocks.py`: basic neural network components  
- 🧠 `mtlvae.py`: the deep variational JSDM model  
- 📊 `eval_functions.py`: loss functions and evaluation routines

🐛 In the above-mentioned paper, we apply **MTEC** to **Lombrics**, a dataset of Earthworm communities sampled across France in the 1970s by Marcel Bouché and described in his book: _"Lombriciens de France"_.  
📚 The dataset was digitized and curated by Mickael Hedde and Sylvain Gérard.

The folder **`Lombrics/`** contains the code for the analysis on the Earthworms dataset:

- 🛠 `prep_data.py` and `prep_eval_data.py`: prepare the training and calibration datasets  
- 🔄 `load_data.py`, `ew_mtl.py`: run MTEC on the Earthworm dataset  
- 💾 Includes the trained model checkpoint  

📦 We compared **MTEC** to machine learning-based SDMs implemented in the **BIOMOD2** package in R.  
📁 The code used to run BIOMOD2 models across species is in the **`BIOMOD2/`** folder.

🔍 To unravel Earthworm habitat preferences and identify key environmental drivers of species distributions, we used interpretable machine learning (aka explainable AI).  
🧰 These analyses were performed using the R package **`iml`**, and the code is located in **`Lombrics/iml/`**.

---

## 📦 Install Dependencies

```bash
conda create -n mtec
conda activate mtec
pip install -r requirements.txt
