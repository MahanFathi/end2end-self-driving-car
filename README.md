# End-to-end Self-driving Car (Behavioral Cloning)

### This repo contains:
* Implementation of [NVIDIA's DAVE-2](https://arxiv.org/pdf/1604.07316.pdf) neural network in pure PyTorch (PilotNet)
* Data augmentation in train-feed pipeline
* Online monitoring of the training process via Visdom
* [VisualBackProp](https://arxiv.org/pdf/1611.05418.pdf) if it's of any help to sanity-checking the network
* Downloads all the data you need for your first run 
* Easy-tweaking configurations
* Ready-to-use inference API

### Usage 
Simply run `python3 main.py` to train a PilotNet from scratch. This should only take a few hours on a descent GPU. 

#### Backstory
This is a work of 3, done in the full-span of the 7th Hackathon of [CafeBazaar](https://cafebazaar.ir) -- two days! Results surprized the shit out of us!
