![Visitors](https://visitor-badge.glitch.me/badge?page_id=tobran/GALIP) 
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/tobran/GALIP/blob/master/LICENSE.md)
![Python 3.9](https://img.shields.io/badge/python-3.8-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.9.0-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/tobran/GALIP)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/tobran/GALIP/graphs/commit-activity))
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
# GALIP

Official Pytorch implementation for our paper GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis


<p align="center">
<b>Generated Images
</p>

![teaser](results.jpg)
---
## Requirements
- python 3.9
- Pytorch 1.9
- At least 1x24GB 3090 GPU (for training)
- Only CPU (for sampling) (GALIP is a small and fast generative model which can generate multiple pictures in one second even on the CPU.)
## Installation

Clone this repo.
```
git clone https://github.com/tobran/GALIP
pip install -r requirements.txt
cd GALIP/code/
```



