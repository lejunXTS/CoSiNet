CoSiNet is a single-object tracking algorithm proposed by Mr. Wenjun Zhou, which is an improved method based on our previous work, SiamDUL.

The code was implemented by Dr. Zhou (zhouwenjun@swpu.edu.cn) and Ms. Liu from the Image Processing and Parallel Computing Laboratory, School of Computer Science, Southwest Petroleum University.

If you need to use it to test your own algorithms, please feel free to download it.

If you intend to use it in your paper, please inform us in advance.

Thank you.

May 16, 2024

# CoSiNet
Implementation of "Dual-Branch Collaborative Siamese Network for Visual Tracking" on Pytorch. 

#  Environment setup
This code has been tested on Ubuntu 22.04, Python 3.8, Pytorch 1.10.0, CUDA 11.3. Please install related libraries before running this code:
```bash
pip install -r requirements.txt
```

# Test tracker
You can directly run the `test.py` file.


# Eval tracker
You can directly run the `eval.py` file.


# Training
Download the datasets：[GOT-10k](http://got-10k.aitestunion.com/downloads)

For detailed environmental configuration and cutting of datasets , please refer to [siamfc-pytorch
](https://github.com/huanglianghua/siamfc-pytorch) or [SiamTrackers](https://github.com/HonglinChu/SiamTrackers).

You can simply run the train.py file to initiate the process.

# Acknowledgement
The code is implemented based on [siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)

We would like to express our sincere thanks to the contributors.

