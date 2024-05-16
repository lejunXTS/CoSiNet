Code will be made public here soon!

# CoSiNet
Implementation of " " on Pytorch. 

#  Environment setup
This code has been tested on Ubuntu 22.04, Python 3.8, Pytorch 1.10.0, CUDA 11.3. Please install related libraries before running this code:
```bash
pip install -r requirements.txt
```

# Compile
cd /path/to/yourproject

python setup.py build_ext --inplace

# Test tracker
You can either directly run the `my_test.py` file or use the following command to execute the script:


# Eval tracker
You can either directly run the `my_eval.py` file or use the following command to execute the script:


# Training
Download the datasets：[GOT-10k](http://got-10k.aitestunion.com/downloads)

For detailed environmental configuration and cutting of datasets , please refer to [siamfc-pytorch
](https://github.com/huanglianghua/siamfc-pytorch) or [SiamTrackers](https://github.com/HonglinChu/SiamTrackers).

You can simply run the my_train.py file to initiate the process.

# Acknowledgement
The code is implemented based on [siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)

We would like to express our sincere thanks to the contributors.

