# pytorch-capsnet
Reimplementation of the official code NIPS 2017 paper Dynamic Routing Between Capsules

I tried to follow the original implementation from https://github.com/naturomics/capsule-model-forked-from-Sarasra, which is forked from the official implementation. Since I am new to pytorch, any correction and improvement are welcome. Thank you. 

## Requirements

[PyTorch](http://pytorch.org/) with torchvision is required (0.3.1 tested). Scipy, numpy, PIL and tqdm are required for data processing and image reconstruction. 

## Usage

Train the model by running

    git clone https://github.com/wetliu/pytorch-capsnet/
    cd pytorch-capsnet
    python main.py
    
Optional arguments are all in the config.py file.
MNIST dataset will be downloaded automatically.

## Benchmarks
The test accuracy was 99.22% after 30 epochs.

The reconstructions of the digit numbers are showed at right and the ground truth at left.
<table>
  <tr>
    <td>
     <img src="results/gt.jpg"/>
    </td>
    <td>
     <img src="results/30.jpg"/>
    </td>
  </tr>
</table> 

## References
- [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
- [Sarasra/capsules](https://github.com/Sarasra/models/tree/master/research/capsules)
