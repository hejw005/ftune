## Introduction

Official implementation of "FA-GANs: Facial Attractiveness Enhancement with Generative Adversarial Networks on Frontal Faces".

This paper proposes FA-GANs for facial attractiveness enhancement, with a newly designed geometry enhancement consistency module to automatically enhance the input face in both geometry and appearance aspects.

The comparison with other state-of-the-arts methods are availbale at [Details Comparison](https://hejw005.github.io/ftune/comparison/compare.html)

## Requirements
- torch == 1.2.0
- torchvison == 0.4.0
- Pillow == 6.0.0
- matplotlib == 3.0.3
- python == 3.5
- Linux CUDA CuDNN

## Getting Started

To test the facial image, just fill the parameters in ftuneStart.py file

```
    src_folder = '../src_imgs'
    batch_size = 8
    dst_folder = '../outputs'
    model_file = '../model/UNet_simple_11-11-12-45.pkl'
```

and run 
```
python3 ftuneStart.py
```

