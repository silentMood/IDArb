# IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations

<div align="center">

 <a href='https://lizb6626.github.io/IDArb/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/lizb6626/IDArb'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> &nbsp;
 <!-- <a href='https://lizb6626.github.io/IDArb/'><img src='https://img.shields.io/badge/arXiv-2412.07759-b31b1b.svg'></a> &nbsp; -->

**[Zhibing Li<sup>1</sup>](https://lizb6626.github.io/), 
[Tong Wu<sup>1 &dagger;</sup>](https://wutong16.github.io/), 
[Jing Tan<sup>1</sup>](https://sparkstj.github.io/), 
[Mengchen Zhang<sup>2,3</sup>](https://kszpxxzmc.github.io/), 
[Jiaqi Wang<sup>3</sup>](https://myownskyw7.github.io/), 
[Dahua Lin<sup>1,3 &dagger;</sup>](http://dahua.site/)** 
<br>
<sup>1</sup>The Chinese University of Hong Kong
<sup>2</sup>Zhejiang University
<sup>3</sup>Shanghai AI Laboratory
<br>
&dagger;: Corresponding Authors

</div>

- [x] Release inference code and pretrained checkpoints.
- [ ] Release training code.

## Install

Our environment has been tested on CUDA 11.8 with A100.

```
git clone git@github.com:Lizb6626/IDArb.git && cd IDArb
conda create -n idarb python==3.8 -y
conda activate idarb
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Inference

### Single Image Intrinsic Decomposition
```
python main.py --data_dir example/single --output_dir output/single --input_type single
```

### Multi-view Intrinsic Decomposition
For multi-view intrinsic decomposition, camera pose can be incorporated by enabling the `--cam` option.
```
## --num_views: number of input views

# Without camera pose information
python main.py --data_dir example/multi --output_dir output/multi --input_type multi --num_views 4

# With camera pose information
python main.py --data_dir example/multi --output_dir output/multi --input_type multi --num_views 4 --cam
```

## Acknowledgement

This project relies on many amazing repositories. Thanks to the authors for sharing their code and data.

- [Wonder3D](https://github.com/xxlong0/Wonder3D)
- [GeoWizard](https://github.com/fuxiao0719/GeoWizard)
- [EscherNet](https://github.com/kxhit/EscherNet)
- [Stanford-ORB](https://github.com/StanfordORB/Stanford-ORB)
- [NeRD](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition/tree/master)

<!-- ## BibTex -->
