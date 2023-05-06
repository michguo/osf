# Learning Object-Centric Neural Scattering Functions for Free-Viewpoint Relighting and Scene Composition

By Hong-Xing Yu*, Michelle Guo*, Alireza Fathi, Yen-Yu Chang, Eric Ryan Chan, Ruohan Gao, Thomas Funkhouser, Jiajun Wu

arXiv link: https://arxiv.org/abs/2303.06138

## Setup

```
git clone https://github.com/michguo/osf.git
cd osf
conda env create -f environment.yml
```

## Data and Models

You can download the data and pretrained models from the following Google Drive links:

- Data (2.0 GB): https://drive.google.com/file/d/1IWNt5R2Mpp1XHo8JiwfHUZXI7Coi4OXO/view?usp=share_link
- Models (0.8 GB): https://drive.google.com/file/d/1oj-F4SWgt-vfRyS9MHIVf9cdJnZe-p81/view?usp=share_link

## Training and Evaluation

Configuration files can be found in the `configs` folder.
To train an OSF, run the following command:

```
python run_osf.py --config ${CONFIG_PATH}
```

For testing, you can run

```
python run_osf.py --config ${CONFIG_PATH} --render_only --render_test
```

## Composing OSFs

After training individual OSFs, you can compose them together into arbitrary
scene arrangements at test time. This example composes the checkers background
and the ObjectFolder objects together into a scene:

```
python run_osf.py --config=configs/compose.txt
```

## Citation

```
@article{yu2023osf,
    title={Learning Object-centric Neural Scattering Functions for Free-viewpoint Relighting and Scene Composition},
    author={Yu, Hong-Xing and Guo, Michelle and Fathi, Alireza and Chang, Yen-Yu and Chan, Eric Ryan and Gao, Ruohan and Funkhouser, Thomas and Wu, Jiajun},
    journal={Transactions on Machine Learning Research},
    year={2023}
}
```

## Acknowledgements

Our code framework is adapted from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [KiloOSF](https://github.com/yuyuchang/KiloOSF).
