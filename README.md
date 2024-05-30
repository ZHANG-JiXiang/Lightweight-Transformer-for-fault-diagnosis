#### Description of datasets

1.The bevel gearbox fault datasets collect from HNU

Data file structure：

```
| Data/
|————work condition1.xx
|----work condition2.xx
|----.....
```

Experimental rig ：

![湖南大学实验装置](https://github.com/bighan123/HNUIDG-Fault-Diagnosis-/blob/main/png/湖大试验台.jpg)

2.The gearbox fault datasets collect from XJTU

Data file structure：

```python
| Data/
|----work condtion1
|    |---- Channel one.xx
|    |---- Channel two.xx
|    |---- ......
|----work condition2
|    |---- Channel one.xx
|    |---- Channel two.xx
|    |---- ......
|......
```

Experimental rig ：

![西安交通大学实验装置](https://github.com/bighan123/HNUIDG-Fault-Diagnosis-/blob/main/png/Figure_XJTUGearbox.jpg)

3.The gearbox fault datasets collect from DDS

Data file structure：

```python
| Data/
|---- work condtion1
|     |---- data.xx
|---- work condtion2
|     |---- data.xx
```

#### Installation Tutorial 

This repository is tested on Windows 10, Python 3.7, Pytorch 1.7.01 and CUDA 10.1

Installing dependency repository:

pip install -r requirement.txt

Download code locally with git：

git clone https://github.com/bighan123/HNUIDG-Fault-Diagnosis-.git
#### Citation

If you have used the the EHcnn model as a comparison model, please cite:

```python
@article{Han2022DL,
        title={Intelligent fault diagnosis of aero-engine high-speed bearing using enhanced convolutional neural network},
        author={Han SongYu and Shao Haidong and Jiang Hongkai and Zhang Xiaoyang},
        journal={航空学报},
        year={2022}}
```

If you have used the EHcnn_dilation model or enhanced cross entropy as a comparison model, please cite:

```
@article{Han2022DL,
        title={Novel multi-scale dilated CNN-LSTM for fault diagnosis of planetary gearbox with unbalanced samples under noisy environment},
        author={Han Songyu and Zhong Xiang and Shao Haidong and Xu Tianao and Zhao Rongding and Cheng Junsheng},
        journal={Measurement Science and Techonology},
        year={2021}}
```

If you have used the Convformer-nse model as a comparison model, please cite:

```
@article{Han2022DL,
        title={Convformer-NSE: A Novel End-to-End Gearbox Fault Diagnosis Framework Under Heavy Noise
Using Joint Global and Local Information},
        author={Han Songyu and Shao Haidong and Cheng Junsheng and Yang Xingkai and Cai Baoping},
        journal={IEEE/ASME Transactions on Mechatronics},
        year={2022}}
```

If you have used dynamic training (train_dynamic.py) as a comparison experiment, please cite:

```
@article{Han2022DL,
        title={End-to-end chiller fault diagnosis using fused attention mechanism and dynamic cross-entropy under imbalanced datasets},
        author={Han SongYu and Shao Haidong and Huo Zhiqiang and Yang Xingkai and Cheng Junsheng},
        journal={Building and Environment},
        year=2022}}
```

If you have used the publicly  dataset from XJTU , please cite:

```
[1] Tianfu Li, Zheng Zhou, Sinan Li, Chuang Sun, Ruqiang Yan, Xuefeng Chen, “The emerging graph 
neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study,”
*Mechanical Systems and Signal Processing*, vol. 168, pp. 108653, 2022. DOI:
10.1016/j.ymssp.2021.108653
```

[XJTU Gearbox Datasets](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay)

if you have used the code of our repository, please star it, thank you very much.

#### Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection,please contact us.

fletahsy@hnu.edu.cn

Mentor E-mail：hdshao@hnu.edu.cn

