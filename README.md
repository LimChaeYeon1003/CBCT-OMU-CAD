# CBCT-Denoising-CAD
We release OMU-Denoising-CAD train/test code.

Contributors: Chae-yeon Lim*, Kyungsu Kim*

Detailed instructions for train/testing the image are as follows.

---

# Implementation
OMU-Denoising-CAD implementation based on pytorch

Internal CBCT dataset : https://drive.google.com/file/d/1oW35VoVKf7W1NBstty1rghGxHWXEgSg2/view?usp=share_link

External CBCT dataset : https://drive.google.com/file/d/1sxct8LRip1mV6k7Lm6hynQcFx4ZDmujN/view?usp=share_link

---

# Environments

Required libraries for training/inference are detailed in requirements.txt
```python
pip install -r requirements.txt
```

---

# Pre-trained weights

Denoising weight: https://drive.google.com/file/d/16xECqeDaei64KiWiIBoaJmjteeZUULak/view?usp=share_link

ResNet weight: https://drive.google.com/file/d/1rbhoJPHuITF8dbW2_t8jCbDvb3SOzwER/view?usp=share_link

```
Classification_OMU
   |-- 1_Codes_pre
   |-- 2_Codes_3D
   |   |-- models
   |   |-- warmup_scheduler
   |   |   |-- r3d18_K_200ep.pth
   |   |   |-- r3d34_K_200ep.pth
   |   |   |-- r3d50_K_200ep.pth
   |-- 3_Codes_external
   |   |-- models
   |   |   |-- r3d18_K_200ep.pth
   |   |   |-- r3d34_K_200ep.pth
   |   |   |-- r3d50_K_200ep.pth
   |-- input
   |   |-- train_jpg
   |   |   |-- success_cbct_90
   |-- input_external
   |   |-- external_jpg_cbct_90
   |-- ISTA-U-Net-main
   |   |-- output_dir
   |   |   |-- ct
   |   |   |   |-- a3d5a1af-e159-43e4-ab89-473fc6af0ff5
   |   |   |   |   |-- config_dict.pickle
   |   |   |   |   |-- ista_unet.pt
   |-- OMU_inference.sh
   |-- OMU_train.sh
   |-- osj_env.yaml
   ```
---

# Train

The training denoising image is set automatically, and each 5 fold image is created at the same time as OMU_train.sh is executed, and key slice-selector and classification 5-fold training are automatically performed.
```
bash OMU_train.sh
```
The internal dataset training results are stored in /Classfication_OMU/2_Codes_3D/eval/

---

# Inference

Inference is started through OMU_inference.sh, and the inference shows GRAD-CAM and evaluation results.
```
bash OMU_inference.sh
```
The external dataset inference results are stored in /Classfication_OMU/3_Codes_external/eval/

The external dataset GRAD-CAM results are stored in /Classfication_OMU/3_Codes_external/cam/

---

# Result

GRAD-CAM result

![GRAD_CAM-1](https://user-images.githubusercontent.com/86760506/200550101-99e5e887-ae94-477f-9998-1d1c8e7d3a2d.png)

AUC-ROC result

![ROC_AUC-1](https://user-images.githubusercontent.com/86760506/200550242-26ff3b9e-6bd3-4fc1-8ea8-c9f50bf293ca.png)

---

# Acknowledgements

thanks to liutianlin0121 for sharing the denoising MUSC code.

MUSC github: https://github.com/liutianlin0121/musc
