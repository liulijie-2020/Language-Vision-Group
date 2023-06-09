# Text-based\OCR-based Image Captioning,面向场景文字的图像字幕生成

## 任务研究常用框架
* [MMF框架](https://mmf.sh/)，与其相关的模型为 [m4c](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c.py)和[m4c_captioner](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c_captioner.py)

* 基于MMF框架，可以编写其它的图像字幕生成模型，例如在其中编写以LSTM架构为基础的代码，再将其import导入其它模型，从而实现MMA-SR，LSTM-R等新的模型



## CLIP提取TextCaps数据集中图像的特征
* 根据Faster-RCNN或OCR检测工具获得的区域，使用CLIP提取每个区域内的图像特征，代码见 extract_features.py


## 对TextCaps数据集中的视觉物体的检测框和OCR文字检测框可视化


## 使用基于主要目标物体(master object)的方法增强模型效果，代码改动部分包括：
* 在/home2/tangwenliang/mmf/mmf/datasets/builders/textcaps/dataset.py中根据检测框筛选出主要目标物体，并以此得到邻接矩阵A形式的图
* 在/home2/tangwenliang/mmf/mmf/models/m4c.py中引入相应的图卷积神经网络GCN模块，根据矩阵A和特征X，使用GCN对其进行编码，得到具有增强关系的特征
* 这种思想属于减少数据中的冗余关系，已经有其它的相关工作，例如ssbaseline，SA-M4C等模型

