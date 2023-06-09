# Text-based\OCR-based Image Captioning,面向场景文字的图像字幕生成

## 任务研究常用框架
* [MMF框架](https://mmf.sh/)，与其相关的模型为 [m4c](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c.py)和[m4c_captioner](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c_captioner.py)

* 基于MMF框架，可以编写其它的图像字幕生成模型，例如在其中编写以LSTM架构为基础的代码，再将其import导入其它模型，从而实现MMA-SR，LSTM-R等新的模型



## CLIP提取TextCaps数据集中图像的特征
* 根据Faster-RCNN或OCR检测工具获得的区域，使用CLIP提取每个区域内的图像特征，代码见 extract_features.py


## 对TextCaps数据集中的视觉物体的检测框和OCR文字检测框可视化


## 使用基于主要目标物体(master object)的方法增强模型效果，代码改动部分包括：
* 在/home2/tangwenliang/mmf/mmf/datasets/builders/textcaps/dataset.py中根据检测框筛选出主要目标物体，并以此得到邻接矩阵A形式的图
* 在/home2/tangwenliang/mmf/mmf/models/lstm_baseline.py中引入相应的图卷积神经网络GCN模块[mr_gcn.py](https://github.com/liulijie-2020/Language-Vision-Group/blob/fdc771f52eec2d3127477727255cbde3eea7a37f/Existing%20Work/Wenliang%20Tang/mr_gcn.py#LL37C45-L37C45)，根据矩阵A和特征X，使用GCN对其进行编码，得到具有增强关系的特征
* 修改/home2/tangwenliang/mmf/mmf/models/m4c_captioner.py内的代码，修改为import导入lstm_baseline的模型。（为什么这么费劲还要改另一个文件，因为我没搞懂怎么直接创立一个能被mmf读取的model，只能先借用m4c-captioner的框架）
* 这种思想属于减少数据中的冗余关系，已经有其它的相关工作，例如ssbaseline，SA-M4C等模型

## 使用CLIP思想，计算模型的OCR特征中的视觉和文本两种模态的特征相似度
* 修改/home2/tangwenliang/mmf/mmf/models/lstm_baseline.py内的函数[_forward_ocr_encoding_](https://github.com/liulijie-2020/Language-Vision-Group/blob/d4f9ea0c3c910b4aa73e4671f74b1a027396f334/Existing%20Work/Wenliang%20Tang/lstm_baseline.py#LL425C19-L425C19) 等编码方式，将OCR特征分开编码为txt和vis
* 通过矩阵相乘的方式，计算两种特征相似度
* 使该相似度矩阵的分布接近单位矩阵
* 这个方法有很多漏洞和缺陷，具体表现为训练后的相似度矩阵S，并没有出现类似单位矩阵一样的在对角线位置上的分布
