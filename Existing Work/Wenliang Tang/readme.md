# Text-based\OCR-based Image Captioning,面向场景文字的图像字幕生成

## 任务研究常用框架
* [MMF框架](https://mmf.sh/)，与其相关的模型为 [m4c](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c.py)和[m4c_captioner](https://github.com/facebookresearch/mmf/blob/main/mmf/models/m4c_captioner.py)

* 基于MMF框架，可以编写其它的图像字幕生成模型，例如在其中编写以LSTM架构为基础的代码，再将其import导入其它模型，从而实现MMA-SR，LSTM-R等新的模型
