# fuck_verifycode
用tensorflow搭建CNN识别验证码

## 用到的库
tensorflow
opencv
numpy
captcha


# 生成验证码
python utlilty.py

# 训练模型和验证模型准确率
python train.py

# GUI
python crack_verifycode.py

# 杂七杂八
tensorboard的日志在log目录，模型保存在model目录，模型训练90个epoch，验证集大概能到98%的准确率