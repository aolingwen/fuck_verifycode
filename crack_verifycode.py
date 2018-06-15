#encoding=utf8


import tkinter as tk
from tkinter import *
from verifyCodeNetwork import verify_code_network
import utility
import cv2


#窗体
class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self._text_answers = None
        self._master = master
        self._capchas_img = None
        self._create_widgets(master)

        #加载模型
        self._vcn = verify_code_network(is_training=False)


    #构建窗体
    def _create_widgets(self, master):
        master.title('神奇海螺')
        master.geometry("200x190")
        Button(master, text='玄学一把', command=self.gen_a_verifycode).place(x=55, y=10, width=90, height=40)
        Button(master, text='问问神奇海螺', command=self.predict_captchas).place(x=55, y=60, width=90, height=40)
        Label(master, text='验证码：').place(x=-40, y=120, width=200, height=40)
        self._text_answers = Text(master, height=10)
        self._text_answers.place(x=90, y=130, width=80, height=20)

    #识别验证码
    def predict_captchas(self):
        if self._capchas_img is not None:
            result = self._vcn.predict(self._capchas_img)
            self._text_answers.delete(1.0, END)
            self._text_answers.insert(1.0, ''.join(result))

    #打开文件对话框并图像图像
    def gen_a_verifycode(self):
        img, label = utility.gen_a_verifycode()
        self._capchas_img = img
        to_show = img.copy()
        to_show = cv2.resize(to_show, (300, 100))
        cv2.imshow(label, to_show)




if __name__=='__main__':
    root = Tk()
    app = Application(master=root)
    app.mainloop()
