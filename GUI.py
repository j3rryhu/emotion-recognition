from tkinter import *
import tkinter.messagebox
from video_detection import start_recognition


def instruction_book():
    tkinter.messagebox.showinfo('说明', '点击\'开始识别将自动开启摄像头进行实时识别\'\n点击\'退出\'将自动退出程序\'\n程序开始后按q键并敲击enter退出程序')


window = Tk()
window.title('智能表情识别工具')
frame1 = Frame(window)
frame1.pack()
frame2 = Frame(window)
frame2.pack()
frame3 = Frame(window)
frame3.pack()
window.geometry('500x300')
wel = Label(frame1, text='欢迎使用智能表情识别工具', font=('Arial', 12), width=30, height=2)
wel.pack()
b1 = Button(frame2, text='开始实时识别', command=lambda: start_recognition(), height=4, width=30)
b1.pack()
b2 = Button(frame2, text='退出', command=lambda: quit(), pady=10, width=30)
b2.pack()
b3 = Button(frame2, text='查看说明', command=lambda: instruction_book())
b3.pack()
l = Label(frame3, text='重要提示：使用前务必阅读使用说明')
l.pack()
mainloop()

