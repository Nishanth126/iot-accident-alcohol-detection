from tkinter import *
import os
from PIL import Image,ImageTk

root=Tk()
root.geometry("1200x600")
root.title("DRIVER DROWSINESS DETECTION SYSTEM")
root.resizable(0,0)

img=Image.open("bag.jpeg")
img=img.resize((1200,600))

bgg=ImageTk.PhotoImage(img)

lbl=Label(root,image=bgg)
lbl.place(x=0,y=0)

def drow():
    os.system("python DROW.py")

def yawn():
    os.system("python YAWN.py")
def seat():
    os.system("python seat.py")
def obj():
    os.system("python bot.py")

title=Label(root,text="DROWSINESS AND DISTRACTION  \n DETECTION SYSTEM",bg="black",fg="white",font=("times",20,"bold italic"))
title.place(x=450,y=50)


btn=Button(root,text="DROWSINESS",bg="black",fg="white",font=("times",18,"bold italic"),command=drow,width=35)
btn.place(x=400,y=150)

btn=Button(root,text="YAWNING AND HEAD NODDING",bg="black",fg="white",font=("times",18,"bold italic"),command=yawn,width=35)
btn.place(x=400,y=250)

btn=Button(root,text="SEAT BELT",bg="black",fg="white",font=("times",18,"bold italic"),command=seat,width=35)
btn.place(x=400,y=350)

btn=Button(root,text="OBJECT DISTRACTION",bg="black",fg="white",font=("times",18,"bold italic"),command=obj,width=35)
btn.place(x=400,y=450)

root.mainloop()