from tkinter import *
from PIL import Image, ImageTk, ImageDraw





def Quit(event):
    global root
    root.destroy()

def LoadFile(event):
    # text1.get('1.0', END)
    img = Image.Open(root, filetypes=[('*.txt files', '.txt')]).show()
    if fn == '':
        return
    textbox.delete('1.0', 'end')
    textbox.insert('1.0', open(fn, 'rt').read())


def SaveFile(event):
    fn = tkFileDialog.SaveAs(root, filetypes=[('*.txt files', '.txt')]).show()
    if fn == '':
        return
    if not fn.endswith(".txt"):
        fn += ".txt"
    open(fn, 'wt').write(textbox.get('1.0', 'end'))











root = Tk()

root.title("recognition")
root.geometry("600x450+300+300")

# frame0 = Frame(root, bg="white", bd=2)  # frame with load
# label0 = Label(frame0, "text" )
# LoadButton = Button(frame0, text="load")
# LoadButton.pack(side="left")
# LoadButton.bind("<Button-1>", LoadFile)
# frame0.pack()
# text1 = Text(frame0, height=7, width=7, font='Arial 14', wrap=WORD)
# text1.pack(side="left")
#
#
panelFrame = Frame(root, height=40, bg='gray')
panelFrame.pack(side='top', fill='x')

textbox = Text(panelFrame, font='Arial 14', wrap='word')
textbox.place(x=52, y=9, width=200, height=20)

loadBtn = Button(panelFrame, text='Load')
loadBtn.bind("<Button-1>", LoadFile)
loadBtn.place(x=10, y=10, width=40, height=20)


#get image
image = Image.open('imgs/text.jpg')  # Открываем изображение
img = ImageTk.PhotoImage(image)

label1 = Label(root, image=img)
label1.image = img
label1.place(x=20, y=150)

txt = "processing"
label2 = Label(root, text=txt)
label2.text = txt
label2.place(x=400, y=150)

# quitBtn = Button(panelFrame, text='Quit')
# quitBtn.bind("<Button-1>", Quit)
# quitBtn.place(x=850, y=10, width=40, height=40)

root.mainloop()
