

'''
    Author : Ziang Xu
    Student number : 180104048
    Code : Bulid GUI page. Include title page, image selection page, classification page
           and segmentation page. 

'''

from tkinter import Tk, Frame, Label, Button, Canvas, filedialog
from PIL import ImageTk, Image
from predict import Vgg16Predictor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class App(Tk):
    def __init__(self, predictor):
        Tk.__init__(self)
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # global variables for different pages.
        self.predictor = predictor
        self.im = None
        self.label = None
        self.heatmap = None
        self.segmentation= None

        # Set pages.
        # Set start page.
        self.frames = {}
        for F in (TitlePage,ImagePage, ResultPage,SegmentPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("TitlePage") 

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        print("Showing {}".format(page_name))
        frame.render()
        frame.tkraise()
# Set title page.
class TitlePage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        # Set title.
        lbl = Label(self, text='The gastrointestinal (GI) tract disease classification and segmentation by machine learning',font=('Arial', 15),
                    width=100, height=8,wraplength=400)
        lbl.pack(side="top", padx=10)
        # Set button.
        startButton = Button(self, text='Start', command=self.start)
        startButton.pack(side="top", padx=20)
    # Set page jump function.
    def start(self):
        frame = self.controller.show_frame("ImagePage")

    def render(self):
        pass
# Set image seletion page.
class ImagePage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        lbl1 = Label(self, text='The gastrointestinal (GI) tract disease classification and segmentation by machine learning',font=('Arial', 15),
                      width=100, height=8,wraplength=400)
        lbl1.pack(side="top", padx=10)
        lbl = Label(self, text='Choose an image:',font=('Arial', 12))
        lbl.pack(side="top", padx=10)
        self.controller.title('Test')
        self.controller.geometry('600x600')
        chooseButton = Button(self, text='Choose', command=self.choose, fg='RED')
        chooseButton.pack(side="top", padx=10)
        predictButton = Button(self, text='Predict', command=self.predict, fg='RED')
        predictButton.pack(side="top", padx=10)
        self.imageCanvas = Canvas(self, width = 300, height = 300)
        self.imageCanvas.pack()
    # Set the function of selecting image from computer.
    def choose(self):
        self.path = filedialog.askopenfilename(initialdir="C:/Users/jy/Desktop",filetypes=[("Image File",'.jpg')])
        print("You selelct {}".format(self.path))
        self.controller.im = Image.open(self.path)
        self.render()

    def render(self):
        if self.controller.im: 
            self.resized_im = self.controller.im.resize((300,300),Image.ANTIALIAS)
            self.imageCanvas.image = ImageTk.PhotoImage(self.resized_im)
            self.imageCanvas.create_image(0, 0, image=self.imageCanvas.image, anchor='nw')
    # Connect predict page.
    def predict(self):
        self.controller.label = self.controller.predictor.predict(self.resized_im)
        self.controller.heatmap = self.controller.predictor.heatmap(self.path)
        self.controller.segmentation = self.controller.predictor.segmentation(self.path)
        frame = self.controller.show_frame("ResultPage")

# Set classification page.
class ResultPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        self.controller.title('Test')
        self.controller.geometry('600x600')
        chooseButton = Button(self, text='Back',command=lambda: self.controller.show_frame("ImagePage"), fg='RED',font=('Arial', 12))
        chooseButton.pack(side="top", fill="x", pady=10)
        segmentButton = Button(self, text='Segment', command=self.segment)
        segmentButton.pack(side="top", fill="x", pady=10)
        self.predictedLabel = Label(self,font=('Arial', 12),text='Predicted label is')
        self.predictedLabel.pack(side="top", padx=20)
        
        
    #  Set the artboard to put in the Grad-CAM. 
    def render(self):
        print("Drawing figures with label {}".format(self.controller.label))
        self.controller.geometry('600x600')
        self.predictedLabel.configure(text='Predicted label is {}'.format(self.controller.label))
        self.attention1=Image.open('image/attention.jpg')
        self.attention=ImageTk.PhotoImage(self.attention1)
        self.heatmapLabel=Label(self,image=self.attention)
        self.heatmapLabel.pack()

    def segment(self):
        frame = self.controller.show_frame("SegmentPage")  

# Set segmentation page.
class SegmentPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        backButton = Button(self, text='Back', command=lambda: self.controller.show_frame("ResultPage"))
        backButton.pack(side="top", fill="x", pady=10)
        self.imageCanvas = Canvas(self, width = 300, height = 300)
        self.imageCanvas.pack()

        self.imagePath = "image/mask.jpg"

    def render(self):
        self.controller.im = Image.open(self.imagePath)
        resized_im = self.controller.im.resize((300, 300), Image.ANTIALIAS)
        self.imageCanvas.image = ImageTk.PhotoImage(resized_im)
        self.imageCanvas.create_image(0, 0, image=self.imageCanvas.image, anchor='nw')


if __name__ == "__main__":
    vgg16 = Vgg16Predictor('model/model1.h5')
    app = App(vgg16)
    app.mainloop()
