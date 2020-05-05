import pickle
import queue
import os
import tkinter as tk
from multiprocessing.queues import Queue
from tkinter import filedialog
from tkinter import messagebox as msg
from tkinter.ttk import Notebook
from threading import Thread

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from tensorflow_core.python.keras.callbacks import CSVLogger, Callback

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from constructor.API import NNConstructorAPI

import sys
class NNI(tk.Tk):

    listbox_items_builder = ['Default Enter layer']
    layerBuffer = [None]
    constructorAPI = NNConstructorAPI()

    def __init__(self):
        super().__init__()
        self.log_array = None
        self.list_folder = []
        self.x = []

    #Windows option
        self.title("NNI")
        self.geometry("1200x600")
        self.resizable(width=False, height=False)

    #Create tab
        self.notebook = Notebook(self)
        builder_tab = tk.Frame(self.notebook)
        self.result_tab = tk.Frame(self.notebook)
        self.storage_tab = tk.Frame(self.notebook)
        self.notebook.add(builder_tab, text="Building architecture")
        self.notebook.add(self.result_tab, text="Result", state="disabled")
        self.notebook.add(self.storage_tab, text="Storage")

    #Create canvas for dinamic blocs
        self.tasks_canvas = tk.Canvas(builder_tab, width=600, bd=0, highlightthickness=0, relief='ridge')
        self.storage_canvas = tk.Canvas(self.storage_tab, width=1200, height=600, bd=0,
                                        highlightthickness=0, relief='ridge')
        self.storage_frame = tk.Frame(self.storage_canvas)
        self.tasks_frame = tk.Frame(self.tasks_canvas, bd=0)
        self.text_frame = tk.Frame(builder_tab, bd=0)
        self.storaget_frame = tk.Frame(self.tasks_canvas, bd=0)
        self.scrollbar = tk.Scrollbar(self.tasks_canvas, orient="vertical", command=self.
                                      tasks_canvas.yview, bd=0)
        self.tasks_canvas.configure(yscrollcommand=self.scrollbar.set, bd=0)
        self.tasks_canvas.place(x=0, y=0)
        self.storage_canvas.place(x=0, y=0)
        self.scrollbar.place_forget()

    #Create lebels
        #visiable lebels
        self.lPath = tk.Label(builder_tab, text="Path:")
        self.lPath.place(x=620, y=15)
        self.lLayers = tk.Label(builder_tab, text="Layers",  font='Arial 10')
        self.lLayers.place(x=230, y=5)
        self.lLearning_options = tk.Label(builder_tab, text="Learning options:")
        self.lLearning_options.place(x=620, y=45)
        self.lBatch_size = tk.Label(builder_tab, text="Batch size:")
        self.lBatch_size.place(x=620, y=240)
        self.lEpochs = tk.Label(builder_tab, text="Epochs:")
        self.lEpochs.place(x=620, y=270)
        # self.lresize_shape = tk.Label(builder_tab, text="Resize shape:")
        # self.lresize_shape.place(x=620, y=300)
        # self.lLearning_metrics = tk.Label(builder_tab, text="Learning metrics:")
        # self.lLearning_metrics.place(x=620, y=210)
        self.lName = tk.Label(self.result_tab, text="Name:")
        self.lName.place(x=720, y=15)
        #invisiable lebels
        self.lLearning_rate = tk.Label(builder_tab, text="learning_rate:")
        self.lLearning_rate.place_forget()
        self.lBeta_1 = tk.Label(builder_tab, text="beta_1:")
        self.lBeta_1.place_forget()
        self.lBeta_2 = tk.Label(builder_tab, text="beta_2:")
        self.lBeta_2.place_forget()
        self.lMomentum = tk.Label(builder_tab, text="momentum:")
        self.lMomentum.place_forget()
        self.lRho = tk.Label(builder_tab, text="rho:")
        self.lRho.place_forget()

    #Create Buttons
        self.browse_button = tk.Button(builder_tab, text='Browse', font='Arial 10')
        self.browse_button.bind('<Button-1>', self.browse)
        self.browse_button.place(x=1120, y=10)
        self.start_button = tk.Button(builder_tab, text='Start', font='Arial 10', width=15)
        self.start_button.bind('<Button-1>', self.start)
        self.start_button.place(x=1040, y=350)
        # self.start_button.place(x=970, y=350)
        # self.stop_button = tk.Button(builder_tab, text='Stop', font='Arial 10', width=10)
        # self.stop_button.bind('<Button-1>', self.stop)
        # self.stop_button.place(x=1080, y=350)
        self.save_button = tk.Button(self.result_tab, text='Save', font='Arial 10', width=10, command=self.save)
        self.save_button.place_forget()
        self.save_button.config(state='disabled')

        #self.enretLayer_button = tk.Button(self.tasks_canvas, width=20, height=2, text='Default Enter layer', font='Arial 10')
        #self.enretLayer_button.bind('<Button-1>', self.openMenuEnter)
        #self.enretLayer_button.place(x=200, y=30)

        self.add_layer_button = self.tasks_canvas.create_text(525, 45, text="+",
                                                              justify=tk.CENTER, font="Verdana 18", activefill='lightgreen')
        self.tasks_canvas.tag_bind(self.add_layer_button, '<Button-1>', self.new_layer)

        self.delete_layer_button = self.tasks_canvas.create_text(525, 75, text="-",
                                                                 justify=tk.CENTER, font="Verdana 20", activefill='lightgreen')
        self.tasks_canvas.tag_bind(self.delete_layer_button, '<Button-1>', self.delete_layer)

        self.change_layer_button = self.tasks_canvas.create_text(525, 105, text="⚙",
                                                                 justify=tk.CENTER, font="Verdana 18", activefill='lightgreen')
        self.tasks_canvas.tag_bind(self.change_layer_button, '<Button-1>', self.change_layer)

        self.select_model_button = self.storage_canvas.create_text(1120, 185, text="✓",
                                                                   justify=tk.CENTER, font="Verdana 30",
                                                                   activefill='lightgreen')

        self.delete_model_button = self.storage_canvas.create_text(1120, 325, text="-",
                                                                   justify=tk.CENTER, font="Verdana 35",
                                                                   activefill='lightgreen')
        self.storage_canvas.tag_bind(self.delete_model_button, '<Button-1>', self.delete_model)
        self.storage_canvas.tag_bind(self.select_model_button, '<Button-1>', self.select_model)

    #Text area
        self.log = tk.Text(self.result_tab, width=55, height=30, state='disabled')
        self.log.place(x=710, y=50)

    #Create Entrys
        # visiable Entrys
        self.path = tk.Entry(builder_tab, width=65)
        self.path.place(x=720, y=15)
        self.name = tk.Entry(self.result_tab, width=40)
        self.name.place(x=800, y=15)
        self.batch_size = tk.Entry(builder_tab, width=74)
        self.batch_size.place(x=720, y=240)
        self.epochs = tk.Entry(builder_tab, width=74)
        self.epochs.place(x=720, y=270)
        # self.resize_shape_1 = tk.Entry(builder_tab, width=10)
        # self.resize_shape_1.place(x=720, y=300)
        # self.resize_shape_2 = tk.Entry(builder_tab, width=10)
        # self.resize_shape_2.place(x=805, y=300)
        # invisiable Entrys
        self.learning_rate = tk.Entry(builder_tab, width=74)
        self.learning_rate.place_forget()
        self.beta_1 = tk.Entry(builder_tab, width=74)
        self.beta_1.place_forget()
        self.beta_2 = tk.Entry(builder_tab, width=74)
        self.beta_2.place_forget()
        self.momentum = tk.Entry(builder_tab, width=74)
        self.momentum.place_forget()
        self.rho = tk.Entry(builder_tab, width=74)
        self.rho.place_forget()

        self.batch_size.insert(0, "32")
        self.epochs.insert(0, "5")
        self.beta_1.insert(0, "0.9")
        self.beta_2.insert(0, "0.999")
        self.learning_rate.insert(0, "0.01")
        self.momentum.insert(0, "0")
        self.rho.insert(0, "0.9")
        self.path.insert(0, "C:\\")
        self.path.config(state='readonly')

        listbox_option_items = ['Adam', 'SGD', 'RMSProp', 'Adagrad', 'Adadelta']
        self.listbox_options = tk.Listbox(builder_tab, width=74, height=5, font=('times', 10), exportselection=False)
        self.listbox_options.bind('<<ListboxSelect>>', self.select_lo_item)
        self.listbox_options.place(x=720, y=45)

        self.grayscale_val = tk.BooleanVar()
        self.grayscale_val.set(0)
        self.r1 = tk.Radiobutton(builder_tab, text='Grayscale', variable=self.grayscale_val, value = 1)
        self.r2 = tk.Radiobutton(builder_tab, text='RGB', variable=self.grayscale_val, value = 0)
        self.r1.place(x=1050+35, y=298)
        self.r2.place(x=990+35, y=298)


        for item in listbox_option_items:
            self.listbox_options.insert(tk.END, item)

        self.listbox_builder = tk.Listbox(self.tasks_canvas, width=56, height=10, font=('times', 12), exportselection=False, bd=0)
        self.listbox_builder.bind('<<ListboxSelect>>')
        self.listbox_builder.place(x=50, y=30)

        for item_metrik in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item_metrik)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

        self.listbox_folder = tk.Listbox(self.storage_tab, width=85, height=18, font=('times', 18),
                                         exportselection=False)
        self.listbox_folder.bind('<<ListboxSelect>>')
        self.listbox_folder.place(x=50, y=30)

        # listbox_items_metrik = ['binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy',
        #                         'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
        # self.listbox_metrik = tk.Listbox(builder_tab, width=74, height=5, font=('times', 10), exportselection=False)
        # self.listbox_metrik.bind('<<ListboxSelect>>', self.select_item_metrik)
        # self.listbox_metrik.place(x=720, y=210)
        #
        # for item_metrik in listbox_items_metrik:
        #     self.listbox_metrik.insert(tk.END, item_metrik)
        self.get_models()
        self.notebook.pack(fill=tk.BOTH, expand=1)

    def delete_model(self, event):

        if len(self.listbox_folder.curselection()) < 1:
            msg.showerror("Error", "No layer selected")
            return

        self.constructorAPI.delete_model(name=(self.listbox_folder.get(self.listbox_folder.curselection())))
        self.get_models()

    def select_model(self, event):

        if len(self.listbox_folder.curselection()) < 1:
            msg.showerror("Error", "No layer selected")
            return

        self.constructorAPI.load_model(name=(self.listbox_folder.get(self.listbox_folder.curselection())))

        self.set_plot()
        self.draw_plot(range(1, self.constructorAPI.model.epochs+1), {'accuracy': self.constructorAPI.model.accuracy,
                                                                      'val_accuracy': self.constructorAPI.model.val_accuracy,
                                                                      'loss': self.constructorAPI.model.loss,
                                                                      'val_loss': self.constructorAPI.model.val_loss})
        self.log.config(state='disabled')
        self.log.delete(1.0, tk.END)
        self.log.insert(tk.END, 'Model was trained for ' + str(self.constructorAPI.model.epochs) + ' epochs\n')
        self.log.insert(tk.END, 'Total training accuracy: ' + str(self.constructorAPI.model.accuracy[-1]) + '\n')
        self.log.insert(tk.END, 'Total validation accuracy: ' + str(self.constructorAPI.model.val_accuracy[-1]) + '\n')
        self.log.insert(tk.END, 'Total training loss: ' + str(self.constructorAPI.model.loss[-1]) + '\n')
        self.log.insert(tk.END, 'Total validation loss: ' + str(self.constructorAPI.model.val_loss[-1]) + '\n')
        self.log.config(state='normal')

        self.notebook.tab(1, state="normal")
        self.notebook.select(1)

    def save(self):

        if len(self.name.get()) == 0:
            msg.showwarning('Error', 'Missing model name')
            return

        if len(self.name.get()) > 25:
            msg.showwarning('Error', 'Title exceeds character limit')
            return

        self.constructorAPI.save_model(name=self.name.get())
        self.get_models()
        self.save_button.config(state="disabled")
        self.name.delete(0, tk.END)

    def browse(self, event):
        self.path.config(state='normal')
        self.foldername = filedialog.askdirectory(initialdir="/", title="Select directory")
        self.path.delete(0, tk.END)
        self.path.insert(0, self.foldername)
        self.path.config(state='readonly')

    def start(self, event):

        try:
            learning_rate = float(self.learning_rate.get())
            if learning_rate < 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Learning rate should be a float number >= 0')
            return

        try:
            beta_1 = float(self.beta_1.get())
            if beta_1 <= 0 or beta_1 >= 1:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Beta 1 should be a float number: 0 < beta_1 < 1')

        try:
            beta_2 = float(self.beta_2.get())
            if beta_2 <= 0 or beta_2 >= 1:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Beta 2 should be a float number: 0 < beta_2 < 1')
            return

        try:
            momentum = float(self.momentum.get())
            if momentum < 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Momentum should be a float number >= 0')
            return

        try:
            rho = float(self.rho.get())
            if rho < 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Rho should be a float number >= 0')
            return

        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0 or batch_size > 256:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Batch size should be an integer between 1 and 256')
            return

        try:
            epochs = int(self.epochs.get())
            if epochs <= 0 or epochs > 1000:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Number of epochs should be an integer between 1 and 10000')
            return


        path = self.path.get()

        sys.stdout = self.Logger(self.log)

        self.constructorAPI = NNConstructorAPI()
        try:
            self.constructorAPI.set_data(path, grayscale=self.grayscale_val.get())
        except FileNotFoundError:
            if not path:
                msg.showwarning('Error', 'You need to enter path to data')
            else:
                msg.showwarning('Error', 'System can not find path: ' + path)
            return
        except ValueError as e:
            msg.showwarning('Error', str(e))
            return

        try:
            self.constructorAPI.set_optimizer(algorithm=self.listbox_options.get(self.listbox_options.curselection()),
                                         learning_rate=learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         momentum=momentum,
                                         rho=rho)
        except:
            msg.showwarning('Error', "Choose an option")
            return

        for i in range(0, len(self.layerBuffer)-1):
            temp = self.layerBuffer[i]
            if temp.name == "Convolutional layer":
                self.constructorAPI.add_conv(filters=temp.filters,
                                             kernel_size=(temp.kernelSize_1, temp.kernelSize_2),
                                             activation=temp.activations)
            elif temp.name == "Max pooling layer":
                self.constructorAPI.add_max_pooling(pool_size=(temp.poolSize_1, temp.poolSize_2))
            elif temp.name == "Dense layer":
                self.constructorAPI.add_dense(temp.neurons)
            elif temp.name == "Flatten layer":
                self.constructorAPI.add_flatten()
            elif temp.name == "Dropout layer":
                self.constructorAPI.add_dropout(temp.dropNeurons)
        try:
            self.constructorAPI.build()
        except ValueError as e:
            msg.showwarning('Info', str(e))
            return

        self.queue = queue.Queue()
        self.log_array = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        self.x = []
        self.set_plot()
        self.after(100, self.update_plot)
        self.ThreadedTask(self.notebook ,self.queue, self.constructorAPI,
                          batch_size,
                          epochs,
                          [self.PlotsUpdate(self.queue), self.EndOfTraining(self.save_button, self.notebook)]).start()


    class ThreadedTask(Thread):
        def __init__(self, notebook, queue, api, batch_size, epochs, callbacks):
            Thread.__init__(self, daemon=True)
            self.queue = queue
            self.notebook = notebook
            self.api = api
            self.batch_size = batch_size
            self.epochs = epochs
            self.callbacks = callbacks

        def run(self):
            self.notebook.tab(1, state="normal")
            self.notebook.tab(0, state="disabled")
            self.notebook.tab(2, state="disabled")
            self.notebook.select(1)
            try:
                self.api.fit(int(self.batch_size), int(self.epochs), callbacks=self.callbacks)
            except:
                msg.showwarning('Info', "No results for model with incorrect layers order.")
                self.notebook.tab(1, state="disabled")
                self.notebook.tab(0, state="normal")
                self.notebook.tab(2, state="normal")
                self.notebook.select(0)
                return

    def get_models(self):
        self.listbox_folder.delete(0, tk.END)
        list_folders = [name for name in os.listdir("models/") if os.path.isdir(os.path.join("models/", name))]

        for item_metrik in list_folders:
            self.listbox_folder.insert(tk.END, item_metrik)

    def select_lo_item(self, event):
        value = (self.listbox_options.get(self.listbox_options.curselection()))
        self.learning_rate.place_forget()
        self.beta_1.place_forget()
        self.beta_2.place_forget()
        self.momentum.place_forget()
        self.rho.place_forget()
        self.lLearning_rate.place_forget()
        self.lBeta_1.place_forget()
        self.lBeta_2.place_forget()
        self.lMomentum.place_forget()
        self.lRho.place_forget()
        if value == 'Adam':
            self.learning_rate.delete(0, tk.END)
            self.beta_1.delete(0, tk.END)
            self.beta_2.delete(0, tk.END)

            self.learning_rate.insert(0, "0.001")
            self.beta_1.insert(0, "0.9")
            self.beta_2.insert(0, "0.999")

            self.lLearning_rate.place(x=620, y=150)
            self.lBeta_1.place(x=620, y=180)
            self.lBeta_2.place(x=620, y=210)
            self.learning_rate.place(x=720, y=150)
            self.beta_1.place(x=720, y=180)
            self.beta_2.place(x=720, y=210)
        if value == 'SGD':
            self.learning_rate.delete(0, tk.END)
            self.momentum.delete(0, tk.END)

            self.learning_rate.insert(0, "0.01")
            self.momentum.insert(0, "0")

            self.lLearning_rate.place(x=620, y=150)
            self.lMomentum.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.momentum.place(x=720, y=180)
        if value == 'RMSProp':
            self.learning_rate.delete(0, tk.END)
            self.rho.delete(0, tk.END)

            self.learning_rate.insert(0, "0.001")
            self.rho.insert(0, "0.9")

            self.lLearning_rate.place(x=620, y=150)
            self.lRho.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.rho.place(x=720, y=180)
        if value == 'Adagrad':
            self.learning_rate.delete(0, tk.END)

            self.learning_rate.insert(0, "0.01")

            self.lLearning_rate.place(x=620, y=150)
            self.learning_rate.place(x=720, y=150)
        if value == 'Adadelta':
            self.learning_rate.delete(0, tk.END)
            self.rho.delete(0, tk.END)

            self.learning_rate.insert(0, "1")
            self.rho.insert(0, "0.95")

            self.lLearning_rate.place(x=620, y=150)
            self.lRho.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.rho.place(x=720, y=180)

    def new_layer(self, event):

        if len(self.listbox_builder.curselection()) < 1:
            msg.showerror("Error", "No layer selected")
            return

        layer = tk.Toplevel(self)
        layer.title("Add layer")
        layer.geometry("450x310")
        layer.resizable(width=False, height=False)

        listbox_item_layer = ['Convolutional', 'MaxPooling', 'Dense', 'Flatten', 'Dropout']
        layer.listbox_layer = tk.Listbox(layer, width=50, height=5, font=('times', 10), exportselection=False)
        layer.listbox_layer.bind('<<ListboxSelect>>',lambda event: self.select_layer(layer, self))
        layer.listbox_layer.place(x=65, y=30)

        listbox_conv_layer = ['relu', 'sigmoid', 'tanh']
        layer.listbox_conv_layer = tk.Listbox(layer, width=28, height=3, font=('times', 10), exportselection=False)
        layer.listbox_conv_layer.bind('<<ListboxSelect>>')

        layer.filters = tk.Entry(layer, width=28)
        layer.filters.place_forget()
        layer.kernelSize_1 = tk.Entry(layer, width=10)
        layer.kernelSize_1.place_forget()
        layer.kernelSize_2 = tk.Entry(layer, width=10)
        layer.poolSize_1 = tk.Entry(layer, width=10)
        layer.poolSize_1.place_forget()
        layer.poolSize_2 = tk.Entry(layer, width=10)
        layer.poolSize_2.place_forget()
        layer.neurons = tk.Entry(layer, width=28)
        layer.neurons.place_forget()
        layer.dropNeurons = tk.Entry(layer, width=28)
        layer.dropNeurons.place_forget()

        layer.filters.insert(0, "32")
        layer.kernelSize_1.insert(0, "3")
        layer.kernelSize_2.insert(0, "3")
        layer.poolSize_1.insert(0, "2")
        layer.poolSize_2.insert(0, "2")
        layer.neurons.insert(0, "64")
        layer.dropNeurons.insert(0, "0.5")

        layer.lFilters = tk.Label(layer, text="Number of filters:")
        layer.lFilters.place_forget()
        layer.lKernelSize = tk.Label(layer, text="Kernel size:")
        layer.lKernelSize.place_forget()
        layer.lPoolSize = tk.Label(layer, text="Pool size:")
        layer.lPoolSize.place_forget()
        layer.lNeurons = tk.Label(layer, text="Number of neurons:")
        layer.lNeurons.place_forget()
        layer.lDropNeurons = tk.Label(layer, text="Discarded neurons:")
        layer.lDropNeurons.place_forget()
        layer.lActivations = tk.Label(layer, text="Activations:")
        layer.lActivations.place_forget()

        for item in listbox_item_layer:
            layer.listbox_layer.insert(tk.END, item)

        for item in listbox_conv_layer:
            layer.listbox_conv_layer.insert(tk.END, item)

        layer.listbox_conv_layer.select_set(0)

        layer.add_button = tk.Button(layer, width=10, height=1, text='Add',
                                       font='Arial 10')
        layer.add_button.bind('<Button-1>', self.eror_selected)
        layer.add_button.place(x=100, y=265)

        layer.clous_button = tk.Button(layer, width=10, height=1, text='Cancel',
                                     font='Arial 10')
        layer.clous_button.bind('<Button-1>', lambda event: layer.destroy())
        layer.clous_button.place(x=250, y=265)

    def eror_selected(self, event):
        msg.showerror("Error", "No architecture selected")

    def select_layer(event, layer, self):
        value = (layer.listbox_layer.get(layer.listbox_layer.curselection()))
        layer.filters.place_forget()
        layer.kernelSize_1.place_forget()
        layer.kernelSize_2.place_forget()
        layer.poolSize_1.place_forget()
        layer.poolSize_2.place_forget()
        layer.neurons.place_forget()
        layer.dropNeurons.place_forget()
        layer.listbox_conv_layer.place_forget()

        layer.lFilters.place_forget()
        layer.lKernelSize.place_forget()
        layer.lPoolSize.place_forget()
        layer.lNeurons.place_forget()
        layer.lDropNeurons.place_forget()
        layer.lActivations.place_forget()

        layer.add_button = tk.Button(layer, width=10, height=1, text='Add',
                                     font='Arial 10')
        layer.add_button.place_forget()

        if value == 'Convolutional':
            layer.lFilters.place(x=65, y=140)
            layer.lKernelSize.place(x=65, y=170)
            layer.filters.place(x=195, y=140)
            layer.kernelSize_1.place(x=195, y=170)
            layer.kernelSize_2.place(x=305, y=170)
            layer.lActivations.place(x=65, y=200)
            layer.listbox_conv_layer.place(x=195, y=200)

            layer.add_button.bind('<Button-1>',lambda event: self.add_Convolutional(layer))
            layer.add_button.place(x=100, y=265)
        if value == 'MaxPooling':
            layer.lPoolSize.place(x=65, y=140)
            layer.poolSize_1.place(x=195, y=140)
            layer.poolSize_2.place(x=305, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_MaxPooling(layer))
            layer.add_button.place(x=100, y=265)
        if value == 'Dense':
            layer.lNeurons.place(x=65, y=140)
            layer.neurons.place(x=195, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_Dense(layer))
            layer.add_button.place(x=100, y=265)
        if value == 'Flatten':
            layer.add_button.bind('<Button-1>', lambda event: self.add_Flatten(layer))
            layer.add_button.place(x=100, y=265)
        if value == 'Dropout':
            layer.lDropNeurons.place(x=65, y=140)
            layer.dropNeurons.place(x=195, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_Dropout(layer))
            layer.add_button.place(x=100, y=265)

    def add_Convolutional(self, layer):
        try:
            kernel_size_1 = int(layer.kernelSize_1.get())
            kernel_size_2 = int(layer.kernelSize_2.get())
            if kernel_size_1 <= 0 or kernel_size_2 <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Kernel size should be a pair of ints > 0')
            return

        try:
            filters = int(layer.filters.get())
            if filters <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Number of filters should be a positive integer')
            return

        activation = (layer.listbox_conv_layer.get(layer.listbox_conv_layer.curselection()))

        newClass = self.layerConvolutional()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0]+1, newClass.name)
        self.layerBuffer.insert(selection[0], newClass)
        newClass.kernelSize_1 = kernel_size_1
        newClass.kernelSize_2 = kernel_size_2
        newClass.filters = filters
        newClass.activations = activation
        layer.destroy()
        self.listbox_builder.delete(0,tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_MaxPooling(self, layer):
        try:
            pool_size_1 = int(layer.poolSize_1.get())
            pool_size_2 = int(layer.poolSize_2.get())
            if pool_size_1 <= 0 or pool_size_2 <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Pool size should be a pair of ints > 0')
            return

        newClass = self.layerMaxPooling()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0], newClass)
        newClass.poolSize_1 = pool_size_1
        newClass.poolSize_2 = pool_size_2
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_Dense(self, layer):
        try:
            neurons = int(layer.neurons.get())
            if neurons <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Number of neurons should be a positive integer')
            return

        newClass = self.layerDense()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0], newClass)
        newClass.neurons = neurons
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_Flatten(self, layer):
        newClass = self.layerFlatten()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0], newClass)
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_Dropout(self, layer):
        try:
            drop_rate = float(layer.dropNeurons.get())
            if not 0 <= drop_rate < 1:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Drop rate should be a float number between 0 and 1')
            return

        newClass = self.layerDropout()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0], newClass)
        newClass.dropNeurons = drop_rate
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def delete_layer(self, event):
        selection = (self.listbox_builder.curselection())

        try:
            error_delete = self.listbox_builder.get(self.listbox_builder.curselection())
        except:
            msg.showerror("Error", "No layer selected")
            return


        if len(self.listbox_builder.curselection()) < 1:
            msg.showerror("Error", "No layer selected")
            return

        if (self.listbox_builder.get(tk.ANCHOR) == 'Default Enter layer') or (
                self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer'):
            msg.showerror("Error", "Unable to delete static layers")
            return

        del self.listbox_items_builder[selection[0]]
        del self.layerBuffer[selection[0] - 1]

        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def change_layer(self, event):

        selection = (self.listbox_builder.curselection())
        try:
            error_change = self.listbox_builder.get(self.listbox_builder.curselection())
        except:
            msg.showerror("Error", "No layer selected")
            return

        value = self.layerBuffer[selection[0] - 1]

        if ((self.listbox_builder.get(self.listbox_builder.curselection())) == 'Default Enter layer') \
                or ((self.listbox_builder.get(self.listbox_builder.curselection())) == 'Default Exit layer'):
            msg.showerror("Error", "Unable to change static layers")
            return

        if value.name == "Flatten layer":
            msg.showwarning('Warning', 'Flatten impossible to change')
            return

        layer = tk.Toplevel(self)
        layer.title("Change layer")
        layer.geometry("450x270")
        layer.resizable(width=False, height=False)

        layer.filters = tk.Entry(layer, width=28)
        layer.filters.place_forget()
        layer.kernelSize_1 = tk.Entry(layer, width=10)
        layer.kernelSize_2 = tk.Entry(layer, width=10)
        layer.kernelSize_1.place_forget()
        layer.kernelSize_2.place_forget()
        layer.poolSize_1 = tk.Entry(layer, width=10)
        layer.poolSize_1.place_forget()
        layer.poolSize_2 = tk.Entry(layer, width=10)
        layer.poolSize_2.place_forget()
        layer.neurons = tk.Entry(layer, width=28)
        layer.neurons.place_forget()
        layer.dropNeurons = tk.Entry(layer, width=28)
        layer.dropNeurons.place_forget()

        layer.lFilters = tk.Label(layer, text="Number of filters:")
        layer.lFilters.place_forget()
        layer.lKernelSize = tk.Label(layer, text="Kernel size:")
        layer.lKernelSize.place_forget()
        layer.lPoolSize = tk.Label(layer, text="Pool size:")
        layer.lPoolSize.place_forget()
        layer.lNeurons = tk.Label(layer, text="Number of neurons:")
        layer.lNeurons.place_forget()
        layer.lDropNeurons = tk.Label(layer, text="Discarded neurons:")
        layer.lDropNeurons.place_forget()
        layer.lActivations = tk.Label(layer, text="Activations:")
        layer.lActivations.place_forget()

        layer.add_button = tk.Button(layer, width=10, height=1, text='Change',
                                     font='Arial 10')
        layer.add_button.bind('<Button-1>', self.eror_selected)
        layer.add_button.place(x=100, y=225)

        listbox_conv_layer = ['relu', 'sigmoid', 'tanh']
        layer.listbox_conv_layer = tk.Listbox(layer, width=28, height=3, font=('times', 10), exportselection=False)
        layer.listbox_conv_layer.bind('<<ListboxSelect>>')
        layer.listbox_conv_layer.place_forget()

        for item in listbox_conv_layer:
            layer.listbox_conv_layer.insert(tk.END, item)

        layer.clous_button = tk.Button(layer, width=10, height=1, text='Cancel',
                                       font='Arial 10')
        layer.clous_button.bind('<Button-1>', lambda event: layer.destroy())
        layer.clous_button.place(x=250, y=225)

        if value.name == 'Convolutional layer':
            layer.lFilters.place(x=65, y=15)
            layer.lKernelSize.place(x=65, y=45)
            layer.filters.place(x=195, y=15)
            layer.filters.insert(0, value.filters)
            layer.kernelSize_1.place(x=195, y=45)
            layer.kernelSize_2.place(x=305, y=45)
            layer.kernelSize_1.insert(0, value.kernelSize_1)
            layer.kernelSize_2.insert(0, value.kernelSize_2)
            layer.add_button.bind('<Button-1>', lambda event: self.change_Convolutional(layer))
            layer.add_button.place(x=100, y=225)
            layer.listbox_conv_layer.place(x=195, y=75)
            layer.lActivations.place(x=65, y=75)
            layer.listbox_conv_layer.select_set(0)
        if value.name == 'Max pooling layer':
            layer.lPoolSize.place(x=65, y=15)
            layer.poolSize_1.place(x=195, y=15)
            layer.poolSize_2.place(x=305, y=15)
            layer.poolSize_1.insert(0, value.poolSize_1)
            layer.poolSize_2.insert(0, value.poolSize_2)
            layer.add_button.bind('<Button-1>', lambda event: self.change_MaxPooling(layer))
            layer.add_button.place(x=100, y=225)
        if value.name == 'Dense layer':
            layer.lNeurons.place(x=65, y=15)
            layer.neurons.place(x=195, y=15)
            layer.neurons.insert(0, value.neurons)
            layer.add_button.bind('<Button-1>', lambda event: self.change_Dense(layer))
            layer.add_button.place(x=100, y=225)
        if value.name == 'Flatten layer':
            layer.add_button.bind('<Button-1>', lambda event: self.change_Flatten(layer))
            layer.add_button.place(x=100, y=225)
        if value.name == 'Dropout layer':
            layer.lDropNeurons.place(x=65, y=15)
            layer.dropNeurons.place(x=195, y=15)
            layer.dropNeurons.insert(0, value.dropNeurons)
            layer.add_button.bind('<Button-1>', lambda event: self.change_Dropout(layer))
            layer.add_button.place(x=100, y=225)

    def change_Convolutional(self, layer):
        try:
            kernel_size_1 = int(layer.kernelSize_1.get())
            kernel_size_2 = int(layer.kernelSize_2.get())
            if kernel_size_1 <= 0 or kernel_size_2 <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Kernel size should be a pair of ints > 0')
            return

        try:
            filters = int(layer.filters.get())
            if filters <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Number of filters should be a positive integer')
            return

        selection = (self.listbox_builder.curselection())
        value = self.layerBuffer[selection[0] - 1]
        value.filters = filters
        value.kernelSize_1 = kernel_size_1
        value.kernelSize_2 = kernel_size_2
        value.activations = (layer.listbox_conv_layer.get(layer.listbox_conv_layer.curselection()))
        layer.destroy()

    def change_MaxPooling(self, layer):
        try:
            pool_size_1 = int(layer.poolSize_1.get())
            pool_size_2 = int(layer.poolSize_2.get())
            if pool_size_1 <= 0 or pool_size_2 <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Pool size should be a pair of ints > 0')
            return

        selection = (self.listbox_builder.curselection())
        value = self.layerBuffer[selection[0] - 1]
        value.poolSize_1 = pool_size_1
        value.poolSize_2 = pool_size_2
        layer.destroy()

    def change_Dense(self, layer):
        try:
            neurons = int(layer.neurons.get())
            if neurons <= 0:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Number of neurons should be a positive integer')
            return

        selection = (self.listbox_builder.curselection())
        value = self.layerBuffer[selection[0] - 1]
        value.neurons = neurons
        layer.destroy()

    def change_Flatten(self, layer):
        msg.showwarning('Warning', 'Flatten impossible to change')

    def change_Dropout(self, layer):
        try:
            drop_rate = float(layer.dropNeurons.get())
            if not 0 <= drop_rate < 1:
                raise ValueError()
        except ValueError:
            msg.showwarning('Error', 'Drop rate should be a float number between 0 and 1')
            return

        selection = (self.listbox_builder.curselection())
        value = self.layerBuffer[selection[0] - 1]
        value.dropNeurons = drop_rate
        layer.destroy()

    ########################################################

    # class place

    class Logger(object):
        def __init__(self, text_field):
            self.terminal = sys.stdout
            self.text = text_field

        def write(self, message):
            self.text.config(state='normal')
            self.terminal.write(message)
            self.text.insert(tk.END, message)
            self.text.see(tk.END)
            self.text.config(state='disabled')

        def flush(self):
            pass

    class layerConvolutional:
        name = "Convolutional layer"
        number = 0
        filters = 32
        kernelSize_1 = 3
        kernelSize_2 = 3
        activations = 'relu'

        def getNumber(self):
            print(self.number)

    class layerMaxPooling:
        name = "Max pooling layer"
        number = 0
        poolSize_1 = 2
        poolSize_2 = 2

        def getNumber(self):
            print(self.number)

    class layerDense:
        name = "Dense layer"
        number = 0
        neurons = 64

        def getNumber(self):
            print(self.number)

    class layerFlatten:
        name = "Flatten layer"
        number = 0

        def getNumber(self):
            print(self.number)

    class layerDropout:
        name = "Dropout layer"
        number = 0
        dropNeurons = 0.5

        def getNumber(self):
            print(self.number)

    class PlotsUpdate(Callback):
        def __init__(self, queue):
            super(Callback, self).__init__()
            self.queue = queue

        def on_epoch_end(self, epoch, logs=None):
            self.queue.put(logs)

    class EndOfTraining(Callback):
        def __init__(self, button, notebook):
            super(Callback, self).__init__()
            self.button = button
            self.notebook = notebook


        def on_train_end(self, logs=None):
            self.button.config(state="normal")
            self.notebook.tab(0, state="normal")
            self.notebook.tab(2, state="normal")

    def set_plot(self):
        figure = Figure(figsize=(9, 8), dpi=75)
        self.accuracy_plot = figure.add_subplot(211)
        self.loss_plot = figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(figure, self.result_tab)  #
        self.canvas.get_tk_widget().grid(row=0, column=0)  #
        self.save_button.place(x=1080, y=10)  #

    def update_plot(self):
        try:
            logs = self.queue.get(False)

            self.log_array['accuracy'].append(logs['accuracy'])
            self.log_array['val_accuracy'].append(logs['val_accuracy'])
            self.log_array['loss'].append(logs['loss'])
            self.log_array['val_loss'].append(logs['val_loss'])

            if len(self.x):
                self.x.append(self.x[-1] + 1)
                self.draw_plot(self.x, self.log_array)
            else:
                self.x = [1]

            self.after(100, self.update_plot)
        except queue.Empty:
            self.after(100, self.update_plot)

    def draw_plot(self, x, log_array):
        self.accuracy_plot.clear()
        self.loss_plot.clear()

        self.accuracy_plot.plot(x, log_array['accuracy'], label='Training')
        self.accuracy_plot.plot(x, log_array['val_accuracy'], label='Validation')
        self.accuracy_plot.legend()
        self.accuracy_plot.set_xlabel('Epochs')
        self.accuracy_plot.set_ylabel('Accuracy')
        self.accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.loss_plot.plot(x, log_array['loss'], label='Training')
        self.loss_plot.plot(x, log_array['val_loss'], label='Validation')
        self.loss_plot.legend()
        self.loss_plot.set_xlabel('Epochs')
        self.loss_plot.set_ylabel('Loss')
        self.loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.canvas.draw()

if __name__ == "__main__":
    nni = NNI()
    nni.mainloop()