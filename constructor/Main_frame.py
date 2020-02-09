import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as msg
from tkinter.ttk import Notebook

class NNI(tk.Tk):

    listbox_items_builder = ['Default Enter layer']
    layerBuffer = []

    def __init__(self):
        super().__init__()

    #Windows option
        self.title("NNI")
        self.geometry("1200x430")
        self.resizable(width=False, height=False)

    #Create tab
        self.notebook = Notebook(self)
        builder_tab = tk.Frame(self.notebook)
        result_tab = tk.Frame(self.notebook)
        self.notebook.add(builder_tab, text="Building architecture")
        self.notebook.add(result_tab, text="Result")

    #Create canvas for dinamic blocs
        self.tasks_canvas = tk.Canvas(builder_tab, width=600)
        self.tasks_frame = tk.Frame(self.tasks_canvas)
        self.text_frame = tk.Frame(builder_tab)
        self.scrollbar = tk.Scrollbar(self.tasks_canvas, orient="vertical", command=self.
                                      tasks_canvas.yview)
        self.tasks_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.tasks_canvas.place(x=0, y=0)
        self.scrollbar.place_forget()

    #Create lebels
        #visiable lebels
        self.lPath = tk.Label(builder_tab, text="Path:")
        self.lPath.place(x=620, y=15)
        self.lLearning_options = tk.Label(builder_tab, text="Learning options:")
        self.lLearning_options.place(x=620, y=45)
        self.lLearning_metrics = tk.Label(builder_tab, text="Learning metrics:")
        self.lLearning_metrics.place(x=620, y=210)
        self.lName = tk.Label(self.tasks_canvas, text="Layers")
        self.lName.place(x=265, y=5)
        #invisiable lebels
        self.lLearning_rate = tk.Label(builder_tab, text="learning_rate:")
        self.lLearning_rate.place_forget()
        self.lBeta_1 = tk.Label(builder_tab, text="beta_1:")
        self.lBeta_1.place_forget()
        self.lMomentum = tk.Label(builder_tab, text="momentum:")
        self.lMomentum.place_forget()
        self.lRho = tk.Label(builder_tab, text="rho:")
        self.lRho.place_forget()

    #Create Buttons
        self.browse_button = tk.Button(builder_tab, text='Обзор', font='Arial 10')
        self.browse_button.bind('<Button-1>', self.delete)
        self.browse_button.place(x=1120, y=10)
        self.start_button = tk.Button(builder_tab, text='Начало', font='Arial 10')
        self.start_button.bind('<Button-1>', self.start)
        self.start_button.place(x=1020, y=320)
        self.stop_button = tk.Button(builder_tab, text='Остановка', font='Arial 10')
        self.stop_button.bind('<Button-1>', self.stop)
        self.stop_button.place(x=1080, y=320)

        #self.enretLayer_button = tk.Button(self.tasks_canvas, width=20, height=2, text='Default Enter layer', font='Arial 10')
        #self.enretLayer_button.bind('<Button-1>', self.openMenuEnter)
        #self.enretLayer_button.place(x=200, y=30)

        self.plus = self.tasks_canvas.create_text(525, 45, text="+",
                                                  justify=tk.CENTER, font="Verdana 18", activefill='lightgreen')
        self.tasks_canvas.tag_bind(self.plus, '<Button-1>', self.new_layer)

        self.minus = self.tasks_canvas.create_text(525, 75, text="-",
                                                  justify=tk.CENTER, font="Verdana 20", activefill='lightgreen')
        self.tasks_canvas.tag_bind(self.minus, '<Button-1>', self.delete_layer)

        #self.exitLayer_button = tk.Button(self.tasks_canvas, state=tk.DISABLED, width=20, height=2,
        #                               text='Default Exit layer', font='Arial 10')
        #self.exitLayer_button.place(x=self.cordx, y=self.cordy)

    #Create Entrys
        # visiable Entrys
        self.path = tk.Entry(builder_tab, width=65)
        self.path.place(x=720, y=15)
        self.name = tk.Entry(builder_tab, width=47)
        self.name.place(x=720, y=325)
        # invisiable Entrys
        self.learning_rate = tk.Entry(builder_tab, width=74)
        self.learning_rate.place_forget()
        self.beta_1 = tk.Entry(builder_tab, width=74)
        self.beta_1.place_forget()
        self.momentum = tk.Entry(builder_tab, width=74)
        self.momentum.place_forget()
        self.rho = tk.Entry(builder_tab, width=74)
        self.rho.place_forget()

        #self.tasks_canvas.create_line(285, 60, 285, 120, fill='black',
        #                              width=2, arrow=tk.LAST,
        #                              arrowshape="10 20 10")


        listbox_option_items = ['Adam', 'SGD', 'RMSProp', 'Adagrad', 'Adadelta']
        self.listbox_options = tk.Listbox(builder_tab, width=74, height=5, font=('times', 10), exportselection=False)
        self.listbox_options.bind('<<ListboxSelect>>', self.select_lo_item)
        self.listbox_options.place(x=720, y=45)

        for item in listbox_option_items:
            self.listbox_options.insert(tk.END, item)

        self.listbox_builder = tk.Listbox(self.tasks_canvas, width=75, height=8, font=('times', 10), exportselection=False)
        self.listbox_builder.bind('<<ListboxSelect>>')
        self.listbox_builder.place(x=50, y=30)

        for item_metrik in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item_metrik)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

        listbox_items_metrik = ['binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy',
                                'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
        self.listbox_metrik = tk.Listbox(builder_tab, width=74, height=5, font=('times', 10), exportselection=False)
        self.listbox_metrik.bind('<<ListboxSelect>>', self.select_item_metrik)
        self.listbox_metrik.place(x=720, y=210)

        for item_metrik in listbox_items_metrik:
            self.listbox_metrik.insert(tk.END, item_metrik)

        self.notebook.pack(fill=tk.BOTH, expand=1)

    def delete(self, event):
        self.filename = filedialog.askdirectory(initialdir="/", title="Select directory")
        self.path.delete(0, tk.END)
        self.path.insert(0, self.filename)

    def start(self, event):
        print('start')

    def stop(self, event):
        print('stop')

    def openMenuEnter(self, event):
        pass

    def select_lo_item(self, event):
        value = (self.listbox_options.get(self.listbox_options.curselection()))
        self.learning_rate.place_forget()
        self.beta_1.place_forget()
        self.momentum.place_forget()
        self.rho.place_forget()
        self.lLearning_rate.place_forget()
        self.lBeta_1.place_forget()
        self.lMomentum.place_forget()
        self.lRho.place_forget()
        if value == 'Adam':
            self.lLearning_rate.place(x=620, y=150)
            self.lBeta_1.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.beta_1.place(x=720, y=180)
        if value == 'SGD':
            self.lLearning_rate.place(x=620, y=150)
            self.lMomentum.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.momentum.place(x=720, y=180)
        if value == 'RMSProp':
            self.lLearning_rate.place(x=620, y=150)
            self.lRho.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.rho.place(x=720, y=180)
        if value == 'Adagrad':
            self.lLearning_rate.place(x=620, y=150)
            self.learning_rate.place(x=720, y=150)
        if value == 'Adadelta':
            self.lLearning_rate.place(x=620, y=150)
            self.lRho.place(x=620, y=180)
            self.learning_rate.place(x=720, y=150)
            self.rho.place(x=720, y=180)

    def select_item_metrik(self, event):
        value = (self.listbox_metrik.get(self.listbox_metrik.curselection()))
        print(value)

    def new_layer(self, event):

        if len(self.listbox_builder.curselection()) < 1:
            msg.showerror("Error", "No layer selected")
            return

        layer = tk.Toplevel(self)
        layer.title("Add layer")
        layer.geometry("450x270")
        layer.resizable(width=False, height=False)
        listbox_item_layer = ['Convolutional', 'MaxPooling', 'Dense', 'Flatten', 'Dropout']

        layer.listbox_layer = tk.Listbox(layer, width=50, height=5, font=('times', 10), exportselection=False)
        layer.listbox_layer.bind('<<ListboxSelect>>',lambda event: self.select_layer(layer, self))
        layer.listbox_layer.place(x=65, y=30)

        layer.filters = tk.Entry(layer, width=20)
        layer.filters.place_forget()
        layer.kernelSize = tk.Entry(layer, width=20)
        layer.kernelSize.place_forget()
        layer.poolSize = tk.Entry(layer, width=20)
        layer.poolSize.place_forget()
        layer.neurons = tk.Entry(layer, width=20)
        layer.neurons.place_forget()
        layer.dropNeurons = tk.Entry(layer, width=20)
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

        for item in listbox_item_layer:
            layer.listbox_layer.insert(tk.END, item)

        layer.add_button = tk.Button(layer, width=10, height=1, text='Add',
                                       font='Arial 10')
        layer.add_button.bind('<Button-1>', self.stop)
        layer.add_button.place(x=100, y=225)

        layer.clous_button = tk.Button(layer, width=10, height=1, text='Cancel',
                                     font='Arial 10')
        layer.clous_button.bind('<Button-1>', lambda event: layer.destroy())
        layer.clous_button.place(x=250, y=225)

    def select_layer(event, layer, self):
        value = (layer.listbox_layer.get(layer.listbox_layer.curselection()))
        layer.filters.place_forget()
        layer.kernelSize.place_forget()
        layer.poolSize.place_forget()
        layer.neurons.place_forget()
        layer.dropNeurons.place_forget()

        layer.lFilters.place_forget()
        layer.lKernelSize.place_forget()
        layer.lPoolSize.place_forget()
        layer.lNeurons.place_forget()
        layer.lDropNeurons.place_forget()

        layer.add_button = tk.Button(layer, width=10, height=1, text='Add',
                                     font='Arial 10')
        layer.add_button.place_forget()

        if value == 'Convolutional':
            layer.lFilters.place(x=65, y=140)
            layer.lKernelSize.place(x=65, y=170)
            layer.filters.place(x=245, y=140)
            layer.kernelSize.place(x=245, y=170)

            layer.add_button.bind('<Button-1>',lambda event: self.add_Convolutional(layer))
            layer.add_button.place(x=100, y=225)
        if value == 'MaxPooling':
            layer.lPoolSize.place(x=65, y=140)
            layer.poolSize.place(x=245, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_MaxPooling(layer))
            layer.add_button.place(x=100, y=225)
        if value == 'Dense':
            layer.lNeurons.place(x=65, y=140)
            layer.neurons.place(x=245, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_Dense(layer))
            layer.add_button.place(x=100, y=225)
        if value == 'Flatten':
            pass
            layer.add_button.bind('<Button-1>', lambda event: self.add_Flatten(layer))
            layer.add_button.place(x=100, y=225)
        if value == 'Dropout':
            layer.lDropNeurons.place(x=65, y=140)
            layer.dropNeurons.place(x=245, y=140)
            layer.add_button.bind('<Button-1>', lambda event: self.add_Dropout(layer))
            layer.add_button.place(x=100, y=225)

    def add_Convolutional(self, layer):
        newClass = self.layerConvolutional()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0]+1, newClass.name)
        self.layerBuffer.insert(selection[0]+1, newClass)
        newClass.kernelSize = layer.kernelSize.get()
        newClass.filters = layer.filters.get()
        print("kernel Size=", newClass.kernelSize)
        print("Filters =", newClass.filters)
        #for i in self.layerBuffer:
        #    name = self.layerBuffer[i]
        layer.destroy()
        self.listbox_builder.delete(0,tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_MaxPooling(self, layer):
        newClass = self.layerMaxPooling()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0] + 1, newClass)
        newClass.poolSize = layer.poolSize.get()
        print("poolSize=", newClass.poolSize)
        # for i in self.layerBuffer:
        #    name = self.layerBuffer[i]
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_Dense(self, layer):
        newClass = self.layerDense()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0] + 1, newClass)
        newClass.neurons = layer.neurons.get()
        print("neurons=", newClass.neurons)
        # for i in self.layerBuffer:
        #    name = self.layerBuffer[i]
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
        self.layerBuffer.insert(selection[0] + 1, newClass)
        # for i in self.layerBuffer:
        #    name = self.layerBuffer[i]
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

    def add_Dropout(self, layer):
        newClass = self.layerDropout()
        selection = (self.listbox_builder.curselection())
        if self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer':
            newClass.number = selection[0]
        else:
            newClass.number = selection[0] + 1
        self.listbox_items_builder.insert(selection[0] + 1, newClass.name)
        self.layerBuffer.insert(selection[0] + 1, newClass)
        # for i in self.layerBuffer:
        #    name = self.layerBuffer[i]
        newClass.dropNeurons = layer.dropNeurons.get()
        print("dropNeurons", newClass.dropNeurons)
        layer.destroy()
        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')

        #self.cordy+= 10
        #self.exitLayer_button.place(x=self.cordx, y=self.cordy)
        #self.plus = self.tasks_canvas.create_text(305, self.cordy-40, text="+",
        #                                          justify=tk.CENTER, font="Verdana 14", activefill='lightgreen')
        #self.tasks_canvas.tag_bind(self.plus, '<Button-1>', self.new_layer)
        #
        #self.tasks_canvas.create_line(285, self.cordy-80, 285, self.cordy, fill='black',
        #                              width=2, arrow=tk.LAST,
        #                              arrowshape="10 20 10")
        #
        #self.dropoutLayer_button = tk.Button(self.tasks_canvas, width=20, height=2, text='Dropout',
        #                                   font='Arial 10')
        #self.dropoutLayer_button.bind('<Button-1>', self.openMenuEnter)
        #self.dropoutLayer_button.place(x=200, y=self.cordy-self.y)

    def delete_layer(self, event):
        selection = (self.listbox_builder.curselection())

        print(selection[0])

        if (self.listbox_builder.get(tk.ANCHOR) == 'Default Enter layer') or (self.listbox_builder.get(tk.ANCHOR) == 'Default Exit layer'):
            msg.showerror("Error", "Unable to delete static layers")
            return

        del self.listbox_items_builder[selection[0]]
        del self.layerBuffer[selection[0]-1]

        self.listbox_builder.delete(0, tk.END)
        for item in self.listbox_items_builder:
            self.listbox_builder.insert(tk.END, item)
        self.listbox_builder.insert(tk.END, 'Default Exit layer')


    class layerConvolutional:
        name = "Convolutional layer"
        number = 0
        filters = 0
        kernelSize = 0

        def getNumber(self):
            print(self.number)


    class layerMaxPooling:
        name = "Max pooling layer"
        number = 0
        poolSize = 0

        def getNumber(self):
            print(self.number)


    class layerDense:
        name = "Dense layer"
        number = 0
        neurons = 0


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
        dropNeurons = 0

        def getNumber(self):
            print(self.number)

if __name__ == "__main__":
    nni = NNI()
    nni.mainloop()