import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


model = None  # model брать из API
with open('./models/test_save_2/test_save_2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


fit, (accuracy_plot, loss_plot) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

x = range(1, len(model.accuracy) + 1)

accuracy_plot.plot(x, model.accuracy, label='Training')
accuracy_plot.plot(x, model.val_accuracy, label='Validation')
accuracy_plot.legend()
accuracy_plot.set_xlabel('Epochs')
accuracy_plot.set_ylabel('Accuracy')
accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

loss_plot.plot(x, model.loss, label='Training')
loss_plot.plot(x, model.val_loss, label='Validation')
loss_plot.legend()
loss_plot.set_xlabel('Epochs')
loss_plot.set_ylabel('Loss')
loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()