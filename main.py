from tkinter import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import os
import time

# path where we will store any models we plan to save and their history
# this is where all files for this application will be stored.
storage_path = "/MnistTrainingApp/"

# get the models that have already been saved by this application
saved_models = []
#print(os.listdir(storage_path))
# get only the different model names
for filename in os.listdir(storage_path):
    if filename.endswith('.h5'):
        saved_models.append(filename.removesuffix('.h5'))
#print(saved_models)
saved_models.sort()

# first we load the training and testing samples from mnist
# preprocessing

# First I preprocess the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# load the mnist data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

#print(y_train[:100])

# shuffle the training labels (classes) to randomize them
# random.shuffle(y_train)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#print(y_train[:100])

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# function that builds and trains a model then returns the trained model and the history
def trainNewModel(model_name="model", learning_rate=0.01, batch_size=128, momentum=0.0, epochs=10):
    print("training new model")
    print("model name = "+model_name)
    print("learning_rate = {0}".format(learning_rate))
    print("batch_size = {0}".format(batch_size))
    print("momentum = {0}".format(momentum))
    print("epochs = {0}".format(epochs))

    # building the model
    model = keras.Sequential()
    model.add(layers.Conv2D(10, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(20, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50))
    model.add(layers.Dense(num_classes, activation="softmax"))

    #model.summary()
    model._name = model_name

    # train the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), metrics=["accuracy"])

    newModelHistory = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test)).history

    # list all data in history
    # print(history.keys())
    return model, newModelHistory

# function to get the accuracy and loss plots from the model history
def getAccuracyFig(history, model, epochs):
    accuracy = currentModelDict["accuracy"]
    # summarize history for accuracy
    accuracyFigure = plt.figure(figsize=(4, 2), dpi=100)
    ax = accuracyFigure.add_subplot(111)
    ax.plot(history['accuracy'], color='r', linestyle='dashed')
    ax.plot(history['val_accuracy'], color='darkblue')
    title = model.name + ' Accuracy : ' +f'Top Accuracy = {accuracy:.3}'
    ax.set_title(title, loc='center')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.set_yticks(np.arange(0, 1, 0.05))
    ax.set_xlim([-1, epochs + 1])
    ax.legend(['Training Accuracy', 'Test Accuracy'], loc='best')
    return accuracyFigure

def getLossFig(history, model, epochs):
    loss = currentModelDict["loss"]
    # summarize history for loss
    lossFigure = plt.figure(figsize=(4, 2), dpi=100)
    ax = lossFigure.add_subplot(111)
    ax.plot(history['loss'], color='r', linestyle='dashed')
    ax.plot(history['val_loss'], color='darkblue')
    title = model.name + ' Loss (categorical cross-entropy) : ' + f'Best Loss = {loss:.3}'
    ax.set_title(title, loc='center')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_xlim([-1, epochs + 1])
    ax.legend(['Training Loss', 'Test Loss'], loc='best')
    return lossFigure

# global variables to store the currently loaded model and history for the model
currentModel = None
currentHistory = None
# current model dict will store information about the current model, like the number of epochs, learning rate,
# momentum final accuracy, final loss, etc.
currentModelDict = None

# function to save the model that is currently stored in the globals
def saveCurrentModel():
    global currentModel, currentHistory, currentModelDict
    currentModel.save(storage_path+currentModel.name+'.h5')
    pickle.dump(currentHistory, open(storage_path+currentModel.name+"_history.p", "wb"))
    pickle.dump(currentModelDict, open(storage_path+currentModel.name+"_info.p", "wb"))
    # add this model to the saved models list
    saved_models.append(currentModel.name)
    saved_models.sort()
    saved_models_spinner.config(values=saved_models)
    # after saving model successfully set currentModel and currentHistory to None and go back to home page
    currentModel, currentHistory, currentModelDict = None, None, None
    openHomePage()

# function to load a model and model history as the current model from a model's name
def loadModel(model_name):
    global currentModel, currentHistory, currentModelDict
    currentModelDict = pickle.load(open(storage_path+model_name+"_info.p", "rb"))
    currentHistory = pickle.load(open(storage_path+model_name+"_history.p", "rb"))
    currentModel = load_model(storage_path+model_name+'.h5')

# build the tkinter window for our GUI
root = Tk()
root.title('MNIST-Convnet Training App')

# fix screen size
screen_width = 1200
screen_height = 1000
root.geometry(str(screen_width)+"x"+str(screen_height))
root.resizable(width=False, height=False)   # not supporting resizable windows currently

# different size fonts to be used
LARGE_FONT= ("Verdana", 12)
MEDIUM_FONT= ("Verdana", 10)
SMALL_FONT= ("Verdana", 8)

# function to open/load the home page
def openHomePage():
    hide_all_frames()
    homePage.pack(fill="both", expand=1)

# function to open the train a new model page
def openTrainingFrame():
    hide_all_frames()
    training_frame.pack(fill="both", expand=1)

# function to open the loading screen
def openLoadingFrame():
    hide_all_frames()
    loading_frame.pack(fill="both", expand=1)

# function to open the screen after a model has been trained to ask the user if they want to save their model
# also displays the results of the trained model on this page
def openSaveFrame():
    hide_all_frames()
    Label(save_frame, text='Results', bg="white", font=LARGE_FONT).pack(pady=10)
    epochs = currentModelDict["epochs"]
    # get accuracy and loss plots for this trained model
    accuracyFig = getAccuracyFig(history=currentHistory, model=currentModel, epochs=epochs)
    # creating Tkinter canvas to display the Matplotlib figures
    canvas1 = FigureCanvasTkAgg(accuracyFig, master = save_frame)
    # placing the canvas on the Tkinter window
    canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1, pady=10, padx=100)

    # repeat for loss figure
    lossFig = getLossFig(history=currentHistory, model=currentModel, epochs=epochs)
    canvas2 = FigureCanvasTkAgg(lossFig, master = save_frame)
    # placing the canvas on the Tkinter window
    canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1, pady=10, padx=100)

    # save button and back to home button
    Button(save_frame, text="Save Model", command=saveCurrentModel, font=MEDIUM_FONT).pack(pady=10)
    Button(save_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)
    save_frame.pack(fill="both", expand=1)

# function to open the saved models page
def openSavedModelsFrame():
    hide_all_frames()
    # add a spinner with the current saved models
    # need to somehow update the spinner each time with the new saved models
    saved_models_frame.pack(fill="both", expand=1)

# function to display the results of a saved model after it is selected in the saved models screen
def openResultsPage():
    hide_all_frames()
    # get the saved models name we want to load from the spinner then load it from the saved files
    #saved_model_name = saved_models_spinner.get()
    saved_model_name = saved_models_spinner.get()
    if saved_model_name == "":
        openHomePage()
        return
    print("saved model name selected: {0}".format(saved_model_name))
    loadModel(saved_model_name)

    # first put a label for the title of this screen
    Label(results_frame, text="Results for "+str(currentModel.name), font=LARGE_FONT).pack(pady=10)

    Label(results_frame, text="Training Hyperparameters", font=MEDIUM_FONT).pack(pady=10)

    epochs = currentModelDict["epochs"]

    Label(results_frame, text="Number of epochs: "+str(epochs), font=SMALL_FONT).pack()
    Label(results_frame, text="Learning rate: "+str(currentModelDict['learning_rate']), font=SMALL_FONT).pack()
    Label(results_frame, text="Batch size: "+str(currentModelDict['batchSize']), font=SMALL_FONT).pack()
    Label(results_frame, text="Momentum: "+str(currentModelDict['momentum']), font=SMALL_FONT).pack()

    # get accuracy and loss plots for this trained model
    accuracyFig = getAccuracyFig(history=currentHistory, model=currentModel, epochs=epochs)
    # creating Tkinter canvas to display the Matplotlib figures
    canvas1 = FigureCanvasTkAgg(accuracyFig, master = results_frame)
    # placing the canvas on the Tkinter window
    canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1, pady=10, padx=100)

    # repeat for loss figure
    lossFig = getLossFig(history=currentHistory, model=currentModel, epochs=epochs)
    canvas2 = FigureCanvasTkAgg(lossFig, master = results_frame)
    # placing the canvas on the Tkinter window
    canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1, pady=10, padx=100)

    Button(results_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)
    results_frame.pack(fill="both", expand=1)

# function to open the info page
def openInfoFrame():
    hide_all_frames()
    info_frame.pack(fill="both", expand=1)

# Hide all frames fn.
def hide_all_frames():
    # if we go back to the home page clear all of the entries in the train a new model screen
    batchSizeEntry.delete(0, END)
    modelNameEntry.delete(0, END)
    learningRateEntry.delete(0, END)
    momentumEntry.delete(0, END)
    epochsEntry.delete(0, END)
    # forget the pack for all the frames. essentially clears every frame from the root window.
    homePage.pack_forget()
    training_frame.pack_forget()
    info_frame.pack_forget()
    save_frame.pack_forget()
    loading_frame.pack_forget()
    saved_models_frame.pack_forget()
    results_frame.pack_forget()
    for widget in save_frame.winfo_children():
        widget.destroy()
    for widget in results_frame.winfo_children():
        widget.destroy()

# fn. to begin training a new model from the training frame upon clicking begin training button
def trainModel():
    modelName = modelNameEntry.get()
    numEpochs = int(epochsEntry.get())
    learningRate = float(learningRateEntry.get())
    batchSize = int(batchSizeEntry.get())
    momentum = float(momentumEntry.get())
    # if this model already exists print an error to the user saying model name already exists
    if modelName in saved_models:
        Label(training_frame, "Model name already exists in your saved models!!!", bg="red", font=SMALL_FONT).pack(pady=10)
        training_frame.pack(fill="both",expand=1)
        return
    # display loading screen just before training and wait until training is complete
    openLoadingFrame()

    # now train the model with the user input hyperparameters
    model, history = trainNewModel(model_name=modelName, learning_rate=learningRate, batch_size=batchSize, momentum=momentum, epochs=numEpochs)
    # Evaluate the trained model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    # set this trained model and its history as the global variables for the current model/history
    # then go to save frame
    modelDict = {
        "model_name": modelName,
        "learning_rate": learningRate,
        "batchSize": batchSize,
        "momentum": momentum,
        "epochs": numEpochs,
        "loss": loss,
        "accuracy": accuracy
    }
    global currentModel, currentHistory, currentModelDict
    currentModel, currentHistory = model, history
    currentModelDict = modelDict
    openSaveFrame()

# function to delete the model currently selected in the spinner and then go to the home page
def deleteModelAndGoToHome():
    modelName = saved_models_spinner.get()
    if os.path.exists(storage_path+modelName+".h5"):
        os.remove(storage_path+modelName+".h5")
    if os.path.exists(storage_path+modelName+"_history.p"):
        os.remove(storage_path+modelName+"_history.p")
    if os.path.exists(storage_path+modelName+"_info.p"):
        os.remove(storage_path+modelName+"_info.p")
    saved_models.remove(modelName)
    saved_models_spinner.config(values=saved_models)
    openHomePage()

# first I make a frame for the homepage
homePage = Frame(root, width=screen_width, height=screen_height, bg="gray")
titleLabel = Label(homePage, text='Welcome to MNIST-Convnet Training', font=LARGE_FONT)
titleLabel.pack(pady=20)
Button(homePage, text="App Info", command=openInfoFrame, font=MEDIUM_FONT).pack(pady=20)
Button(homePage, text="Train New Model", command=openTrainingFrame, font=MEDIUM_FONT).pack(pady=10)
Button(homePage, text="Saved Models", command=openSavedModelsFrame, font=MEDIUM_FONT).pack(pady=20)
Button(homePage, text="Quit", command=root.destroy).pack(pady=10)  # button to quit amicably from the program

# loading frame for when a model is being trained to provide a loading screen
loading_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")
Label(loading_frame, text='Training your model...', bg="white", font=LARGE_FONT).pack(pady=10)

# save a model page
save_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")

# create frame for setting parameters to train new model
training_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")
Label(training_frame, text='Train a New Model', font=LARGE_FONT).pack(pady=10)
# create the different labels and entries for setting the hyperparameters
# model name
modelNameLabel = Label(training_frame, text="Model Name: ", font=MEDIUM_FONT)
modelNameLabel.pack(pady=20)
modelNameEntry = Entry(training_frame, width=50)
modelNameEntry.pack()
# epochs
Label(training_frame, text="Number of Epochs: ", font=MEDIUM_FONT).pack(pady=20)
epochsEntry = Entry(training_frame, width=50)
epochsEntry.pack()
# learning rate
learningRateLabel = Label(training_frame, text="Learning Rate: ", font=MEDIUM_FONT)
learningRateLabel.pack(pady=20)
learningRateEntry = Entry(training_frame, width=50)
learningRateEntry.pack()
# batch size
batchSizeLabel = Label(training_frame, text="Batch Size: ", font=MEDIUM_FONT)
batchSizeLabel.pack(pady=20)
batchSizeEntry = Entry(training_frame, width=50)
batchSizeEntry.pack()
# momentum
momentumLabel = Label(training_frame, text="Momentum: ", font=MEDIUM_FONT)
momentumLabel.pack(pady=20)
momentumEntry = Entry(training_frame, width=50)
momentumEntry.pack()
# button to begin training
Button(training_frame, text="Begin Training", command=trainModel, bg="green", font=MEDIUM_FONT).pack(pady=20)
Button(training_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)

# create frame for the saved models page
saved_models_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")
Label(saved_models_frame, text="Saved Models", font=LARGE_FONT).pack(pady=10)
saved_models_spinner = Spinbox(saved_models_frame, values=saved_models, font=LARGE_FONT, state = 'readonly')
saved_models_spinner.pack(pady=10)
Button(saved_models_frame, text="Load Selected Model", command=openResultsPage, bg="green", font=MEDIUM_FONT).pack(
    pady=20)
Button(saved_models_frame, text="Delete Selected Model", command=deleteModelAndGoToHome, bg="red", font=MEDIUM_FONT).pack(
    pady=10)
Button(saved_models_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=20)

# create a frame for the results page
results_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")

# create frame for info page
info_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")
Label(info_frame, text='App Info', font=LARGE_FONT).pack(pady=10)
app_info = "This is a GUI desktop application to give people with little to no experience working with neural " \
           "networks an easy to use interface where they can train and experiment with neural networks.\n\n" \
           "In this app we are working the MNIST dataset, which is a set of 60000 training samples, 10000 testing samples" \
           "where each sample is a 28x28 image representing a handwritten digit 0-9.\n The goal of our neural network is for " \
           "it to correctly classify samples/images as the correct digit that they represent. We hope that" \
           "after training a number of epochs on the 60000 training samples,\n that the model will perform well on the testing" \
           "samples, which are previously unseen by the model. The trained model performs well when it classifies all or most" \
           "of the testing samples correctly.\n\n" \
           "In this application you as the user are allowed to specify the following hyperparameters when training the neural network model:\n" \
           "1. Number of epochs - the number of epochs or iterations of training the model will perform on the training set. Each iteration the model learns.\n" \
           "2. Learning Rate - the learning rate affects how fast the model learns. Simply put, we are using stochastic gradient descent to optimize the model\n" \
           "each epoch, which adjust the weights of the model according to the negative gradient and the learning rate determines how much the weights are\n" \
           "adjusted at the end of each epoch/batch.\n" \
           "3. Batch Size - This also affects how quickly the model learns. Essentially the model trains over mini-batches of batch size oppose to the whole \n" \
           "60000 samples in the training set, so it adjusts the weights much more frequently and learns faster over one epoch when we have a smaller batch size.\n" \
           "Smaller batch sizes are more computationally expensive.\n" \
           "4. Momentum - momentum allows the gradients of previous batches or epochs affect the change in weights of the current epoch. The older gradients are \n" \
           "expontentially decaying, but they still have an effect on the current epoch's weight adjustments or \"learning\". This helps reduce the gradient exploding\n" \
           "and gradient vanishing problems.\n\n" \
           "After you, the user, trains a model you will be displayed the results of the training, and be given the option to save the model.\n" \
           "You are able to go back and view the training results for any of the models you save as well as the hyperparameters used in training the model.\n" \
           "You are also able to delete saved models you no longer need.\n\n" \
           "The training results displayed will include:\n" \
           "The accuracy and loss plots, which shows the accuracy and loss on the training and testing sets after each epoch of training.\n"

Label(info_frame, text=app_info, font=SMALL_FONT).pack(pady=10)
Button(info_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)

# open the home page initially
openHomePage()

root.mainloop()
