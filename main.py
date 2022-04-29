from tkinter import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import pickle
from tensorflow import keras
from keras.models import load_model
from keras import layers
import os
import sympy as sp
from matplotlib.widgets import Slider

def scos(x): return sp.N(sp.cos(x))
def ssin(x): return sp.N(sp.sin(x))

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

# custom keras callback to get the model's prediction vectors at the end of each epoch for all the training samples
# this will be used in our Grand Tour visualization method
class TestPredictionsCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.test_predictions = []
    def on_epoch_end(self, epoch, logs=None):
        # use the model at this point to get the softmax output
        # for all of the testing samples
        y_pred_test = self.model.predict(x_test)
        self.test_predictions.append(y_pred_test)
    def get_data(self):
        return self.test_predictions

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

    # use the custom callback fn. to get the test predictions at the end of each epoch
    testPredictionsCallback = TestPredictionsCallback()

    # train the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), metrics=["accuracy"])

    newModelHistory = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test), callbacks=[testPredictionsCallback]).history

    test_predictions = testPredictionsCallback.get_data()

    # list all data in history
    # print(history.keys())
    return model, newModelHistory, test_predictions

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
    title = model.name + ' Loss (categorical cross-entropy) : ' + f'Final Loss = {loss:.3}'
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

# vector to be used in the visualization of the softmax layers output for each epoch using grand tour method.
# essentially 10 vectors of length one from the origin to represent the 10 different classes/dimensions.
# representing a 10 dimensional space with 10 2-dimensional vectors
dimensions_vectors = np.array([[1,0],[scos(sp.pi/5),ssin(sp.pi/5)],[scos(2*sp.pi/5),ssin(2*sp.pi/5)],[scos(3*sp.pi/5),ssin(3*sp.pi/5)],
              [scos(4*sp.pi/5),ssin(4*sp.pi/5)],[scos(5*sp.pi/5),ssin(5*sp.pi/5)],[scos(6*sp.pi/5),ssin(6*sp.pi/5)],
              [scos(7*sp.pi/5),ssin(7*sp.pi/5)],[scos(8*sp.pi/5),ssin(8*sp.pi/5)],[scos(9*sp.pi/5),ssin(9*sp.pi/5)]])
dimensions_vectors = np.array(dimensions_vectors, dtype=float)
#print(dimensions_vectors.shape)

# colors to be used in the grand tour softmax visualization
colors = ['b','r','grey', 'g', 'c', 'm', 'y', 'indigo', 'greenyellow', 'darkorange']

# function to get a list where list is the size of the number of epochs for the model predictions
# and each index of the list has a dictionary with keys 0-9 as the true label for each sample
# where value is a list of predictions for each test sample with that true label
def getPredictionsDictionariesList(predictions):
    #print(len(predictions))
    #print(len(predictions[0]))
    prediction_dictionaries_list = []
    for i in range(len(predictions)):       # iterate thru number of epochs
        # get the points for all of the predictions and group them by true label (correct class 0-9)
        # for this epoch
        predictions_dict = {l: {"X": [], "Y": []} for l in range(10)}

        for j, prediction in enumerate(predictions[i]):     # iterate through the testing samples for this epoch
            sumX = 0
            sumY = 0
            # get the x,y for this prediction based on our 10-dimensional space
            for k in range(10):
                sumX += prediction[k] * dimensions_vectors[k][0]
                sumY += prediction[k] * dimensions_vectors[k][1]

            true_label = np.argmax(y_test[j])

            # print(true_label)
            predictions_dict[true_label]["X"].append(sumX)
            predictions_dict[true_label]["Y"].append(sumY)

        prediction_dictionaries_list.append(predictions_dict)

    #print(prediction_dictionaries_list)
    return prediction_dictionaries_list

# function that displays/returns a plot to visualize the output of the softmax layer on the testing set at each epoch. Grand Tour method.
def visualizeSoftmaxLayerWithGrandTour(predictions):
    # predictions_dict initially for first epoch
    predictions_dict_list = getPredictionsDictionariesList(predictions)

    predictions_dict = predictions_dict_list[0]

    fig = plt.figure()
    ax = fig.subplots()

    plt.subplots_adjust(bottom = 0.5)  # create some space at bottom to insert the slider button

    for i in range(10):
        ax.arrow(0.0, 0.0, dimensions_vectors[i][0], dimensions_vectors[i][1], head_width=0.0, head_length=0.0, color=colors[i], length_includes_head=True)
        # ax.plot(predictions_dict[i]["X"], predictions_dict[i]["Y"], "o", markersize=2, color=colors[0], label=str(0))

    markersize = 4

    p0, = ax.plot(predictions_dict[0]["X"], predictions_dict[0]["Y"], "o", markersize=markersize, color=colors[0], label=str(0))
    p1, = ax.plot(predictions_dict[1]["X"], predictions_dict[1]["Y"], "o", markersize=markersize, color=colors[1], label=str(1))
    p2, = ax.plot(predictions_dict[2]["X"], predictions_dict[2]["Y"], "o", markersize=markersize, color=colors[2], label=str(2))
    p3, = ax.plot(predictions_dict[3]["X"], predictions_dict[3]["Y"], "o", markersize=markersize, color=colors[3], label=str(3))
    p4, = ax.plot(predictions_dict[4]["X"], predictions_dict[4]["Y"], "o", markersize=markersize, color=colors[4], label=str(4))
    p5, = ax.plot(predictions_dict[5]["X"], predictions_dict[5]["Y"], "o", markersize=markersize, color=colors[5], label=str(5))
    p6, = ax.plot(predictions_dict[6]["X"], predictions_dict[6]["Y"], "o", markersize=markersize, color=colors[6], label=str(6))
    p7, = ax.plot(predictions_dict[7]["X"], predictions_dict[7]["Y"], "o", markersize=markersize, color=colors[7], label=str(7))
    p8, = ax.plot(predictions_dict[8]["X"], predictions_dict[8]["Y"], "o", markersize=markersize, color=colors[8], label=str(8))
    p9, = ax.plot(predictions_dict[9]["X"], predictions_dict[9]["Y"], "o", markersize=markersize, color=colors[9], label=str(9))

    # Defining the slider button
    ax_slide = plt.axes([0.22, 0.2, .6, .07])     # xpos, ypos, width, height
    # Properties of the slider
    epochs_slider = Slider(ax_slide, 'Epoch', valmin=1, valmax=len(predictions), valinit=1, valstep=1)

    # function to be called whenever the value of the slider changes. updates the plot for new epoch #
    def update(val):
        new_epoch = epochs_slider.val
        predictions_dict = predictions_dict_list[new_epoch-1]
        p0.set_xdata(predictions_dict[0]["X"])
        p0.set_ydata(predictions_dict[0]["Y"])
        p1.set_xdata(predictions_dict[1]["X"])
        p1.set_ydata(predictions_dict[1]["Y"])
        p2.set_xdata(predictions_dict[2]["X"])
        p2.set_ydata(predictions_dict[2]["Y"])
        p3.set_xdata(predictions_dict[3]["X"])
        p3.set_ydata(predictions_dict[3]["Y"])
        p4.set_xdata(predictions_dict[4]["X"])
        p4.set_ydata(predictions_dict[4]["Y"])
        p5.set_xdata(predictions_dict[5]["X"])
        p5.set_ydata(predictions_dict[5]["Y"])
        p6.set_xdata(predictions_dict[6]["X"])
        p6.set_ydata(predictions_dict[6]["Y"])
        p7.set_xdata(predictions_dict[7]["X"])
        p7.set_ydata(predictions_dict[7]["Y"])
        p8.set_xdata(predictions_dict[8]["X"])
        p8.set_ydata(predictions_dict[8]["Y"])
        p9.set_xdata(predictions_dict[9]["X"])
        p9.set_ydata(predictions_dict[9]["Y"])
        fig.canvas.draw()

    epochs_slider.on_changed(update)

    # TODO: add a play/pause button, couldnt get it to work before

    # plt.plot(0,0,'ok') #<-- plot a black point at the origin
    # plt.axis('equal')  #<-- set the axes to the same scale
    ax.set_xlim([-1.25, 1.5])  # <-- set the x axis limits
    ax.set_ylim([-1.25, 1.25])  # <-- set the y axis limits
    ax.grid(visible=False, which='major')  # <-- plot grid lines
    ax.legend(loc="best", title="Digits")
    return fig

# build the tkinter window for our GUI
root = Tk()
root.title('MNIST-Convnet Training App')

# fix screen size
screen_width = 1200
screen_height = 900
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
    if saved_model_name not in saved_models:
        openHomePage()
        return
    #print("saved model name selected: {0}".format(saved_model_name))
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

def openGrandTourPage():
    numTestSamples = int(number_of_points_entry.get().strip())
    hide_all_frames()
    # get the saved models name we want to load from the spinner then load it from the saved files
    #saved_model_name = saved_models_spinner.get()
    saved_model_name = saved_models_spinner.get()
    if saved_model_name not in saved_models:
        openHomePage()
        return
    #print("saved model name selected: {0}".format(saved_model_name))
    loadModel(saved_model_name)

    # first put a label for the title of this screen
    Label(results_frame, text="2D Grand Tour Softmax Visualization for "+str(currentModel.name), font=LARGE_FONT).pack(pady=10)

    Label(results_frame, text="Training Hyperparameters", font=MEDIUM_FONT).pack(pady=10)
    epochs = currentModelDict["epochs"]
    Label(results_frame, text="Number of epochs: "+str(epochs), font=SMALL_FONT).pack()
    Label(results_frame, text="Learning rate: "+str(currentModelDict['learning_rate']), font=SMALL_FONT).pack()
    Label(results_frame, text="Batch size: "+str(currentModelDict['batchSize']), font=SMALL_FONT).pack()
    Label(results_frame, text="Momentum: "+str(currentModelDict['momentum']), font=SMALL_FONT).pack(pady=10)
    accuracy = currentModelDict["accuracy"]
    loss = currentModelDict["loss"]
    Label(results_frame, text=f'Top Accuracy = {accuracy:.3}', font=MEDIUM_FONT).pack(pady=10)
    Label(results_frame, text=f'Final Loss = {loss:.3}', font=MEDIUM_FONT).pack(pady=10)

    test_predictions = np.array(currentModelDict["test_predictions"])
    #print(len(test_predictions))       # len of first dimension is number of epochs
    #print(len(test_predictions[0]))    # len of 2nd is number of test samples. 10000
    #print(len(test_predictions[0][0])) # len is 10 representing the softmax output for each test sample

    softmax_visualization = visualizeSoftmaxLayerWithGrandTour(test_predictions[:, :numTestSamples])   # visualize with just 1000 test samples for now to improve efficiency
    # creating Tkinter canvas to display the Matplotlib figures
    canvas1 = FigureCanvasTkAgg(softmax_visualization, master = results_frame)
    # placing the canvas on the Tkinter window
    canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1, pady=10, padx=100)

    # get the figure for the grand tour softmax visualizaiton
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
    number_of_points_entry.delete(0,END)
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
    numEpochs = int(epochsEntry.get().strip())
    learningRate = float(learningRateEntry.get().strip())
    batchSize = int(batchSizeEntry.get().strip())
    momentum = float(momentumEntry.get().strip())
    # if this model already exists print an error to the user saying model name already exists
    if modelName in saved_models:
        Label(training_frame, "Model name already exists in your saved models!!!", bg="red", font=SMALL_FONT).pack(pady=10)
        training_frame.pack(fill="both",expand=1)
        return
    # display loading screen just before training and wait until training is complete
    openLoadingFrame()

    # now train the model with the user input hyperparameters
    model, history, test_predictions = trainNewModel(model_name=modelName, learning_rate=learningRate, batch_size=batchSize, momentum=momentum, epochs=numEpochs)
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
        "accuracy": accuracy,
        "test_predictions": test_predictions
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
    global saved_models
    saved_models.remove(modelName)
    saved_models_spinner.config(values=saved_models)
    if len(saved_models) == 0:
        saved_models_spinner.config(values=[])
    openHomePage()

def exitProgramGracefully():
    root.destroy()
    exit()

# first I make a frame for the homepage
homePage = Frame(root, width=screen_width, height=screen_height, bg="gray")
titleLabel = Label(homePage, text='Welcome to MNIST-Convnet Training', font=LARGE_FONT)
titleLabel.pack(pady=20)
Button(homePage, text="App Info", command=openInfoFrame, font=MEDIUM_FONT).pack(pady=20)
Button(homePage, text="Train New Model", command=openTrainingFrame, font=MEDIUM_FONT).pack(pady=10)
Button(homePage, text="Saved Models", command=openSavedModelsFrame, font=MEDIUM_FONT).pack(pady=20)
Button(homePage, text="Quit", command=exitProgramGracefully).pack(pady=10)  # button to quit amicably from the program

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
saved_models_spinner.pack(pady=30)
Label(saved_models_frame, text="Number of Points to Display (only applicable for Grand Tour Visualization; there are 10,000 test samples. Enter number b/w 100-10,000):", font=SMALL_FONT).pack(pady=5)
number_of_points_entry = Entry(saved_models_frame, width=50)
number_of_points_entry.pack(pady=5)
Button(saved_models_frame, text="Visualize with the Grand Tour", command=openGrandTourPage, bg="light blue", font=MEDIUM_FONT).pack(
    pady=20)
Button(saved_models_frame, text="Load Selected Model", command=openResultsPage, bg="green", font=MEDIUM_FONT).pack(
    pady=10)
Button(saved_models_frame, text="Delete Selected Model", command=deleteModelAndGoToHome, bg="indian red", font=MEDIUM_FONT).pack(
    pady=20)
Button(saved_models_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)

# create a frame for the results page
results_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")

# create frame for info page
info_frame = Frame(root, width=screen_width, height=screen_height, bg="gray")
Label(info_frame, text='App Info', font=LARGE_FONT).pack(pady=10)
app_info = "This is a GUI desktop application to give people with little to no experience working with neural " \
           "networks an easy to use interface where they can train and experiment with neural networks.\n\n" \
           "In this app we are working the MNIST dataset, which is a set of 60000 training samples, 10000 testing samples " \
           "where each sample is a 28x28 image representing a handwritten digit 0-9.\n The goal of our neural network is for " \
           "it to correctly classify samples/images as the correct digit that they represent. We hope that " \
           "after training a number of epochs on the 60000 training samples,\n that the model will perform well on the testing " \
           "samples, which are previously unseen by the model. The trained model performs well when it classifies all or most " \
           "of the testing samples correctly.\n\n" \
           "In this application you as the user are allowed to specify the following hyperparameters when training the neural network model:\n\n" \
           "1. Number of epochs - the number of epochs or iterations of training the model will perform on the training set. Each iteration the model learns.\n\n" \
           "2. Learning Rate - the learning rate affects how fast the model learns. Simply put, we are using stochastic gradient descent to optimize the model\n\n" \
           "each epoch, which adjust the weights of the model according to the negative gradient and the learning rate determines how much the weights are\n" \
           "adjusted at the end of each epoch/batch.\n\n" \
           "3. Batch Size - This also affects how quickly the model learns. Essentially the model trains over mini-batches of batch size oppose to the whole \n" \
           "60000 samples in the training set, so it adjusts the weights much more frequently and learns faster over one epoch when we have a smaller batch size.\n" \
           "Smaller batch sizes are more computationally expensive.\n\n" \
           "4. Momentum - momentum allows the gradients of previous batches or epochs to affect the change in weights of the current epoch. The gradient is calculated \n" \
           "as an aggregate of the current iterations gradient and the gradients from past iterations. The older gradients are exponentially decaying,\n" \
           "but they still have an effect on the current epoch's weight adjustments or \"learning\". This helps reduce the gradient exploding\n" \
           "and gradient vanishing problems. The momentum rate set here is the rate of exponential decay of previous iterations gradients.\n\n" \
           "After you, the user, trains a model you will be displayed the results of the training, and be given the option to save the model. You are\n" \
           "able to go back and view the training results for any of the models you save as well as the hyperparameters used in training the model.\n" \
           "You are also able to delete saved models you no longer need.\n\n" \
           "The training results displayed will include:\n\n" \
           "1. The accuracy and loss plots, which shows the accuracy and loss on the training and testing sets after each epoch of training.\n\n" \
           "2. The user will also be able to visualize the learning of each model on the test set from epoch to epoch using a basic 2d visualization dubbed\n" \
           "the 'Grand Tour'; this is not the true 'Grand Tour' representation since it does not rotate about all 10 dimensions, but essentially this\n" \
           "visualization technique shows the grouping of the test samples into 10 distinct classes and shows how each test sample moves toward its correct\n" \
           "label or classification as training persists. How we do this is by converting our 10 dimensional space into 10 equal length 2d vectors and using\n" \
           "these as axes where the softmax output of each test sample is a size 10 vector with each index corresponding to one of the 10 vectors.\n\n" \
           "This application should provide you, the user, an easy way to tweak the different hyperparameters mentioned above and visualize the training results so that you\n" \
           "can find the optimal values for the hyperparameters for this model, and also observe different cases of learning caused by different hyperparameter choices."

Label(info_frame, text=app_info, font=SMALL_FONT).pack(pady=10)
Button(info_frame, text="Back to Home", command=openHomePage, font=MEDIUM_FONT).pack(pady=10)

# open the home page initially
openHomePage()

root.mainloop()
