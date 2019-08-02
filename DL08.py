import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import csv
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# An unemployment dataset from the Phoenix City website is used to train our classifiers
#Dataset is a excel file
#Head is called on the train file to only get the top few lines of data
train = pd.read_csv("phoenixazunemployment.csv")
train.head()

# Setting the year and unemployment rate column as the feature columns
feature_cols = ['year', 'unemployment_rate']


# X is set to train on feature columns
X = train.loc[:, feature_cols]

# y is set to year which is what out data set will try to predict
y = train.year

# The test size is set to 40% of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# Using four different types of classifiers, decision tree classifier, Kneighbors classifier, logitistc regression classifier, and MLP classifier
decision_tree = tree.DecisionTreeClassifier()
kneighbors = KNeighborsClassifier()
logistic_regression = LogisticRegression()
ann = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# Fitting the training data to all four classifiers
decision_tree.fit(X_train, y_train)
kneighbors.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
ann.fit(X_train, y_train)

# Using all four classifiers for prediction
decision_tree_prediction = decision_tree.predict(X_test)
kneighbors_prediction = kneighbors.predict(X_test)
logistic_regresion_prediction = logistic_regression.predict(X_test)
ann_prediction = ann.predict(X_test)


# Creates a tkinter window titled 'window' with a white background
window = Tk()
window.title("Phoenix Dataset")
window.configure(background="black")

# Breaks the tkinter window into a top frame and bottom frame
topFrame = LabelFrame(window, background='black')
topFrame.pack()
bottomFrame = Frame(window, background='black')
bottomFrame.pack(side=BOTTOM)

# Displays an image for the menu
# Displays "Unemployment Rate" on the bottom frame of the main window
menu_font = Label(bottomFrame, text='Unemployment Rate Dataset', background='Black', foreground='White', font=44)
menu_font.pack()
image_background = PhotoImage(file="menu_image.png")
menu_image = Label(window, image=image_background, background='black')
menu_image.pack()


# Text with program name information
name_text = "Project Name: City of Phoenix Unemployment Rate" \
            "\nBy: Samael Newgate" \
            "\nClass: CSC44 - Deep Learning - SU19158" \
            "\nAssignment: TensorFlow and scikit learn"

# Text with program description text
description_text = "Project Description: " \
                   "\nTakes the years and their unemployment rates " \
                   "\nfrom a city of Phoenix unemployment dataset. " \
                   "\nIt then predicts what year the unemployment rates belong to" \
                   "\n\nPurpose: " \
                   "\nThis is to see if dates have a correlation with unemployment rates  "

# Text with program guide text
guide_text = "Button Functionality Legend:" \
             "\nName - Displays name information" \
             "\nDescription - Displays a description of the program" \
             "\nGuide - Displays a program using guide" \
             "\nDataset - Displays the dataset" \
             "\nResults - Displays the accuracy of the prediction. (1.0 = 100%, 0.50 = 50%, 0.0=0%)" \
             "\nExit - Closes the program"

# Text that displays the accuracy of all four of the classifiers
results_text ="The accuracy score for the DT Classifier is:", accuracy_score(y_test, decision_tree_prediction),\
              "\n\nThe accuracy score for the KN Classifier is:", accuracy_score(y_test, kneighbors_prediction),\
              "\n\nThe accuracy score for the LR Classifier is:", accuracy_score(y_test, logistic_regresion_prediction),\
              "\n\nThe accuracy score for the ANN Classifier is:", accuracy_score(y_test, ann_prediction)




# Creates another window for the dataset
# This window is structures to become a table to display all our rows and columns from the csv file
def dataset_window():
    data_window = Tk()
    data_window.title("Unempoyment Rate")
    width = 500
    height = 400
    screen_width = data_window.winfo_screenwidth()
    screen_height = data_window.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    data_window.geometry("%dx%d+%d+%d" % (width, height, x, y))
    data_window.resizable(0, 0)

    TableMargin = Frame(data_window, width=600)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin,
                        columns=("city", "state", "year", "month", "period", "date", "unemployment_rate"),
                        height=400, selectmode="extended", yscrollcommand=scrollbary.set,
                        xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill='x')
    tree.heading('city', text="city", anchor=W)
    tree.heading('state', text="state", anchor=W)
    tree.heading('year', text="year", anchor=W)
    tree.heading('month', text="month", anchor=W)
    tree.heading('period', text="period", anchor=W)
    tree.heading('date', text="date", anchor=W)
    tree.heading('unemployment_rate', text="unemployment_rate", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=200)
    tree.column('#2', stretch=NO, minwidth=0, width=200)
    tree.column('#3', stretch=NO, minwidth=0, width=200)
    tree.column('#4', stretch=NO, minwidth=0, width=200)
    tree.column('#5', stretch=NO, minwidth=0, width=200)
    tree.column('#6', stretch=NO, minwidth=0, width=200)
    tree.column('#7', stretch=NO, minwidth=0, width=200)
    tree.pack()

    # Opens up the csv file and creates rows and columns of selected labels
    with open('display_unemployment.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            city = row['city']
            state = row['state']
            year = row['year']
            month = row['month']
            period = row['period']
            date = row['date']
            unemployment_rate = row["unemployment_rate"]
            tree.insert("", 0, values=(city, state, year, month, period, date, unemployment_rate))

    # Runs the data window on a loop until closed out
    data_window.mainloop()


# Window that shows  name information text
def name_window():
    toplevel = Toplevel()
    label1 = Label(toplevel, text=name_text, height=0, width=50)
    label1.pack()

#Window that shows description text
def description_window():
    toplevel = Toplevel()
    label2 = Label(toplevel, text=description_text, height=0, width=70)
    label2.pack()

#Window that shows guide text,
def guide_window():
    toplevel = Toplevel()
    label3 = Label(toplevel, text=guide_text, height=0, width=70)
    label3.pack()

def results_window():
    toplevel = Toplevel()
    label4 = Label(toplevel, text=results_text, height=0, width=70)
    label4.pack()


# Name button set to the top frame, name window is bound to this button
button1 = Button(topFrame, text='Name', padx=5, pady=5, command = name_window)
button1.pack(side=LEFT)

# Description button set to the top frame, description window is bound to this button
button1 = Button(topFrame, text='Description', padx=5, pady=5, command = description_window)
button1.pack(side=LEFT)

# Guide button set to the top frame, guide window is bound to this button
button1 = Button(topFrame, text='Guide', padx=5, pady=5, command = guide_window)
button1.pack(side=LEFT)

# Dataset button set to the top frame, this runs the dataset window which displays the dataset
button4 = Button(topFrame, text='Dataset', background='green', padx=5, pady=5, command=dataset_window)
button4.pack(side=LEFT)

# Results window, set to the top frame, this runs the result window which displays the result of the classifiers
button5 = Button(topFrame, text='Results', background='blue', padx=5, pady=5,command=results_window)
button5.pack(side=LEFT)

# Exit button set to the top frame, window.destroy is bound to this button which destroys all active windows
button6 = Button(topFrame, text='Exit', background='red', padx=5, pady=5,command=window.destroy)
button6.pack(side=LEFT)

# Runs a loop for the main window until it is closed
window.mainloop()





