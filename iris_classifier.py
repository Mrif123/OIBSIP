import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("C://Users//MOHAMMED RIFAIZ//Iris.csv")

X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y=df['Species'].values

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

accuracy=accuracy_score(y_test,model.predict(X_test))
print("Model Accuracy:",accuracy)

def classify_iris():
    try:
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())

        input_data=[[sepal_length,sepal_width,petal_length,petal_width]]

        prediction=model.predict(input_data)[0]
        Species=label_encoder.inverse_transform([prediction])[0]

        messagebox.showinfo("Prediction",f'The Predicted Species is:{Species}')
    except ValueError:
        messagebox.showerror("Invalid Input","Please enter valid numeric values.")

window=tk.Tk()
window.title("Iris Data Classifier")

tk.Label(window,text="Sepal Length (cm):").grid(row=0,column=0)
entry_sepal_length=tk.Entry(window)
entry_sepal_length.grid(row=0,column=1)

tk.Label(window,text="Sepal Width (cm):").grid(row=1,column=0)
entry_sepal_width=tk.Entry(window)
entry_sepal_width.grid(row=1,column=1)

tk.Label(window,text="Petal Length (cm):").grid(row=2,column=0)
entry_petal_length=tk.Entry(window)
entry_petal_length.grid(row=2,column=1)

tk.Label(window,text="Petal Width (cm):").grid(row=3,column=0)
entry_petal_width=tk.Entry(window)
entry_petal_width.grid(row=3,column=1)

classify_button=tk.Button(window,text="Classify",command=classify_iris)
classify_button.grid(row=4,column=0,columnspan=2)

window.mainloop()
