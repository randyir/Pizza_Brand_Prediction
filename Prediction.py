import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('Pizza.csv', header=0)
x = dataset.iloc[:,1:8].values
y = dataset['Brand'].values

model = KNeighborsClassifier(n_neighbors = 7, )
model.fit(x, y)

mois = input("Input kadar Moisture: ")
prot = input("Input kadar Protein: ")
fat = input("Input kadar Lemak: ")
ash = input("Input kadar Ash: ")
sodium = input("Input kadar Sodium: ")
carb = input("Input kadar Karbohidrat: ")
cal = input("Input kadar Kalori: ")

moisData = float(mois)
protData = float(prot)
fatData = float(fat)
ashData = float(ash)
sodiumData = float(sodium)
carbData = float(carb)
calData = float(cal)

prediction = model.predict([[moisData, protData, fatData, ashData, sodiumData, carbData, calData]])
print('Prediction : ' + prediction)