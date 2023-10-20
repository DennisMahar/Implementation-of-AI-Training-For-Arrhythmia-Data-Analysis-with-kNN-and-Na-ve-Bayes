# Disabling warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Importing library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Pertama tama kita baca dulu file yang sudah diproses tadiu
df  = pd.read_csv("cleaned_data.csv")
x = df.drop(["classes"], axis=1)
y = df["classes"]

# Selanjutnya kita normalisasi data, saya menggunakan metode min max
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Lalu kita split datanya untuk training dan testing
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.3)

# kNN
# Kita buat terlebih dahulu model knn dengan k = 5 dan fit data trainingnya
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model = knn_model.fit(x_train, y_train)

# Kita cek prediksinya dengan data testing dan tes akurasi, presisi, recall, dan f1
knn_prediction = knn_model.predict(x_test)
# weighted karena datanya inbalance
knn_accuracy = accuracy_score(y_test, knn_prediction)
knn_precision = precision_score(y_test, knn_prediction, average="weighted")
knn_recall = recall_score(y_test, knn_prediction, average="weighted")
knn_f1 = f1_score(y_test, knn_prediction, average="weighted")
print(f"kNN accuracy score: {knn_accuracy}")
print(f"kNN precision score: {knn_precision}")
print(f"kNN recall score: {knn_recall}")
print(f"kNN f1 score: {knn_f1}")

# uji coba untuk k sampai 20, kita tes dengan berbagai hyperparameter lalu plot grafiknya
k = []
f1 = []

for i in range(1, 21):
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model = knn_model.fit(x_train, y_train)

    knn_prediction = knn_model.predict(x_test)
    knn_f1 = f1_score(y_test, knn_prediction, average="weighted")
    k.append(i)
    f1.append(knn_f1)

default_x_ticks = range(len(k))
plt.plot(default_x_ticks, f1)
plt.xticks(default_x_ticks, k)
plt.show()

# Naive Bayes
# Setelah saya cocokan slide dengan dokumentasi sklearn, ternyata yang digunakkan adalah Gaussian
# Pertama tama kita buat dulu modelnya dan fit dengan data training
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)

# Kita cek prediksinya dengan data testing dan tes akurasi, presisi, recall, dan f1
nb_prediction = nb_model.predict(x_test)
nb_accuracy = accuracy_score(y_test, nb_prediction)
nb_precision = precision_score(y_test, nb_prediction, average="weighted")
nb_recall = recall_score(y_test, nb_prediction, average="weighted")
nb_f1 = f1_score(y_test, nb_prediction, average="weighted")
print(f"\nNaive Bayes accuracy score: {nb_accuracy}")
print(f"Naive Bayes precision score: {nb_precision}")
print(f"Naive Bayes recall score: {nb_recall}")
print(f"Naive Bayes f1 score: {nb_f1}")
