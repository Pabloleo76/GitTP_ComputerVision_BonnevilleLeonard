# %%
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as img 
import os
import cv2



# 1.d.vi : Fixer l’aléa pour pouvoir reproduire les résultats et vérifier que la cellule donne maintenant des valeurs constantes 
np.random.seed(5) # On remarque bien des résultats constant

# 1.d.i : Créer une liste, X, de 1000 points avec valeur aléatoire dans l’intervalle [0, 3] 
X = 3 * np.random.rand(1000) # Multiplier par 3 pour obtenir des valeurs dans l'intervalle [0, 3]

# 1.d.ii : Calculer la moyenne, l’écart type et la médiane de cette liste. 
mean_X = round(np.mean(X), 2) # 2 pour obtenir 2 chiffre après la virgule donc arrondir au centième
std_X = round(np.std(X), 2)
median_X = round(np.median(X), 2)
print(f'Moyenne de X: {mean_X}, Écart-type de X: {std_X}, Médiane de X: {median_X}')

# 1.d.iii : Créer une liste, X_bis, de 1000 points avec valeur aléatoire dans l’intervalle [0, 3] 
X_bis = 3 * np.random.rand(1000)

# 1.d.iv : Calculer la moyenne, l’écart type et la médiane de cette nouvelle liste. 
mean_X_bis = round(np.mean(X_bis), 2)
std_X_bis = round(np.std(X_bis), 2)
median_X_bis = round(np.median(X_bis), 2)
print(f'Moyenne de X_bis: {mean_X_bis}, Écart-type de X_bis: {std_X_bis}, Médiane de X_bis: {median_X_bis}')

# 1.d.v : Comparer les résultats de moyenne, écart type et médiane des listes X et X_bis. 
print(f'Différence de moyennes: {round(abs(mean_X - mean_X_bis),2)}, Différence d\'écart-type: {round(abs(std_X - std_X_bis),2)}, Différence de médianes: {round(abs(median_X - median_X_bis),2)}')

# 1.d.vi : Créer une liste, y, de 1000 points ayant la valeur de sin(X) auquel on ajoute un bruit gaussien aléatoire ayant une amplitude de 10% (0.1)
noise = np.random.randn(1000)
y = np.sin(X) + noise

# 1.d.vii : Visualiser y en fonction de X sous forme de graph ‘scatter’ 
plt.figure(figsize=(8,6)) 
plt.scatter(X, y) 
plt.show() 

# 1.d.ix et 1.d.x : Changer taille figure et Visualiser le bruit gaussien, noise, sous forme d’histogramme. avec bins = 50
plt.figure(figsize=(10,8))
plt.hist(noise, bins=50)
plt.show()
# 1.d.xi : Cela ressemble a une courbe en cloche



# 2 Données

# 2.b.i. Nombre d'images
nb_image = sum([len(files) for r, d, files in os.walk('data1')])  # Modifier le chemin si nécessaire
print(f'Nombre d\'images: {nb_image}')

# 2.b.ii. Format et taille des images (en prenant une image comme exemple)
exemple_image = img.imread("data1\car\Car (874).jpeg")
print(f'Format de l\'image: {exemple_image.dtype}, Taille de l\'image: {exemple_image.shape[0]} de hauteur et {exemple_image.shape[1]} de largeur')

# 2.c.i. Visualiser une des images en couleur 
image = img.imread("data1\car\Car (874).jpeg") 
plt.imshow(image)
plt.show() # permet d'afficher l'image

# 2.c.ii. Visualiser la même image en noir et blanc 
plt.imshow(image[:,:,1], cmap="gray") # les ":" signifie que cela comprend tout, donc ici on prend tout du 1er et 2eme et 1 du 3eme
plt.show()

# 2.c.iii Visualiser la même image à l’envers 
plt.imshow(image, origin="lower")
plt.show()

# 2.d.i Définir les chemins aux dossiers bike et car
bike_folder = 'data1/bike'  # Modifier le chemin si nécessaire
car_folder = 'data1/car'    # Modifier le chemin si nécessaire

# 2.d.ii Définir la taille voulue
target_size = (224, 224)

# 2.d.iii Créer une méthode
def populate_images_and_labels_lists(image_folder_path):
    images = []
    labels = []
    for filename in os.listdir(image_folder_path):
        image = cv2.imread(os.path.join(image_folder_path, filename))
        resized_image = cv2.resize(image, target_size)  # Redimensionnement de l'image
        images.append(resized_image)
        labels.append(image_folder_path.split('/')[-1])  # bike ou car basé sur le nom du dossier
    return images, labels

# 2.d.iv Créer des array numpy pour les images et les labels
images_bike, labels_bike = populate_images_and_labels_lists(bike_folder)
images_car, labels_car = populate_images_and_labels_lists(car_folder)
images = np.array(images_bike + images_car)
labels = np.array(labels_bike + labels_car)

# 2.e Preprocessing des images
images = np.array([image.flatten() for image in images])

# 2.f.i Importer la méthode adaptée
from sklearn.model_selection import train_test_split

# 2.f.ii & 2.f.iii Séparer les sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

# A quoi sert l’argument random_state ?
# L'argument random_state permet de fixer la graine (la seed) du générateur de nombres aléatoires pour que la division des données soit reproductible.

# 3 Modèles de classification 

# 3.a.i Importer la classe DecisionTreeClassifier de sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# 3.a.ii Définir l’arbre de décision
clf_tree = DecisionTreeClassifier(random_state=0)

# 3.a.iii Entraîner l’arbre de décision
clf_tree.fit(X_train, y_train)

# 3.a.iv Comment prédire le label de la première image du set de test ?
first_test_image = X_test[0].reshape(1, -1)  # Reshape nécessaire car la fonction predict attend un 2D array, reshape(1 = 1 dimension, -1 = toutes les valeurs)
predicted_label_tree = clf_tree.predict(first_test_image)
print(f'Label prédit pour la première image du set de test: {predicted_label_tree[0]}')

# 3.b Deuxième modèle de classification avec sklearn: Support Vector Machine (SVM)
from sklearn.svm import SVC

# Suivre les mêmes étapes que précédemment avec la classe SVC de sklearn.svm
clf_svm = SVC(random_state=0)
clf_svm.fit(X_train, y_train)
predicted_label_svm = clf_svm.predict(first_test_image)
print(f'Label prédit pour la première image du set de test avec SVM: {predicted_label_svm[0]}')

# 3.c.i Accuracy
from sklearn.metrics import accuracy_score

# Calculer l’accuracy du modèle 1
accuracy_tree = accuracy_score(y_test, clf_tree.predict(X_test))
print(f'Accuracy du modèle 1 (arbre de décision): {accuracy_tree}')

# Calculer l’accuracy du modèle 2
accuracy_svm = accuracy_score(y_test, clf_svm.predict(X_test))
print(f'Accuracy du modèle 2 (SVM): {accuracy_svm}')

# 3.c.ii Matrice de confusion
from sklearn.metrics import confusion_matrix

# Calculer la matrice de confusion du modèle 1
conf_matrix_tree = confusion_matrix(y_test, clf_tree.predict(X_test))
print(f'Matrice de confusion du modèle 1 (arbre de décision):\n{conf_matrix_tree}')
print(f'Vrai positif: {conf_matrix_tree[1][1]}, Vrai négatif: {conf_matrix_tree[0][0]}, Faux positif: {conf_matrix_tree[0][1]}, Faux négatif: {conf_matrix_tree[1][0]}')




# Interprétation
# Les valeurs hors de la diagonale représentent les classifications incorrectes.
# conf_matrix_tree[0][1] donne le nombre de bike classifiés comme des car
# conf_matrix_tree[1][0] donne le nombre de car classifiés comme des bike
print(f'Bike classifiés comme des car: {conf_matrix_tree[0][1]}')
print(f'Car classifiés comme des bike: {conf_matrix_tree[1][0]}')

# Calculer la matrice de confusion du modèle 2
conf_matrix_svm = confusion_matrix(y_test, clf_svm.predict(X_test))
print(f'Matrice de confusion du modèle 2 (SVM):\n{conf_matrix_svm}')
print(f'Vrai positif: {conf_matrix_svm[1][1]}, Vrai négatif: {conf_matrix_svm[0][0]}, Faux positif: {conf_matrix_svm[0][1]}, Faux négatif: {conf_matrix_svm[1][0]}')

# Bonus
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Calcul de la précision et de la spécificité (recall)
precision_tree = precision_score(y_test, clf_tree.predict(X_test), pos_label='car')
recall_tree = recall_score(y_test, clf_tree.predict(X_test), pos_label='car')
print(f'Précision du modèle 1 (arbre de décision): {precision_tree}')
print(f'Spécificité du modèle 1 (arbre de décision): {recall_tree}')

# Tracer la courbe ROC avec le modèle 1
y_score_tree = clf_tree.predict_proba(X_test)[:, 1]  # Probabilités pour la classe 'car'
falsePositiveRate_tree, truePositiveRate_tree, _ = roc_curve(y_test, y_score_tree, pos_label='car')
roc_auc_tree = auc(falsePositiveRate_tree, truePositiveRate_tree)  # auc = area under the curve 

plt.figure()
plt.plot(falsePositiveRate_tree, truePositiveRate_tree, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_tree)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positif')
plt.ylabel('Taux de vrai positif')
plt.title('Courbe ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.show()

# 
# 4 Comparaison de pipeline et fine tuning

# 4.a.a Quelle est la profondeur de l’arbre de décision ?
depth = clf_tree.get_depth()
print(f'Profondeur de l\'arbre de décision: {depth}')

# 4.a.b. Variation de l’accuracy en fonction de l’hyperparamètre max_depth
# 4.a.b.i. Créer une liste max_depth_list
max_depth_list = list(range(1, 13))

# 4.a.b.ii. Créer les listes train_accuracy et test_accuracy
train_accuracy = []
test_accuracy = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    train_accuracy.append(accuracy_score(y_train, clf.predict(X_train)))
    test_accuracy.append(accuracy_score(y_test, clf.predict(X_test)))

# 4.a.b.iii. Afficher le graph
plt.figure()
plt.plot(max_depth_list, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_list, test_accuracy, label='Test Accuracy')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.show()

# iv. Meilleure valeur de max_depth
# L'analyse du graphique permettra de déterminer la meilleure valeur de max_depth.

# 4.b Fine tuning du modèle 2 (le SVM)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# Définir la grille des hyperparamètres à rechercher
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': list(range(1, 6))
}

# Définir la recherche sur grille
grid_search = GridSearchCV(estimator=clf_svm, param_grid=param_grid, cv=4, n_jobs=-1)
grid_search.fit(X_train, y_train)  # Ajuster la recherche sur grille aux données
best_params = grid_search.best_params_  # Obtenir les meilleurs hyperparamètres
print(f'Meilleurs hyperparamètres: {best_params}')
# on applique donc nos paramètres à un nouveau modèle de SVM
best_svm = SVC(kernel=best_params['kernel'], degree=best_params['degree'], random_state=0)
best_svm.fit(X_train, y_train)
# Et évaluer le modèle sur les données de test
svm_test_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
print(f'Accuracy du modèle SVM sur les données de test: {svm_test_accuracy}')

# 4.c.b homogénéisation et de preprocessing des images, créer des array numpy val_images et val_labels
val_bike_folder = 'val/bike'  
val_car_folder = 'val/car'    
val_images_bike, val_labels_bike = populate_images_and_labels_lists(val_bike_folder)
val_images_car, val_labels_car = populate_images_and_labels_lists(val_car_folder)
val_images = np.array(val_images_bike + val_images_car)
val_labels = np.array(val_labels_bike + val_labels_car)
# Preprocessing des images
val_images = np.array([image.flatten() for image in val_images])

best_max_depth = 6
clf_best_tree = DecisionTreeClassifier(max_depth=best_max_depth, random_state=0)
clf_best_tree.fit(X_train, y_train)
val_accuracy = accuracy_score(val_labels, clf_best_tree.predict(val_images))
print(f'Accuracy de classification des données de validation : {val_accuracy}')
# %%
