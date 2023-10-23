# TP1 : Pipeline complète de classification d’images 

## 1- Mise en place de l’environnement de code 

### 1.d.ii. Calculer la moyenne, l’écart type et la médiane de cette liste.
Moyenne de X: 1.49
Écart-type de X: 0.86 
Médiane de X: 1.49

### 1.d.iv. Calculer la moyenne, l’écart type et la médiane de cette nouvelle liste. 
Moyenne de X_bis: 1.56
Écart-type de X_bis: 0.85
Médiane de X_bis: 1.57

### 1.d.v. Comparer les résultats de moyenne, écart type et médiane des listes X et X_bis.
Différence de moyennes: 0.07, Différence d'écart-type: 0.01, Différence de médianes: 0.08

### 1.d.viii. Visualiser y en fonction de X sous forme de graph ‘scatter’ 
On aperçois un nuage de point majoritairement situé ente -1 et 2 

### 1.d.x. Visualiser le bruit gaussien, noise, sous forme d’histogramme.
On aperçois une courbe en cloche

## 2. Données

### 2.b. Quel est le format et la taille des images ?  
Nombre d'images: 916
Format de l'image: uint8, Taille de l'image: 183 de hauteur et 275 de largeur

### 2.f.iii. A quoi sert l’argument random_state ? 
L'argument random_state permet de fixer la graine (la seed) du générateur de nombres aléatoires pour que la division des données soit reproductible.

## 3. Données

### 3.a.iv. Comment prédire le label de la première image du set de test ?
Label prédit pour la première image du set de test: bike
Label prédit pour la première image du set de test avec SVM: bike

### 3.c.i. Calculer l’accuracy du modèle 1 et 2
Accuracy du modèle 1 (arbre de décision): 0.9293478260869565
Accuracy du modèle 2 (SVM): 0.9402173913043478

### 3.c.ii. Matrice de confusion
Matrice de confusion du modèle 1 (arbre de décision):
[[82  5]
 [ 8 89]]
Vrai positif: 89, Vrai négatif: 82, Faux positif: 5, Faux négatif: 8

Bike classifiés comme des car: 5
Car classifiés comme des bike: 8

Matrice de confusion du modèle 2 (SVM):
[[82  5]
 [ 6 91]]
Vrai positif: 91, Vrai négatif: 82, Faux positif: 5, Faux négatif: 6

### 3.c.iii. Bonus : Calculer la précision, spécificité (recall) et tracer la courbe ROC avec le modèle 1 
Précision du modèle 1 (arbre de décision): 0.9468085106382979
Spécificité du modèle 1 (arbre de décision): 0.9175257731958762

On peut voir un taux de vrai positif élevé pour un taux de faux positif faible tout en étant au dessus de l'auc (area = 0.93) donc les résultats sont bon


## 4. Comparaison de pipeline et fine tuning 

### 4.a.a. Quelle est la profondeur de l’arbre de décision ? 
Profondeur de l'arbre de décision: 7

### 4.a.b.iv. Quelle est la meilleure valeur de max_depth à choisir ? Pourquoi ? 
On observe que la profondeur "max-depth" la plus interessante est 6 car on dela, la courbe diminue.

### 4.b. Choisir quelle est la meilleure valeur pour l’hyperparamètre degree et pour l’hyperparamètre kernel. 
Meilleurs hyperparamètres: {'degree': 1, 'kernel': 'rbf'}

### 4.c.c. calculer l’accuracy de classification des données de validation. 
Accuracy du modèle SVM sur les données de test: 0.9402173913043478
Accuracy de classification des données de validation : 0.8571428571428571
On remarque que l'accuracy diminue car les don

### 4.c.d. et 4.c.e. Que peut-on dire de cette valeur et comment l'expliquer ? 
On remarque que l'accuracy diminue avec les données de validation
Cela veut donc dire que le modèle test a probablement fait de l'overfitting (trops appris, ou que les données soient trops spécifiques) ou que les hyperparamètres ne sont pas optimisé

### 4.d.c. et 4.d.d. Quelle est la dimension d’une grey_image après la première ligne ? Quelle est la dimension d’une grey_image après la deuxième ligne ? Sur quel paramètre joue-ton ?  
Après la première ligne : la dimension de grey_image est (X, X), c'est une image en noir et blanc
Après la deuxième ligne : la dimension de grey_image est (X, X, 3), c'est une image en RGB
On joue sur la representation de la couleur de l'image, en RGB on a 3 couleurs, en noir et blanc on a 1 une seule "couleur"

### 4.d.f. Comment a évolué l’accuracy de classification sur les données de validation ? 
Accuracy avec augmentation des données : 0.8781818181818182
On remarque que l'accuracy avec l'augmentation des donnée a légèrement augmenté. c'est du au faîtes qu'on diversifie les informations donc on diminue le risque d'overfitting car si jamais les images de voitures étaient toutes bleu et celle de moto etaient rouge, ils aurait mal assimilé les éléments.


### 4.e. Plus de fine-tuning !
Meilleurs paramètres pour l'arbre de décision : {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}
Accuracy de l'arbre de décision avec les meilleurs paramètres sur les tests : 0.9347826086956522

### 4.f. Plus de Modèles !

Test avec forêt aléatoire :
Accuracy de test de la forêt aléatoire : 0.9510869565217391
Accuracy de validation de la forêt aléatoire : 0.14285714285714285
Accuracy avec donnée augmenté de la forêt aléatoire : 0.9163636363636364

Test avec forêt K-voisins :
Accuracy de test de K-NN : 0.9510869565217391
Accuracy de validation de K-NN : 0.8095238095238095
Accuracy avec données augmentées de K-NN : 0.8218181818181818

Test avec réseau de neurones :
Accuracy de test du réseau de neurones : 0.8586956521739131
Accuracy de validation du réseau de neurones : 0.42857142857142855
Accuracy avec données augmentées du réseau de neurones : 0.8654545454545455

on observe que l'accuracy diminue plus pour les validation que le reste et est globalement plus haute pour les test que lorsqu'on a une augmentation de données.
Sauf pour le réseau de neurones où l'augmentation de donnée entraine une hausse de l'accuracy, on peut donc supposer que les reseaux de neurones sont une solution interessante et qui a su eviter l'overfitting sans necessiter une augmentation des données.