# PE_Nantes_Groupe6

Ce projet se compose de deux scripts principaux :

* `projet_math.py` : réalise une **analyse statistique** complète sur un petit ensemble de points bi-dimensionnels.
* `algo_CAH.py` : effectue une **analyse de clustering** sur des données avec ou sans réduction de dimension.

## 📁 Fichiers requis

* `algo_CAH.py` : script principal pour le clustering.
* `projet_math.py` : script pour l'analyse statistique préliminaire.
* `fonctions.py` : fichier contenant les fonctions utilisées.
* `DATA.csv` : fichier de données réelles (utilisé avec `--data final`).

## 📦 Dépendances

* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

Installe-les avec :

```bash
pip install numpy matplotlib seaborn scikit-learn
```

## 🚀 Exécution

### Pour l'analyse de clustering

Lance le script avec :

```bash
python algo_CAH.py [options]
```

#### Options disponibles

| Option     | Valeurs possibles        | Par défaut | Description                                       |
| ---------- | ------------------------ | ---------- | ------------------------------------------------- |
| `--data`   | `test`, `final`          | `test`     | Choix du jeu de données                           |
| `--k`      | entier (ex: `1`, `2`)    | `1`        | Nombre de clusters à utiliser                     |
| `--reddim` | `PCA`, `TSNE`, `compare` | `PCA`      | Méthode de réduction de dimension                 |
| `--hm`     | (flag)                   | `False`    | Affiche les heatmaps si activé                    |
| `--clean`  | (flag)                   | `False`    | Supprime les impressions intermédiaires si activé |

#### Exemples

```bash
# Exécution avec les données finales, 3 clusters, PCA, affichage des heatmaps
python algo_CAH.py --data final --k 3 --reddim PCA --hm

# Exécution avec les données de test, réduction PCA, sans affichage des étapes
python algo_CAH.py --clean
```

## 🔍 Fonctionnalités (algo\_CAH.py)

* **Réduction de dimension** avec `PCA` ou `TSNE`
* **Clustering hiérarchique** (CAH)
* **K-means** sur les données réduites
* **DBSCAN** avec visualisation des clusters
* **Heatmaps** pour l'affichage des matrices
* **Score de silhouette** (optionnel, déjà présent en commentaire)

## 🧪 Analyse statistique (projet\_math.py)

Le fichier `projet_math.py` explore les données à travers une **analyse statistique descriptive** et une **régression linéaire**. Il traite un jeu de 7 points bi-dimensionnels.

### Partie 1 : Statistiques descriptives

* Moyenne, médiane, variance, écart-type
* Valeurs min/max et étendue

### Partie 2 : Régression linéaire

* Calcul de la covariance
* Coefficients de régression `b0` et `b1`
* Affichage de la droite de régression

### Partie 3 : Analyse des résidus

* Calcul des résidus, SCE, SCT, SCR
* Coefficient de détermination R2
* Erreurs MSE et RMSE

### Partie 4 : Test d'hypothèses sur les coefficients

* Erreur standard de la pente et de l'ordonnée à l'origine
* Statistiques de test `t` et intervalles de confiance
* P-valeurs et conclusion des tests d'hypothèses

## 📄 Auteur

Projet réalisé par Pierre ZBORIL-BERTEAUD, Émile DUPLAIS, Tom-loup PÉRIVIER et Yanis RUFFLÉ dans le cadre d'un projet d'analyse de données en Python.

