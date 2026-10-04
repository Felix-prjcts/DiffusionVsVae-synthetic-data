# Diffusion vs VAE sur des données tabulaires

Projet perso pour apprendre à coder un modèle de diffusion (DDPM, Ho et al. 2020) et voir ce qu'il vaut sur des tableaux de données.

La question de départ : si on entraîne un modèle sur des données générées au lieu de vraies données, on perd combien ? Et est-ce que les données qui ressemblent le plus aux vraies sont aussi celles qui servent le plus ?

Je compare deux générateurs :

- un DDPM codé à la main en PyTorch ;
- un VAE, plus classique, comme point de comparaison.

## Les données

J'ai pris deux jeux sans variables catégorielles, pour rester simple.

- California Housing (scikit-learn) : 20 640 quartiers, 8 variables. On prédit le prix médian des logements.
- MAGIC Gamma Telescope (OpenML, id 1120) : 18 905 événements après suppression de 115 doublons, 10 variables. On prédit si c'est un rayon gamma ou un hadron (65 % / 35 %).

Deux choses à savoir sur California. `HouseAge` est plafonné à 52 ans et le prix à 500 000 $, et environ 5 % des lignes sont pile au plafond. La carte des logements a aussi deux gros pôles, Los Angeles et la baie de San Francisco. C'est pratique pour vérifier un générateur : s'il lisse tout vers la moyenne, la carte devient une tache.

## Comment je m'y prends

J'ai fixé ces règles avant d'avoir le moindre résultat.

Chaque jeu est coupé une fois pour toutes en train, validation et test (70 / 15 / 15, graine 0). Tous les réglages se font sur la validation, et le test ne sert qu'à la fin.

Avant d'entrer dans les générateurs, chaque colonne est ramenée à une loi normale avec un `QuantileTransformer` ajusté sur le train. J'ajoute un bruit minuscule juste avant, sinon toutes les valeurs au plafond se retrouvent empilées au même point. Les données générées repassent par la transformation inverse, donc tout est évalué dans les vraies unités.

Pour la cible, je fais comme TabDDPM. Pour California, le prix est généré avec les autres colonnes. Pour MAGIC, on tire la classe dans les proportions du train, et le modèle génère les variables sachant la classe.

Sur un même jeu, diffusion et VAE suivent le même schéma. Ils ont à peu près le même nombre de paramètres (environ 140 000), le même nombre d'itérations et génèrent le même nombre de lignes. Je m'autorise au plus trois réglages par générateur, choisis sur California puis appliqués tels quels à MAGIC.

Le DDPM suit le papier : calendrier de bruit linéaire (β de 1e-4 à 0,02, T = 1000), un réseau qui prédit le bruit ajouté, la perte simplifiée, puis la génération en 1000 pas. Le papier utilise un U-Net parce qu'il travaille sur des images. Ici c'est un simple MLP de 3 couches de 256. J'ai testé deux façons de donner le pas de temps au réseau : t/T directement, ou l'embedding sinusoïdal du papier.

Pour mesurer l'utilité, j'entraîne un gradient boosting et un petit réseau (2 couches de 64) sur les données générées, puis je les teste sur le vrai test : R² pour California, AUC pour MAGIC. Leurs réglages sont fixés une fois sur les vraies données et ne bougent plus.

Je fais aussi une courbe de mélange. À taille fixe, le jeu d'entraînement contient 0, 25, 50, 75 ou 100 % de vraies données, et le reste est généré. Pour la lire, je la compare aux scores obtenus avec la même part de vraies données toute seule.

Pour la ressemblance, je regarde les marginales (distance de Wasserstein) et les corrélations (Spearman). Pour la mémorisation, je vérifie que les lignes générées ne sont pas plus proches du train que du test. Si c'était le cas, le modèle recopierait.

Pour voir si ressemblance et utilité vont ensemble, je garde des sauvegardes des générateurs tout au long de l'entraînement. Ça donne une vingtaine de points par jeu au lieu de quatre.

Mon intuition de départ : le VAE devrait bien reproduire les corrélations, mais lisser les queues de distribution et les zones à plusieurs modes, comme la carte de Californie.

## Les fichiers

```
data_loading.ipynb     chargement, nettoyage, figures, découpage
baseline_model.ipynb   modèles entraînés sur les vraies données (la référence)
ddpm.ipynb             le modèle de diffusion, California puis MAGIC
figures/               les figures
```

Les dossiers `data/` et `checkpoints/` ne sont pas dans le dépôt : tout se régénère avec les notebooks.

Pour relancer, il faut Python 3.12 avec pandas, scikit-learn, scipy, matplotlib et PyTorch, puis exécuter les notebooks dans l'ordre ci-dessus.
