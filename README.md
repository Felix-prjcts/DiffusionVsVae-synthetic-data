# Données tabulaires synthétiques : un modèle de diffusion face à un VAE

Ce projet part d'une question simple. Si l'on entraîne un modèle sur des données générées plutôt que sur des données réelles, que perd-on ? Et les métriques qui mesurent la ressemblance entre données synthétiques et réelles disent-elles quelque chose de cette perte ?

Pour y répondre, je compare deux générateurs sur deux jeux de données tabulaires :

- un modèle de diffusion (DDPM, Ho, Jain et Abbeel, 2020) adapté aux tableaux, que j'implémente moi-même ;
- un VAE conditionnel, qui sert de point de comparaison.

Ce qui m'intéresse le plus, c'est de savoir si le générateur dont les données sont les plus utiles est aussi celui dont les données ressemblent le plus aux vraies. Rien ne garantit que ce soit le cas.

*Projet en cours : les résultats seront ajoutés au fil de l'avancement.*

## Données

| | California Housing | MAGIC Gamma Telescope |
|---|---|---|
| Domaine | immobilier (recensement américain de 1990) | astrophysique (événements simulés d'un télescope Tcherenkov) |
| Tâche | régression : valeur médiane des logements d'un quartier | classification : gamma ou hadron |
| Lignes | 20 640 | 18 905 (19 020 avant dédoublonnage) |
| Variables | 8, toutes continues | 10, toutes continues |
| Source | scikit-learn | OpenML, identifiant 1120 |

J'ai choisi deux jeux sans variables catégorielles. C'est un choix de périmètre pour tenir dans le temps imparti, pas une propriété des méthodes.

Quelques particularités comptent pour la suite.

- **Deux colonnes plafonnées dans California.** `HouseAge` est plafonné à 52 ans (6,2 % des lignes) et la cible à 5,00001, soit 500 000 $ (4,7 %). Ces valeurs veulent dire « au moins », mais je les garde telles quelles : un bon générateur doit reproduire ces pics.
- **Une géographie bimodale.** La latitude et la longitude de California se concentrent autour de Los Angeles et de la baie de San Francisco. La carte des logements sert de test visuel : un générateur qui lisse vers la moyenne la transforme en tache floue.
- **MAGIC.** Le jeu contenait 115 doublons exacts, retirés avant le découpage. Ses variables ont des queues lourdes, et les classes sont déséquilibrées (65 % gamma, 35 % hadron).

## Protocole

J'ai fixé les règles ci-dessous avant de voir le moindre résultat. Si je dois m'en écarter, je l'indiquerai dans la section Limites, avec la raison.

**Découpage.** Chaque jeu est coupé une seule fois en train (70 %), validation (15 %) et test (15 %), avec une graine fixe. Le découpage est stratifié sur la classe pour MAGIC, et sur les déciles de la cible pour California. Le train sert à entraîner les générateurs et les modèles, la validation à tous les réglages. Le test n'est ouvert qu'une fois, pour les chiffres finaux.

**Prétraitement des générateurs.** Les deux générateurs voient exactement les mêmes données. Chaque colonne est ramenée vers une loi normale par une transformation par quantiles, ajustée sur le train uniquement. J'ajoute avant la transformation un bruit négligeable (de l'ordre de 10⁻⁶ écart-type) pour départager les valeurs répétées. Sans lui, toutes les valeurs plafonnées seraient envoyées au même point, à l'extrémité de la gaussienne. Les données générées repassent par la transformation inverse : toute l'évaluation se fait en unités d'origine.

**Conditionnement.** Les deux générateurs produisent des lignes conditionnellement à la cible. Pour fabriquer un jeu synthétique, je tire d'abord les valeurs de la cible dans leur distribution observée sur le train, puis je génère les variables explicatives correspondantes.

**Équité de la comparaison.** Les deux générateurs :

- ont un nombre de paramètres du même ordre ;
- reçoivent le même nombre d'itérations d'entraînement ;
- produisent le même nombre de lignes.

Chacun a droit à trois configurations au plus, choisies sur la validation de California. Ces réglages sont ensuite appliqués sans retouche à MAGIC, qui joue le rôle de test de robustesse.

**Modèles d'évaluation.** J'utilise un gradient boosting (`HistGradientBoosting` de scikit-learn) et un petit réseau de neurones (deux couches cachées de 64 neurones). Leurs hyperparamètres sont fixés une fois pour toutes sur données réelles. Ils ne changent plus, quelle que soit l'origine des données d'entraînement.

**Mesures.**

- *Utilité.* Les modèles d'évaluation sont entraînés sur les données synthétiques et testés sur le test réel (protocole « train on synthetic, test on real »). La référence est le même modèle entraîné sur le train réel. J'utilise le R² pour California et l'AUC pour MAGIC.
- *Courbe de mélange.* La taille du jeu d'entraînement reste celle du train. La part de données réelles vaut 0, 25, 50, 75 ou 100 %, et le reste est synthétique. Pour lire cette courbe, je la compare aux scores obtenus avec les mêmes parts de réel seul, sans complément. Le générateur ayant vu tout le train, la question est combien de réel on peut remplacer, et non comment augmenter un petit jeu de données.
- *Ressemblance.* Je compare les marginales (distance de Wasserstein, colonne par colonne) et la structure de dépendance (écart entre matrices de corrélation de Spearman).
- *Mémorisation.* Pour chaque ligne synthétique, je mesure la distance à la ligne la plus proche du train et à la ligne la plus proche du test. Des lignes synthétiques nettement plus proches du train que du test signaleraient un générateur qui recopie, et donc des scores d'utilité trompeurs.
- *Lien entre ressemblance et utilité.* Avec deux générateurs, la question n'aurait que quatre points. Je sauvegarde donc chaque générateur à plusieurs moments de son entraînement, ce qui donne une vingtaine de couples (ressemblance, utilité) par jeu. J'en regarde la corrélation de rang.

**Hypothèse de départ.** Je m'attends à ce que le VAE reproduise correctement les corrélations, mais lisse les queues de distribution et les zones à plusieurs modes, comme la carte de Californie. Il pourrait alors paraître fidèle selon les métriques de corrélation tout en étant moins utile que le modèle de diffusion.

## Avancement

- [x] Préparation des données
- [ ] Modèles de référence sur données réelles
- [ ] Modèle de diffusion
- [ ] VAE conditionnel
- [ ] Évaluation
- [ ] Résultats et limites

## Organisation du dépôt

```
data_loading.ipynb   chargement, nettoyage, figures, découpage train / val / test
baseline.ipynb       modèles de référence entraînés sur données réelles
figures/             figures de référence sur les données réelles
data/                non versionné, régénéré par data_loading.ipynb
```

Pour reproduire : Python 3.12 avec pandas, scikit-learn, matplotlib et PyTorch, puis exécuter `data_loading.ipynb`, qui télécharge les deux jeux.
