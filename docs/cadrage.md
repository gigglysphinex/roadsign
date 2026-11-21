# Note de cadrage – RoadSign (sujet C)
Équipe : [noms] · Module : Vision par ordinateur avancée · Revue de lancement, semaine 1

## 1. Besoin reformulé
Un équipementier ADAS veut, à partir d'une caméra embarquée, **détecter les panneaux de signalisation, les reconnaître (43 classes) et fournir pour chaque image la liste des panneaux avec position, classe et score**. Le système doit tourner **en temps réel sur un matériel embarqué** : le modèle doit être léger et rapide, la précision seule ne suffit pas.

## 2. Données et rôles
| Jeu | Rôle | Contenu | Remarque |
|---|---|---|---|
| GTSDB | détection / localisation | 900 images : 600 entraînement, 300 test officiel | panneaux parfois très petits ; peu d'images |
| GTSRB | classification de vignettes | plus de 50 000 vignettes, 43 classes | séries de vignettes d'un même panneau physique |

Analyse exploratoire à produire cette semaine (chiffres à remplir) : répartition des 43 classes dans chaque jeu, distribution des tailles de boîtes, rapport d'anomalies (boîtes hors image, dégénérées, dupliquées, classes invalides, images sans annotation).

## 3. Protocole d'évaluation (test gelé)
- **Test GTSDB (300 images) et test officiel GTSRB : gelés, utilisés une seule fois** pour le rapport final. Aucun choix de modèle ni de réglage dessus.
- **Validation** tirée des données d'entraînement uniquement.
  - GTSDB : découpage **par blocs d'images consécutives**, car des images voisines sont quasi identiques (fuite sinon).
  - GTSRB : découpage **par séquence** : toutes les vignettes d'un même panneau physique restent du même côté.
- Découpages sauvegardés en JSON, graine fixée, rejouables (ENF6).
- Mêmes conditions de mesure pour tous les modèles : même découpage, mêmes seuils IoU, même matériel déclaré.

## 4. Architecture visée
1. **Baseline classique** (chapitre 1) : segmentation couleur HSV + filtrage de forme pour la détection ; HOG + SVM linéaire pour la classification. Mesurée sur la validation.
2. **Classification (EF2)** : CNN conçu par l'équipe, entraîné depuis zéro, comparé à un modèle pré-entraîné (extraction de caractéristiques puis fine-tuning). Dimensions et paramètres calculés à la main puis vérifiés par le code.
3. **Détection (EF3)** : un détecteur à un étage (YOLO) et un à deux étages (Faster R-CNN), même protocole.
4. **Post-traitement (EF4)** : IoU et NMS implémentés par l'équipe, seuils choisis sur la validation, sortie JSON par image.
5. **Démonstrateur (EF5)** : image d'entrée, image annotée, JSON.

## 5. Améliorations proposées
- **A1 – Détecter puis classer** : un détecteur « panneau » (une seule classe) suivi du CNN GTSRB sur la vignette. Comparé au détecteur à 43 classes.
- **A2 – Données synthétiques par copier-coller** : vignettes GTSRB d'entraînement collées sur des fonds d'entraînement GTSDB (taille, lumière et flou aléatoires, sans recouvrir une annotation réelle) pour compenser les 600 images. Évalué uniquement sur la validation réelle.
- **A3 – Benchmark embarqué** : export ONNX et quantification INT8 ; mesure ms/image, FPS, nombre de paramètres et Mo, sur le matériel déclaré.

## 6. Objectifs chiffrés (provisoires, à calibrer après la baseline)
| ID | Mesure | Objectif provisoire |
|---|---|---|
| ENF1 | mAP@0.5 et rappel des panneaux (détection) ; F1 macro (classification) | à fixer après la baseline |
| ENF2 | ms/image et FPS, matériel et conditions déclarés | à fixer après la première mesure |
| ENF3 | paramètres et taille du fichier (Mo) | à fixer après la première mesure |
| ENF4 | résultats par sous-ensemble : taille de panneau, éclairage, densité | rapport par sous-ensemble |

## 7. Risques
| Risque | Parade |
|---|---|
| Seulement 600 images de détection | A2, augmentation, modèles pré-entraînés |
| Classes rares sans exemples en validation | rapport par classe, regroupement documenté si nécessaire |
| Fuite par images ou vignettes voisines | découpages par blocs et par séquence |
| Faux gain dû aux images synthétiques | validation sur réel uniquement, ablation avec et sans A2 |
| Temps de calcul GPU limité (Colab/Kaggle) | petits modèles, seeds fixes, points de sauvegarde |

## 8. Répartition des rôles
[Membre 1 : données et baseline] · [Membre 2 : classification et CNN] · [Membre 3 : détection, NMS et démo]. Chacun doit pouvoir expliquer toute la chaîne.
