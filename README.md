# Astronomical Image Denoising (Starlet Transform vs Gaussian)

## 🌌 Description
Ce projet implémente un pipeline complet de traitement d'images astronomiques visant à restaurer des structures fines (bras de galaxies, nébuleuses) corrompues par du bruit de Poisson et Gaussien.

**Contexte :** Projet d'Analyse d'Images Master 2 Ingénierie Mathématique et Data Science (Université de Haute-Alsace).

## 🚀 Fonctionnalités
* **Acquisition Automatisée :** Récupération d'images (FITS) via `astroquery` (CDS Strasbourg).
* **Algorithme Custom :** Implémentation manuelle de la **Transformée en Ondelettes "À Trous" (Starlet)** avec stabilisation de variance (Anscombe).
* **Validation :** Calcul automatisé des métriques SSIM et PSNR.

## 📊 Résultats Clés
Sur la galaxie M51, la méthode par Ondelettes surclasse le filtre Gaussien en préservant les bras spiraux.

![Comparaison M51](comparatif_M51.jpg)

* **Gain structurel (SSIM) :** +0.31 (Passage de 0.38 à 0.69).
* **Détails :** Les étoiles faibles et la granulosité sont conservées.

## 🛠 Installation et Usage

1. Cloner le repo :
```bash
git clone [https://github.com/ton-user/nom-du-repo.git](https://github.com/ton-user/nom-du-repo.git)
