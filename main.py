# Imports pour le "Pipeline Strasbourg"
from astroquery.hips2fits import hips2fits  # Le générateur d'images du CDS
from astropy.coordinates import SkyCoord    # Pour gérer les positions
from astropy import units as u              # Pour gérer les unités (degrés)

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pandas as pd
import warnings

# Import pour le SSIM (Indice de similarité structurelle)
from skimage.metrics import structural_similarity as ssim

# On ignore les warnings FITS mineurs
warnings.filterwarnings('ignore')

# 1. FONCTIONS MATHÉMATIQUES (Algo Ondelettes "À Trous" - Version Soft)


def b3_spline_kernel_2d():
    h = np.array([1, 4, 6, 4, 1]) / 16.0
    return np.outer(h, h)

def starlet_transform(image, n_scales=4):
    c = image
    w = []
    h_base = b3_spline_kernel_2d()
    for j in range(n_scales):
        dilation = 2**j
        size_base = 5
        size_dilated = (size_base - 1) * dilation + 1
        h_dilated = np.zeros((size_dilated, size_dilated))
        for x in range(size_base):
            for y in range(size_base):
                h_dilated[x*dilation, y*dilation] = h_base[x, y]
        
        c_next = nd.convolve(c, h_dilated, mode='mirror')
        w.append(c - c_next)
        c = c_next
    return w, c

def denoise_wavelet(image):
    
    image_safe = np.maximum(image, 0)
    
    # 1. Stabilisation Anscombe
    img_anscombe = 2.0 * np.sqrt(image_safe + 3.0/8.0)
    
    # 2. Décomposition
    n_scales = 4
    coeffs, approx = starlet_transform(img_anscombe, n_scales=n_scales)
    
    # 3. Seuillage DOUX Adaptatif (Soft Thresholding)
    coeffs_denoised = []
    
    for j, w in enumerate(coeffs):
        # Estimation du bruit (MAD)
        sigma = 1.4826 * np.median(np.abs(w - np.median(w)))
        
        # Adaptation du facteur K selon l'échelle
        k_factor = 2.8 - (0.5 * j)
        if k_factor < 0.5: k_factor = 0.5 
            
        threshold = k_factor * sigma 
        
        # Soft Thresholding
        w_clean = np.sign(w) * np.maximum(0, np.abs(w) - threshold)
        coeffs_denoised.append(w_clean)
    
    # 4. Reconstruction
    rec = np.zeros_like(approx)
    weights = [1.0] * n_scales 
    
    for j in range(n_scales):
        rec += weights[j] * coeffs_denoised[j]
    
    rec += approx
    
    # 5. Inverse Anscombe
    return (rec / 2.0)**2 - 3.0/8.0

# 2. FONCTION CONCURRENTE (Le classique Gaussien)

def denoise_gaussian(image, sigma=1.0):
    return nd.gaussian_filter(image, sigma=sigma)

# 3. OUTILS DATA SCIENCE & MÉTRIQUES

def calculer_psnr(original, test):
    mse = np.mean((original - test) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(np.max(original) / np.sqrt(mse))

def calculer_ssim(original, test):
    # data_range=1.0 car les images sont normalisées entre 0 et 1
    return ssim(original, test, data_range=1.0)

def ajouter_bruit(image, niveau=0.1):
    sigma_bruit = niveau * np.max(image) 
    noise = np.random.normal(0, sigma_bruit, image.shape)
    return image + noise

def get_dataset_list():
   
    return [
        "M51", "M101", "M31", 
        "M82", "M63", "M104", "M64", "M74",
        "M83", "M106"
    ]

# 4. PIPELINE DE TRAITEMENT

def run_batch_processing():
    dossier_raw = "dataset_raw"
    dossier_out = "dataset_clean"
    dossier_rapport = "rapport_images"
    
    for d in [dossier_raw, dossier_out, dossier_rapport]:
        if not os.path.exists(d): os.makedirs(d)
    
    objets = get_dataset_list()
    resultats = [] 
    
    print(f"\n=== DÉMARRAGE DU BENCHMARK FINAL (10 Objets) ===", flush=True)

    for i, obj_name in enumerate(objets):
        filename = f"{obj_name}.fits"
        path_in = os.path.join(dossier_raw, filename)
        
        print(f"\n--- Image {i+1}/{len(objets)} : {obj_name} ---", flush=True)
        
        #  A. Téléchargement (Via CDS Strasbourg / Hips2Fits) ---
        if not os.path.exists(path_in):
            print(f"   [1/6] Connexion CDS (Strasbourg)...", flush=True)
            try:
                coord = SkyCoord.from_name(obj_name)
                hdu = hips2fits.query(
                    hips='CDS/P/DSS2/red',
                    ra=coord.ra,
                    dec=coord.dec,
                    width=300,
                    height=300,
                    fov=0.15 * u.deg,
                    projection="TAN"
                )
                if hdu is not None:
                    hdu.writeto(path_in, overwrite=True)
                else:
                    print(f"   [!] Image vide renvoyée pour {obj_name}")
                    continue
            except Exception as e:
                print(f"   [!] Erreur CDS : {e}")
                continue
        
        #  B. Chargement 
        try:
            with fits.open(path_in) as hdul:
                data = hdul[0].data.astype(float)
        except Exception: continue

        # Normalisation Min-Max [0, 1]
        img_ref = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        #  C. Simulation Bruit
        print(f"   [2/6] Ajout bruit...", flush=True)
        img_noisy = ajouter_bruit(img_ref, niveau=0.2)
        psnr_in = calculer_psnr(img_ref, img_noisy)
        ssim_in = calculer_ssim(img_ref, img_noisy)
        
        #  D. Traitement 1 : GAUSSIEN 
        print(f"   [3/6] Filtre Gaussien...", flush=True)
        img_gauss = denoise_gaussian(img_noisy, sigma=1.0)
        psnr_gauss = calculer_psnr(img_ref, img_gauss)
        ssim_gauss = calculer_ssim(img_ref, img_gauss)
        
        #  E. Traitement 2 : ONDELETTES (Soft) 
        print(f"   [4/6] Filtre Ondelettes (Soft)...", flush=True)
        img_wav = denoise_wavelet(img_noisy)
        psnr_wav = calculer_psnr(img_ref, img_wav)
        ssim_wav = calculer_ssim(img_ref, img_wav)
        
        #  F. Résultats 
        print(f"   [5/6] Ondelettes: {psnr_wav:.2f} dB (SSIM {ssim_wav:.3f})")
        print(f"         Gaussien:   {psnr_gauss:.2f} dB (SSIM {ssim_gauss:.3f})")
        
        resultats.append({
            "Objet": obj_name, 
            "PSNR_Gaussien": psnr_gauss,
            "SSIM_Gaussien": ssim_gauss,
            "PSNR_Ondelettes": psnr_wav, 
            "SSIM_Ondelettes": ssim_wav
        })
        
        #  G. Visualisation
        print(f"   [6/6] Sauvegarde image...", flush=True)
        
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        
        # Originale
        axes[0].imshow(img_ref, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"{obj_name}\nOriginale")
        axes[0].axis('off')

        # Bruitée
        axes[1].imshow(img_noisy, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Bruitée\nPSNR: {psnr_in:.1f}")
        axes[1].axis('off')
        
        # Gaussien
        axes[2].imshow(img_gauss, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Gaussien\nSSIM: {ssim_gauss:.2f}")
        axes[2].axis('off')
        
        # Ondelettes
        axes[3].imshow(img_wav, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f"Ondelettes (Soft)\nSSIM: {ssim_wav:.2f}")
        axes[3].axis('off')
        
        # Résidus
        axes[4].imshow(img_noisy - img_wav, cmap='gray')
        axes[4].set_title(f"Résidus\n(Ce qui a été retiré)")
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_rapport, f"comparatif_{obj_name}.png"))
        plt.close(fig) # Important pour libérer la mémoire

    #  H. Synthèse Finale 
    if len(resultats) > 0:
        print(f"\n=== RÉSULTATS FINAUX (SUR 10 OBJETS) ===")
        df = pd.DataFrame(resultats)
        
        df["Diff_PSNR"] = df["PSNR_Ondelettes"] - df["PSNR_Gaussien"]
        df["Diff_SSIM"] = df["SSIM_Ondelettes"] - df["SSIM_Gaussien"]
        
        # Affichage des 10 meilleurs gains SSIM
        print("\n--- Top 10 des objets où les Ondelettes gagnent (Trié par gain SSIM) ---")
        print(df.sort_values(by="Diff_SSIM", ascending=False)[["Objet", "Diff_PSNR", "Diff_SSIM"]].head(10).round(3))
        
        moy_psnr = df["Diff_PSNR"].mean()
        moy_ssim = df["Diff_SSIM"].mean()
        
        print("-" * 50)
        print(f"GAIN MOYEN GLOBAL vs GAUSSIEN :")
        print(f"PSNR : {moy_psnr:+.2f} dB")
        print(f"SSIM : {moy_ssim:+.3f}")
        print("-" * 50)
        
        df.round(4).to_csv("resultats_finaux_10_objets.csv", index=False)
        print("Fichier CSV 'resultats_finaux_10_objets.csv' généré avec succès.")

if __name__ == "__main__":
    run_batch_processing()
