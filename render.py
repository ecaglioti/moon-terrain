"""
render.py
=========

Dato un heightfield e la posizione del sole (azimut/altezza), calcola
un'immagine di luminosita' usando:

  1. Shading Lambertiano: la luce riflessa e' proporzionale al coseno
     dell'angolo tra la normale alla superficie e la direzione del sole
     (I = max(0, N . L)).
  2. Ombre portate: un punto non riceve luce diretta se, lungo la
     direzione del sole, un rilievo del terreno blocca la vista del sole
     (calcolato con ray-marching).
  3. Una piccola componente di luce ambiente costante, per evitare che
     le zone in ombra risultino di un nero totale innaturale (sulla Luna
     reale questo corrisponde alla luce diffusa dal terreno circostante
     / earthshine).

Convenzione degli angoli (coerente con lo script di partenza):
  - azimut (az): angolo nel piano x-y, misurato in gradi dall'asse x
    (colonne dell'immagine) in senso antiorario.
  - altezza (alt): angolo del sole sopra l'orizzonte, in gradi (0 =
    all'orizzonte, 90 = allo zenit).

Nota importante: questo e' un modello di riflessione Lambertiano con
ALBEDO UNIFORME. Nelle immagini lunari reali la riflettivita' del suolo
varia (mari scuri vs altopiani chiari, raggi degli impatti, ecc.)
indipendentemente dalla pendenza: e' un'ambiguita' intrinseca fra "forma"
e "colore del terreno" che questo script NON risolve (nessun metodo di
shape-from-shading a immagine singola puo' farlo senza assunzioni
aggiuntive). I parametri di guadagno/offset in fit.py servono ad
assorbire in parte questa ambiguita' a livello globale, non pixel per
pixel.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def sun_vector(az_deg: float, alt_deg: float) -> tuple[float, float, float]:
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    Lx = np.cos(alt) * np.cos(az)
    Ly = np.cos(alt) * np.sin(az)
    Lz = np.sin(alt)
    return Lx, Ly, Lz


def lambert_shading(h: np.ndarray, az_deg: float, alt_deg: float, dx: float = 1.0) -> np.ndarray:
    """Shading Lambertiano puro (senza ombre), in [0, 1]."""
    dh_dy, dh_dx = np.gradient(h, dx)
    Lx, Ly, Lz = sun_vector(az_deg, alt_deg)
    numer = (-dh_dx * Lx - dh_dy * Ly + Lz)
    denom = np.sqrt(dh_dx ** 2 + dh_dy ** 2 + 1.0)
    I = numer / denom
    return np.clip(I, 0.0, 1.0)


def cast_shadows(h: np.ndarray, az_deg: float, alt_deg: float, dx: float = 1.0,
                  step: float = 1.0, max_dist: float | None = None) -> np.ndarray:
    """Maschera di illuminazione diretta (1 = raggiunto dal sole, 0 = in
    ombra), calcolata marciando lungo la direzione del sole con
    campionamento bilineare (evita sia il wraparound di np.roll sia gli
    artefatti da spostamento a pixel interi dello script originale).

    Fuori dai bordi della griglia si assume "nessun ostacolo" (non c'e'
    modo di sapere cosa c'e' oltre il ritaglio), quindi i pixel vicino al
    bordo possono risultare leggermente piu' ottimisti (meno ombreggiati)
    di quanto sarebbero in un'immagine piu' grande.
    """
    ny, nx = h.shape
    if max_dist is None:
        max_dist = float(np.hypot(ny, nx))

    az = np.deg2rad(az_deg)
    alt_deg_clamped = max(alt_deg, 0.05)  # evita tan(0) esplosivo
    alt = np.deg2rad(alt_deg_clamped)
    dirx = np.cos(az)
    diry = np.sin(az)
    tan_alt = np.tan(alt)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    lit = np.ones((ny, nx), dtype=bool)

    n_steps = max(1, int(max_dist / step))
    for s in range(1, n_steps + 1):
        d = s * step
        sx = xx + dirx * d / dx
        sy = yy + diry * d / dx
        h_sample = map_coordinates(h, [sy, sx], order=1, mode='constant', cval=-1e9)
        blocked = h_sample > (h + d * tan_alt)
        lit &= ~blocked

    return lit.astype(np.float64)


def render(h: np.ndarray, az_deg: float, alt_deg: float, ambient: float = 0.08,
           dx: float = 1.0, shadow_step: float = 1.0, max_dist: float | None = None,
           with_shadows: bool = True) -> np.ndarray:
    """Immagine di luminosita' finale in [0, 1]:
    I = ambient + (1 - ambient) * shading_lambertiano * maschera_ombre
    """
    I_shading = lambert_shading(h, az_deg, alt_deg, dx=dx)
    if with_shadows:
        shadow = cast_shadows(h, az_deg, alt_deg, dx=dx, step=shadow_step, max_dist=max_dist)
    else:
        shadow = 1.0
    I = ambient + (1.0 - ambient) * I_shading * shadow
    return np.clip(I, 0.0, 1.0)
