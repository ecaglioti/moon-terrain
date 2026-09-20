"""
Generatore di lune sintetiche con crateri REALI (trasporto ottimo alla
Brenier fra coppie della libreria curata GLD100), invece del profilo
analitico di moon_disk.py. Non tocca moon_disk.py/terrain.py: riusa le
loro funzioni (geometria sferica, mari, shading, ombre) importandole
direttamente, e sostituisce solo il passo "genera e assembla i crateri"
con la pipeline di ot_crater.py.

Uso base:
    python3 moon_ot.py --grid 600 --n-craters-min 400 --n-craters-max 500

Vedi il README del progetto (sezione sul sottoprogetto trasporto
ottimo) per il contesto e le scelte di design.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # per importare moon_disk/terrain/render

import time
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter

from moon_disk import (spectral_synthesis_terrain, generate_maria_mask_flood, generate_maria_mask,
                        generate_crater_seeded_mare,
                        sphere_geometry, spherical_lambert_shading, MOON_RADIUS_KM)
from render import cast_shadows
from ot_crater import load_library, precompute_pool, scatter_ot_craters, assemble_ot_terrain_capped

DEFAULT_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), 'crater_library', 'curated_library.pkl')


def generate_moon_ot(grid_size: int = 400, seed: int | None = None, phase_deg: float | None = None,
                      n_craters: tuple[int, int] = (300, 400),
                      library_path: str = DEFAULT_LIBRARY_PATH, pool: list | None = None, n_pool: int = 300,
                      max_crater_diam_km: float = 90.0, min_crater_diam_km: float = 5.0,
                      size_freq_exponent: float = 1.9,
                      enable_maria: bool = True, maria_style: str = "flood", maria_coverage=(0.15, 0.35),
                      maria_flood_smooth_px: float = 0.0, maria_smoothing: float = 0.6,
                      maria_crater_suppression: float = 0.75, maria_density_mult: float = 1.0,
                      maria_depth_atten: float = 0.4,
                      highland_roughness_mult: float = 2.0,
                      terrain_beta: float = 3.1, base_amp_frac: float = 0.02,
                      crater_relief_scale: float = 1.0, crater_foreshorten_strength: float = 1.0,
                      avoid_crater_boundary_intersections: bool = True,
                      relief_strength: float = 1.4, ambient: float = 0.02,
                      with_shadows: bool = True, shadow_step: float = 1.5,
                      albedo_contrast: float = 0.45,
                      add_crater_mare: bool = False, crater_mare_diam_km: float = 200.0,
                      crater_mare_max_coverage: float = 0.05, maria_n_basins: int | None = None,
                      maria_edge_width_frac: float = 0.05,
                      verbose: bool = False):
    """Come moon_disk.generate_moon, ma i crateri vengono da
    scatter_ot_craters/assemble_ot_terrain_capped (trasporto ottimo su
    crateri reali) invece che dal profilo analitico. La maggior parte
    dei parametri ha lo stesso nome e lo stesso significato di
    generate_moon — vedi li' per i dettagli non ripetuti qui.

    Differenze rispetto a generate_moon:
    - una sola popolazione di crateri (non "normali"+"grandi" separati):
      dato che ogni cratere piazzato porta gia' con se' la texture reale
      del terreno circostante (incluse screziature/crateri minori), non
      serve una popolazione dedicata ai crateri grandi ne' un
      meccanismo per "permettere crateri piccoli dentro quelli grandi"
      -- l'evitamento delle intersezioni di bordo (gia' presente) basta.
    - niente peak_diam_threshold_km/peak_prob/irregularity/rim_frac:
      questi parametri esistevano per costruire artificialmente
      complessita' morfologica (picco centrale, bordo irregolare) che
      qui viene gia' dal dato reale.
    - pool/n_pool: le forme dei crateri sono precalcolate UNA VOLTA
      (vedi ot_crater.precompute_pool) e riusate con rotazione casuale —
      necessario per le prestazioni, vedi ot_crater.py. Passa un pool
      gia' calcolato (da una chiamata precedente) per rigenerare piu'
      lune velocemente senza ripetere il costo del pool.
    - add_crater_mare/crater_mare_diam_km/crater_mare_max_coverage:
      stesso mare opzionale "nato da un cratere" di generate_moon (vedi
      moon_disk.generate_crater_seeded_mare) — importato direttamente da
      moon_disk.py, non reimplementato. Default disattivato.
    - maria_n_basins: come in generate_moon — forza il numero di mari
      PRINCIPALI (0, 1 o 2) invece di lasciarlo casuale. Non conta il
      mare da cratere (add_crater_mare), sempre in piu'.
    - maria_edge_width_frac: come in generate_moon — quanto e' graduale
      la transizione di albedo al bordo di un mare (frazione di
      grid_size), per tutti i mari (principali e quello da cratere).
    - maria_density_mult: come in generate_moon — diradamento uniforme
      (indipendente dalla taglia) della densita' di crateri nei mari
      (1.0 = nessun effetto, 0.5 = meta' densita').

    Ritorna (img, meta, pool): meta come generate_moon (con l'aggiunta
    di n_pool_used), pool per poterlo riusare in chiamate successive.
    """
    rng = np.random.default_rng(seed)
    if phase_deg is None:
        phase_deg = float(rng.uniform(0, 360))
    n = int(rng.integers(n_craters[0], n_craters[1] + 1))

    t_pool0 = time.time()
    library = load_library(library_path)
    if pool is None:
        pool = precompute_pool(library, rng, n_pool=n_pool, verbose=verbose)
    if verbose:
        print(f"libreria: {len(library)} crateri, pool: {len(pool)} forme ({time.time()-t_pool0:.1f}s)")

    base = spectral_synthesis_terrain(grid_size, beta=terrain_beta, seed=int(rng.integers(0, 2**31 - 1)))

    if enable_maria:
        if maria_style == "flood":
            maria_mask, maria_coverage_used = generate_maria_mask_flood(
                grid_size, base, rng, smooth_radius_px=maria_flood_smooth_px, n_basins=maria_n_basins,
                edge_width_frac=maria_edge_width_frac)
        else:
            maria_mask, maria_coverage_used = generate_maria_mask(grid_size, rng, coverage=maria_coverage,
                                                                    n_basins=maria_n_basins,
                                                                    edge_width_frac=maria_edge_width_frac)
    else:
        maria_mask, maria_coverage_used = None, 0.0

    if enable_maria and add_crater_mare:
        # Mare aggiuntivo, opzionale, "nato" da un grande cratere invece
        # che da un minimo naturale del terreno — vedi
        # moon_disk.generate_crater_seeded_mare (stessa funzione usata da
        # generate_moon, non duplicata qui).
        crater_mare_mask, crater_mare_coverage_used, crater_mare_pos = generate_crater_seeded_mare(
            grid_size, base, rng, crater_diam_km=crater_mare_diam_km,
            max_coverage=crater_mare_max_coverage, edge_width_frac=maria_edge_width_frac,
            excluded=(maria_mask > 0.5) if maria_mask is not None else None)
        maria_mask = np.maximum(maria_mask, crater_mare_mask) if maria_mask is not None else crater_mare_mask
        maria_coverage_used = float(maria_coverage_used) + crater_mare_coverage_used
    else:
        crater_mare_coverage_used, crater_mare_pos = 0.0, None

    t_craters0 = time.time()
    ot_craters = scatter_ot_craters(grid_size, n, rng, library, pool,
                                     max_diam_km=max_crater_diam_km, min_diam_km=min_crater_diam_km,
                                     size_freq_exponent=size_freq_exponent,
                                     maria_mask=maria_mask, maria_suppression=maria_crater_suppression,
                                     maria_density_mult=maria_density_mult,
                                     maria_depth_atten=maria_depth_atten,
                                     foreshorten_strength=crater_foreshorten_strength,
                                     avoid_boundary_intersections=avoid_crater_boundary_intersections,
                                     verbose=verbose)
    if verbose:
        print(f"crateri piazzati: {len(ot_craters)} ({time.time()-t_craters0:.1f}s)")

    h_base = base * (grid_size * base_amp_frac)
    if maria_mask is not None:
        mare_level_frac = 0.30
        mare_residual_frac = float(np.clip(0.40 * (1.0 - maria_smoothing), 0.03, 0.40))
        mare_level = mare_level_frac * (grid_size * base_amp_frac)
        h_base_mare = mare_level + (base - 0.5) * (grid_size * base_amp_frac) * mare_residual_frac
        highland_residual_frac = float(np.clip(highland_roughness_mult * mare_residual_frac, 0.0, 1.0))
        h_base_highland = 0.5 * (grid_size * base_amp_frac) + \
            (base - 0.5) * (grid_size * base_amp_frac) * highland_residual_frac
        h_base = h_base_highland * (1.0 - maria_mask) + h_base_mare * maria_mask

    h_base = gaussian_filter(h_base, sigma=0.8)
    h_craters = assemble_ot_terrain_capped(np.zeros_like(base), amp=0.0, ot_craters=ot_craters)
    h = h_base + h_craters * crater_relief_scale

    phase_rad = np.deg2rad(phase_deg)
    sun_dir = np.array([np.sin(phase_rad), 0.0, np.cos(phase_rad)])

    n0, Tx, Ty, disk_mask = sphere_geometry(grid_size)
    I_shading = spherical_lambert_shading(h, n0, Tx, Ty, sun_dir, relief_strength=relief_strength)

    if with_shadows:
        local_alt = 15.0 + 70.0 * max(0.0, float(np.cos(phase_rad)))
        local_az = float(np.degrees(np.arctan2(sun_dir[1], sun_dir[0])))
        effective_shadow_step = shadow_step * (grid_size / 400.0)
        shadow = cast_shadows(h, local_az, local_alt, step=effective_shadow_step)
    else:
        shadow = 1.0

    img = ambient + (1.0 - ambient) * I_shading * shadow

    if maria_mask is not None:
        albedo_field = 1.0 - albedo_contrast * maria_mask
        img = img * albedo_field

    img = img * disk_mask
    img = np.clip(img, 0.0, 1.0)

    meta = dict(grid_size=grid_size, seed=seed, phase_deg=phase_deg, n_craters=len(ot_craters),
                n_pool_used=len(pool), maria_coverage=maria_coverage_used,
                add_crater_mare=add_crater_mare, crater_mare_coverage=crater_mare_coverage_used,
                crater_mare_pos=crater_mare_pos)
    return img, meta, pool


if __name__ == '__main__':
    import argparse
    from PIL import Image

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=int, default=600)
    parser.add_argument("--phase", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-craters-min", type=int, default=300)
    parser.add_argument("--n-craters-max", type=int, default=400)
    parser.add_argument("--n-pool", type=int, default=300)
    parser.add_argument("--albedo-contrast", type=float, default=0.45,
                         help="quanto sono piu' scuri i mari rispetto agli altopiani, in [0,1]")
    parser.add_argument("--add-crater-mare", action="store_true",
                         help="opzionale: aggiunge un mare piu' piccolo in piu', 'nato' da un grande cratere "
                              "(vedi moon_disk.generate_crater_seeded_mare) invece che da un minimo naturale del terreno")
    parser.add_argument("--crater-mare-diam-km", type=float, default=200.0,
                         help="solo con --add-crater-mare: diametro (km) del cratere da cui parte la conca scavata")
    parser.add_argument("--crater-mare-max-coverage", type=float, default=0.05,
                         help="solo con --add-crater-mare: tetto di copertura (frazione del disco), molto piu' basso "
                              "del cap dei mari normali")
    parser.add_argument("--maria-n-basins", type=int, default=None, choices=[0, 1, 2],
                         help="forza il numero di mari PRINCIPALI (0, 1 o 2) invece di lasciarlo casuale. "
                              "Non conta il mare da cratere (--add-crater-mare), sempre in piu'")
    parser.add_argument("--maria-flood-smooth", type=float, default=0.0,
                         help="solo con maria_style='flood' (il default qui): raggio (in pixel) della dilatazione "
                              "morfologica che arrotonda/smussa il contorno frastagliato del mare a percolazione. "
                              "0=disattivato (default), valori tipici 3-10 a grid~500 (scala con la risoluzione)")
    parser.add_argument("--maria-edge-width", type=float, default=0.05,
                         help="quanto e' graduale/sfumata (invece che netta) la transizione di albedo al bordo di "
                              "un mare, come frazione di --grid. Vale per tutti i mari (principali e quello da "
                              "cratere). Valori piu' alti = bordo piu' smussato/sfumato")
    parser.add_argument("--maria-crater-suppression", type=float, default=0.75,
                         help="probabilita' massima (per i crateri piu' grandi, in piena zona mare) che un cratere venga 'allagato' e scartato")
    parser.add_argument("--maria-density-mult", type=float, default=1.0,
                         help="fattore sulla densita' complessiva di crateri nei mari, indipendente dalla taglia "
                              "(1.0 = nessun effetto, 0.5 = meta' crateri rispetto agli altopiani, in piena zona mare)")
    parser.add_argument("--out", type=str, default="luna_ot.png")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    img, meta, pool = generate_moon_ot(grid_size=args.grid, seed=args.seed, phase_deg=args.phase,
                                        n_craters=(args.n_craters_min, args.n_craters_max),
                                        n_pool=args.n_pool, albedo_contrast=args.albedo_contrast,
                                        add_crater_mare=args.add_crater_mare,
                                        crater_mare_diam_km=args.crater_mare_diam_km,
                                        crater_mare_max_coverage=args.crater_mare_max_coverage,
                                        maria_n_basins=args.maria_n_basins,
                                        maria_flood_smooth_px=args.maria_flood_smooth,
                                        maria_edge_width_frac=args.maria_edge_width,
                                        maria_crater_suppression=args.maria_crater_suppression,
                                        maria_density_mult=args.maria_density_mult,
                                        verbose=args.verbose)
    print(f"generata in {time.time()-t0:.0f}s")
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print(f"salvata: {args.out}")
    print(f"seed={meta['seed']}, fase={meta['phase_deg']:.1f} gradi, n_crateri={meta['n_craters']}, "
          f"pool={meta['n_pool_used']}, copertura mare={meta.get('maria_coverage')}")
    if meta.get('add_crater_mare'):
        print(f"mare da cratere: copertura={meta['crater_mare_coverage']:.4f}, pos={meta['crater_mare_pos']}")
