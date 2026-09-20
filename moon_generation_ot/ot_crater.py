"""
Crateri artificiali via trasporto ottimo (alla Brenier) fra coppie di
crateri reali della libreria curata (crater_library/curated_library.pkl,
vedi scripts/build_library.py), usati come "stampella" da moon_ot.py al
posto del profilo analitico di terrain.add_crater/crater_fields.

Ricetta, fissata e validata nel sottoprogetto crater_ot/ (vedi README,
sezione "Sottoprogetto: crateri artificiali via trasporto ottimo"):
  - ogni cratere della libreria e' rappresentato come nuvola di punti
    (x, y, h) in R^3 su una griglia canonica (raggio del cratere = 1,
    estensione CANON_EXTENT unita' oltre il bordo);
  - fra due crateri scelti si calcola un assegnamento ESATTO
    (scipy.optimize.linear_sum_assignment, l'algoritmo ungherese —
    provato PIU' VELOCE e piu' fedele del Sinkhorn entropico usato in un
    primo tentativo, che produceva una dissolvenza incrociata invece di
    una vera deformazione);
  - si usa SEMPRE t=0.5 (il "punto medio" alla Brenier, non un t
    casuale — scelta esplicita dell'utente: piu' semplice, sempre ben
    definito);
  - il campo di spostamento fra i due punti accoppiati viene smussato
    (gaussiana) prima di interpolare, per eliminare il rumore "a sale e
    pepe" che un assegnamento esatto puo' produrre quando piu' coppie
    hanno costo quasi uguale.

Nota sulle prestazioni: l'assegnamento ungherese e' economico (~0.3-0.5s
per una griglia 40x40) ma NON abbastanza per farlo una volta per OGNI
cratere piazzato — un generatore tipico ne piazza da qualche centinaio a
decine di migliaia. Si usa quindi un POOL precalcolato di forme (poche
centinaia, vedi precompute_pool): ogni forma e' generata una volta sola
da una coppia scelta a caso nella libreria, poi riusata (con rotazione
casuale e ricalibrazione della profondita' al bersaglio) per molti
crateri della stessa fascia dimensionale. La varieta' resta comunque
ampia (centinaia di forme distinte, ciascuna orientata a caso ogni
volta), molto di piu' di quanta ne desse mai il profilo analitico.
"""
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment

CANON_EXTENT = 1.7  # deve combaciare con build_library.py


def load_library(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _downsample(h, n):
    n_full = h.shape[0]
    factor = n_full // n
    if factor <= 1:
        return h[:n, :n]
    h2 = h[:factor * n, :factor * n].reshape(n, factor, n, factor).mean(axis=(1, 3))
    return h2


def _norm_h(h):
    h = h - h.mean()
    return h / (h.std() + 1e-9)


def brenier_midpoint(patchA, patchB, ot_grid_n=40, flow_sigma=1.3, lam=1.0,
                      canon_extent=CANON_EXTENT, render_n=None):
    """Cratere 'a meta'' (t=0.5) fra due patch canoniche (stessa griglia,
    stesso raggio normalizzato a 1). Ritorna una forma ADIMENSIONALE
    (z-scored, non ancora calibrata in profondita' fisica — vedi
    calibrate_depth) alla risoluzione render_n (default: quella nativa
    delle patch in ingresso)."""
    n_full = patchA.shape[0]
    render_n = render_n or n_full

    hA = _downsample(patchA, ot_grid_n)
    hB = _downsample(patchB, ot_grid_n)
    hAn, hBn = _norm_h(hA), _norm_h(hB)

    lin = np.linspace(-canon_extent, canon_extent, ot_grid_n)
    gx, gy = np.meshgrid(lin, lin)
    XA = np.stack([gx.ravel(), gy.ravel(), hAn.ravel()], axis=1)
    XB = np.stack([gx.ravel(), gy.ravel(), hBn.ravel()], axis=1)
    XA_w = XA.copy(); XA_w[:, 2] *= np.sqrt(lam)
    XB_w = XB.copy(); XB_w[:, 2] *= np.sqrt(lam)

    C = ((XA_w[:, None, :] - XB_w[None, :, :]) ** 2).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(C)

    T_pos = XB[col_ind, :2]
    T_val = XB[col_ind, 2]

    dpos = (T_pos - XA[:, :2]).reshape(ot_grid_n, ot_grid_n, 2)
    dval = (T_val - XA[:, 2]).reshape(ot_grid_n, ot_grid_n)
    dpos_s = np.stack([gaussian_filter(dpos[..., 0], flow_sigma),
                        gaussian_filter(dpos[..., 1], flow_sigma)], axis=-1).reshape(-1, 2)
    dval_s = gaussian_filter(dval, flow_sigma).ravel()

    t = 0.5
    pos_t = XA[:, :2] + t * dpos_s
    val_t = XA[:, 2] + t * dval_s

    lin_hi = np.linspace(-canon_extent, canon_extent, render_n)
    gxr, gyr = np.meshgrid(lin_hi, lin_hi)
    grid_val = griddata(pos_t, val_t, (gxr, gyr), method='cubic')
    if np.isnan(grid_val).any():
        grid_nn = griddata(pos_t, val_t, (gxr, gyr), method='nearest')
        grid_val = np.where(np.isnan(grid_val), grid_nn, grid_val)

    # Anche dopo aver smussato il campo di flusso, restano occasionali
    # spuntoni isolati (1-2 pixel) vicino al bordo, dove un assegnamento
    # ungherese "quasi alla pari" fra punti adiacenti puo' produrre un
    # salto netto che griddata (cubica) amplifica localmente. Nel demo
    # originale (crater_ot/run_morph_final.py) questo si notava poco
    # perche' veniva applicato solo per l'hillshading di anteprima; qui
    # perche' la forma diventa DIRETTAMENTE l'altezza del terreno (e poi
    # va in un'ombreggiatura basata sul gradiente), va ripulita alla
    # fonte con la stessa leggera sfocatura gia' necessaria altrove nel
    # progetto per il rumore di quantizzazione dei DEM.
    grid_val = gaussian_filter(grid_val, sigma=1.2)

    # La nuvola di punti spostata (pos_t) e' una griglia REGOLARE quadrata
    # perturbata dal campo di flusso: vicino agli angoli del quadrato
    # (raggio > ~1.2-1.3, cioe' oltre la zona affidabile attorno al
    # cratere) i punti sono radi e spesso fuori dal fascio convesso, quindi
    # griddata ripiega sul vicino piu' vicino -> un mosaico "a Voronoi"
    # rumoroso e spigoloso, non fisico (si vede chiaramente rendendo la
    # forma da sola). Quella zona non contiene comunque informazione utile
    # sul cratere (e' gia' oltre il bordo + un margine), quindi la si
    # sfuma dolcemente a 0 (= piana locale) con una finestra radiale, cosi'
    # sia il riferimento di "piana" usato da calibrate_depth sia il
    # campionamento in ot_crater_fields restano puliti.
    rr = np.hypot(gxr, gyr)
    taper_start, taper_end = 1.05, 1.3
    w = np.clip((taper_end - rr) / (taper_end - taper_start), 0.0, 1.0)
    w = w * w * (3 - 2 * w)  # smoothstep
    grid_val = grid_val * w
    return grid_val


def pick_pair(library, target_r_km, rng, k_neighbors=15, rim_std_max=0.11):
    """2 crateri della libreria di dimensione 'simile' al bersaglio: fra
    i k piu' vicini in raggio (km) a target_r_km, pescati a caso senza
    reinserimento.

    rim_std_max: la libreria curata (vedi build_library.py) tiene tutto
    cio' che ha rim_std<=0.17, ma l'analisi di validazione aveva gia'
    mostrato che la qualita' scende sensibilmente sopra ~0.11-0.12 (sotto,
    quasi tutti bordi puliti; sopra, una parte crescente sono crateri
    degradati o coppie non pulite). Qui, dove il crater serve da
    STAMPO per il trasporto ottimo (non solo da mostrare com'e'), si
    preferiscono le fonti migliori quando ce ne sono abbastanza vicino
    alla taglia richiesta; altrimenti si ripiega sull'intera libreria
    pur di avere due candidati alla taglia giusta."""
    r_all = np.array([c['r_km'] for c in library])
    rim_std_all = np.array([c.get('rim_std', 0.0) for c in library])
    good = rim_std_all <= rim_std_max
    pool_idx = np.where(good)[0] if good.sum() >= 2 * k_neighbors else np.arange(len(library))
    order = pool_idx[np.argsort(np.abs(r_all[pool_idx] - target_r_km))]
    k = min(k_neighbors, len(order))
    idx = rng.choice(order[:k], size=2, replace=False)
    return library[idx[0]], library[idx[1]]


def precompute_pool(library, rng, n_pool=300, ot_grid_n=40, render_n=96,
                     size_bins_km=(5, 8, 12, 16, 22, 30, 45, 65), verbose=False):
    """Genera n_pool forme di cratere (adimensionali) una volta sola,
    scegliendo ogni volta una coppia a caso vicina a una taglia bersaglio
    campionata dai bin dati (cosi' il pool copre l'intera libreria, non
    solo la sua parte piu' numerosa). Ogni voce del pool porta con se'
    'r_km_ref' (media delle taglie native della coppia sorgente) per la
    ricerca per vicinanza in scatter_ot_craters."""
    pool = []
    for i in range(n_pool):
        target = float(rng.choice(size_bins_km))
        cA, cB = pick_pair(library, target, rng)
        shape = brenier_midpoint(cA['patch'], cB['patch'], ot_grid_n=ot_grid_n, render_n=render_n)
        r_km_ref = 0.5 * (cA['r_km'] + cB['r_km'])
        pool.append(dict(shape=shape.astype(np.float32), r_km_ref=r_km_ref))
        if verbose and (i + 1) % 50 == 0:
            print(f"  pool: {i+1}/{n_pool} forme generate", flush=True)
    return pool


def calibrate_depth(shape, target_depth_px, canon_extent=CANON_EXTENT):
    """Da una forma adimensionale a un profilo in unita' di altezza
    coerenti col resto del generatore: 0 = livello della piana locale,
    negativo = fondo/conca, positivo = bordo/picco. target_depth_px e'
    la stessa quantita' ('depth', gia' calcolata da leggi di scala
    tipo Pike) usata dal generatore procedurale per un cratere dello
    stesso diametro — cosi' i crateri OT restano fisicamente comparabili
    a quelli analitici invece di avere un'ampiezza arbitraria ereditata
    dal DEM reale (che varia da cratere a cratere per ragioni che non
    hanno a che fare con quanto "profondo" dovrebbe essere QUESTO
    cratere alla taglia bersaglio)."""
    N = shape.shape[0]
    lin = np.linspace(-canon_extent, canon_extent, N)
    gx, gy = np.meshgrid(lin, lin)
    rr = np.hypot(gx, gy)
    floor_mask = rr < 0.35
    # oltre rr=1.4 la forma e' azzerata a forza (vedi la sfumatura radiale
    # in brenier_midpoint), quindi qui dentro plain_val e' garantito ~0 in
    # modo pulito, invece di essere calcolato su una zona che in passato
    # poteva contenere rumore di estrapolazione (vedi commento li').
    plain_mask = (rr > 1.45) & (rr < 1.65)
    floor_val = float(shape[floor_mask].mean()) if floor_mask.any() else float(shape.min())
    plain_val = float(shape[plain_mask].mean()) if plain_mask.any() else float(shape[rr > 1.0].mean())
    native_depth = plain_val - floor_val
    if native_depth < 1e-6:
        native_depth = float(shape.std()) + 1e-6
    scale = target_depth_px / native_depth
    # Alcune coppie reali hanno un fondo poco profondo rispetto al resto
    # del rilievo (bordo, screziature): normalizzare SOLO sulla profondita'
    # del fondo puo' allora amplificare tutto il resto in modo eccessivo.
    # Un limite (empirico) al fattore di scala evita crateri implausibili
    # senza impedire la normale variabilita' da cratere a cratere.
    scale = float(np.clip(scale, 0.3, 3.0))
    return (shape - plain_val) * scale


def ot_crater_fields(shape, cx, cy, radius, calibrated_shape, rotation_angle,
                      radial_unit=None, foreshorten_w=1.0, canon_extent=CANON_EXTENT,
                      pad_factor=1.05):
    """Come terrain.crater_fields, ma campiona una patch REALE
    (gia' calibrata in profondita', vedi calibrate_depth) invece di
    valutare una formula analitica. Ritorna (y0,y1,x0,x1,bowl,bump) con
    la stessa convenzione (bowl<=0, bump>=0) cosi' da poter riusare
    esattamente la combinazione per max/min di assemble_terrain_capped
    (vedi assemble_ot_terrain_capped sotto).

    radial_unit/foreshorten_w: stessa correzione ellittica da prospettiva
    usata dal generatore procedurale (terrain._foreshorten_offsets),
    applicata qui PRIMA di campionare la patch, cosi' un cratere reale
    vicino al lembo del disco viene schiacciato esattamente come lo
    sarebbe un cratere analitico nello stesso punto.
    rotation_angle: orientamento casuale (radianti) — senza, tutti i
    crateri costruiti dalla stessa forma del pool sarebbero visibilmente
    allineati allo stesso modo.
    """
    from terrain import _foreshorten_offsets  # import locale per evitare un ciclo a livello di modulo

    ny, nx = shape
    pad = radius * canon_extent * pad_factor + 4
    y0 = max(0, int(np.floor(cy - pad))); y1 = min(ny, int(np.ceil(cy + pad)))
    x0 = max(0, int(np.floor(cx - pad))); x1 = min(nx, int(np.ceil(cx + pad)))
    if y1 <= y0 or x1 <= x0:
        return y0, y1, x0, x1, None, None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    dx_eff, dy_eff = _foreshorten_offsets(dx, dy, radial_unit, foreshorten_w)

    ca, sa = np.cos(-rotation_angle), np.sin(-rotation_angle)
    dxr = dx_eff * ca - dy_eff * sa
    dyr = dx_eff * sa + dy_eff * ca

    u = dxr / radius
    v = dyr / radius
    N = calibrated_shape.shape[0]
    px = (u / canon_extent * 0.5 + 0.5) * (N - 1)
    py = (v / canon_extent * 0.5 + 0.5) * (N - 1)
    outside = (np.abs(u) > canon_extent) | (np.abs(v) > canon_extent)

    vals = map_coordinates(calibrated_shape, [py.ravel(), px.ravel()], order=1,
                            mode='nearest').reshape(px.shape)
    vals = np.where(outside, 0.0, vals)
    bowl = np.minimum(vals, 0.0)
    bump = np.maximum(vals, 0.0)
    return y0, y1, x0, x1, bowl, bump


def scatter_ot_craters(grid_size, n_craters, rng, library, pool,
                        max_diam_km=100.0, min_diam_km=3.0, size_freq_exponent=1.9,
                        maria_mask=None, maria_suppression=0.75, maria_density_mult=1.0,
                        maria_depth_atten=0.4,
                        foreshorten_strength=1.0,
                        avoid_boundary_intersections=True, max_placement_attempts=100,
                        avoid_craters=None, pool_k_neighbors=8, verbose=False):
    """Come moon_disk.scatter_craters (stessa distribuzione dei raggi,
    stesso ordine di piazzamento dal piu' grande al piu' piccolo, stesso
    meccanismo anti-intersezione-bordi: riusa direttamente le funzioni di
    moon_disk.py, non le reimplementa), ma invece di generare un profilo
    analitico per ogni cratere pesca dal POOL una forma reale gia'
    costruita (vedi precompute_pool) alla taglia piu' vicina disponibile,
    la ruota a caso e la ricalibra in profondita' alla taglia bersaglio
    (calibrate_depth) -- vedi il docstring del modulo per il perche' del
    pool invece di una nuova coppia+trasporto per ogni cratere.

    Ritorna una lista di dict pronti per assemble_ot_terrain_capped
    (**non** compatibili con terrain.add_crater/crater_fields — sono due
    rappresentazioni diverse dello stesso concetto di "cratere
    piazzato").
    """
    import math
    from moon_disk import (power_law_radii, crater_depth_over_diameter, disk_radial_geometry,
                            _crater_grid_insert, _crater_boundary_conflict, MOON_RADIUS_KM)

    km_per_pixel = MOON_RADIUS_KM / (grid_size / 2.0)
    r_max = (max_diam_km / 2.0) / km_per_pixel
    r_min_physical = (min_diam_km / 2.0) / km_per_pixel
    r_min = max(r_min_physical, 0.6)
    r_min = min(r_min, r_max * 0.5)

    pdf_slope = size_freq_exponent + 1.0
    radii = power_law_radii(n_craters, r_min, r_max, pdf_slope, rng)
    radii = np.sort(radii)[::-1]

    pool_r_km_ref = np.array([p['r_km_ref'] for p in pool])

    placement_grid: dict = {}
    max_avoid_radius = max((c["radius"] for c in avoid_craters), default=0.0) if avoid_craters else 0.0
    cell_size = 2.0 * max(r_max, max_avoid_radius, 1.0)
    if avoid_craters:
        for c in avoid_craters:
            _crater_grid_insert(placement_grid, c["cx"], c["cy"], c["radius"], cell_size)

    out = []
    n_forced_overlap = 0
    for radius in radii:
        cx = rng.uniform(0, grid_size)
        cy = rng.uniform(0, grid_size)
        D_km = 2.0 * radius * km_per_pixel

        m = 0.0
        if maria_mask is not None:
            iy = int(np.clip(round(cy), 0, grid_size - 1))
            ix = int(np.clip(round(cx), 0, grid_size - 1))
            m = float(maria_mask[iy, ix])
            size_frac = float(np.clip((radius - r_min) / (r_max - r_min + 1e-9), 0.0, 1.0))
            if rng.uniform() < maria_suppression * m * size_frac:
                continue
            if maria_density_mult < 1.0 and rng.uniform() < (1.0 - maria_density_mult) * m:
                continue  # diradamento uniforme (indipendente dalla taglia) nei mari

        if avoid_boundary_intersections:
            for _attempt in range(max_placement_attempts):
                if not _crater_boundary_conflict(cx, cy, radius, placement_grid, cell_size):
                    break
                cx = rng.uniform(0, grid_size)
                cy = rng.uniform(0, grid_size)
            else:
                n_forced_overlap += 1
            if maria_mask is not None:
                iy = int(np.clip(round(cy), 0, grid_size - 1))
                ix = int(np.clip(round(cx), 0, grid_size - 1))
                m = float(maria_mask[iy, ix])
        _crater_grid_insert(placement_grid, cx, cy, radius, cell_size)

        jitter = float(rng.lognormal(mean=0.0, sigma=0.15))
        d_over_D = crater_depth_over_diameter(D_km) * jitter
        if m > 0:
            d_over_D *= (1.0 - maria_depth_atten * m)
        diameter_px = 2.0 * radius
        target_depth_px = d_over_D * diameter_px

        order = np.argsort(np.abs(pool_r_km_ref - D_km))
        k = min(pool_k_neighbors, len(pool))
        chosen = pool[int(rng.choice(order[:k]))]
        calibrated_shape = calibrate_depth(chosen['shape'], target_depth_px)
        rotation_angle = float(rng.uniform(0, 2 * np.pi))

        radial_unit, foreshorten_w = disk_radial_geometry(cx, cy, grid_size, strength=foreshorten_strength)

        out.append(dict(cx=cx, cy=cy, radius=radius, calibrated_shape=calibrated_shape,
                         rotation_angle=rotation_angle, radial_unit=radial_unit,
                         foreshorten_w=foreshorten_w))

    if verbose and avoid_boundary_intersections:
        print(f"scatter_ot_craters: {n_forced_overlap}/{len(radii)} crateri piazzati "
              f"nonostante un conflitto di bordo (budget tentativi esaurito)")
    return out


def assemble_ot_terrain_capped(base_pattern, amp, ot_craters):
    """Come terrain.assemble_terrain_capped, ma per crateri OT: ogni
    elemento di ot_craters e' un dict con le chiavi di ot_crater_fields
    (cx, cy, radius, calibrated_shape, rotation_angle, radial_unit,
    foreshorten_w)."""
    h = base_pattern * amp
    ny, nx = h.shape
    bowl_min = np.zeros_like(h)
    bump_max = np.zeros_like(h)
    for c in ot_craters:
        y0, y1, x0, x1, bowl, bump = ot_crater_fields((ny, nx), **c)
        if bowl is None:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        np.minimum(bowl_min[sl], bowl, out=bowl_min[sl])
        np.maximum(bump_max[sl], bump, out=bump_max[sl])
    return h + bowl_min + bump_max
