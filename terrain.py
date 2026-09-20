"""
terrain.py
==========

Generazione del terreno (heightfield) usato per approssimare un'immagine
lunare.

Il terreno e' la somma di due componenti:

1. Un pattern frattale di base "ruvido", generato con l'algoritmo di
   *midpoint displacement* (diamond-square) sulla griglia quadrata: e'
   esattamente l'idea descritta nella richiesta originale ("prendo un
   triangolo/quadrato, alzo un punto a caso al centro, poi suddivido
   ricorsivamente") ma implementata sulla griglia quadrata invece che su
   una mesh di triangoli, perche' cosi' produce direttamente un array
   NxN allineato ai pixel dell'immagine da approssimare. Statisticamente
   e' lo stesso tipo di frattale auto-affine (rumore 1/f) prodotto dalla
   suddivisione triangolare.

   Questo pattern viene generato UNA SOLA VOLTA con un seed fisso, e poi
   durante l'ottimizzazione viene solo riscalato da uno scalare
   ``amp`` (ampiezza). Rigenerarlo ad ogni valutazione della funzione di
   costo sarebbe troppo lento (l'algoritmo e' intrinsecamente sequenziale,
   non vettorizzabile in numpy puro).

2. Uno o piu' crateri, aggiunti esplicitamente come conche + bordo
   rialzato (con leggera irregolarita' angolare), perche' i crateri sono
   la struttura dominante e riconoscibile in un'immagine lunare e un
   rumore frattale generico non li produce da solo.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# 1. Terreno frattale di base (diamond-square / midpoint displacement)
# ---------------------------------------------------------------------

def diamond_square(power: int, roughness: float = 0.6, seed: int | None = 0) -> np.ndarray:
    """Genera un heightfield frattale NxN con N = 2**power + 1.

    roughness: in [0,1] circa. Piu' e' alto, piu' il terreno risulta
    frastagliato (i dettagli fini restano "alti" anche quando la
    dimensione delle celle si riduce). Valori tipici: 0.4 (dolce) - 0.8
    (molto accidentato).
    """
    size = 2 ** power + 1
    rng = np.random.default_rng(seed)
    h = np.zeros((size, size), dtype=np.float64)

    # inizializza i 4 angoli
    h[0, 0] = rng.uniform(-1, 1)
    h[0, -1] = rng.uniform(-1, 1)
    h[-1, 0] = rng.uniform(-1, 1)
    h[-1, -1] = rng.uniform(-1, 1)

    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        scale *= (2.0 ** (-roughness))

        # --- diamond step: centro di ogni quadrato ---
        for y in range(half, size - 1, step):
            for x in range(half, size - 1, step):
                avg = (h[y - half, x - half] + h[y - half, x + half] +
                       h[y + half, x - half] + h[y + half, x + half]) / 4.0
                h[y, x] = avg + rng.uniform(-1, 1) * scale

        # --- square step: centro di ogni diamante ---
        for y in range(0, size, half):
            x_start = half if (y // half) % 2 == 0 else 0
            for x in range(x_start, size, step):
                total = 0.0
                count = 0
                if y - half >= 0:
                    total += h[y - half, x]
                    count += 1
                if y + half < size:
                    total += h[y + half, x]
                    count += 1
                if x - half >= 0:
                    total += h[y, x - half]
                    count += 1
                if x + half < size:
                    total += h[y, x + half]
                    count += 1
                h[y, x] = total / count + rng.uniform(-1, 1) * scale

        step = half

    h -= h.min()
    h /= (h.max() + 1e-12)
    return h


def resample_to(h: np.ndarray, size: int) -> np.ndarray:
    """Ricampiona (bilineare) un heightfield quadrato a size x size."""
    from scipy.ndimage import zoom
    factor = size / h.shape[0]
    out = zoom(h, factor, order=1)
    return out[:size, :size]


def make_base_pattern(grid_size: int, roughness: float = 0.6, seed: int | None = 0) -> np.ndarray:
    """Genera il pattern frattale di base gia' ricampionato a grid_size,
    normalizzato in [0, 1]. Da chiamare una volta sola, prima del fit."""
    power = int(np.ceil(np.log2(max(grid_size - 1, 1))))
    raw = diamond_square(power, roughness=roughness, seed=seed)
    out = resample_to(raw, grid_size)
    out -= out.min()
    out /= (out.max() + 1e-12)
    return out


# ---------------------------------------------------------------------
# 1b. Terreno di fondo via sintesi spettrale (alternativa a diamond-square)
# ---------------------------------------------------------------------
#
# Confrontando il pattern frattale di diamond_square con un vero DEM lunare
# (un ritaglio reale scaricato da LROC/GLD100, terreno "di fondo" lontano da
# un singolo cratere dominante), lo spettro di potenza radiale del terreno
# reale ha una pendenza log-log di circa -3.1 (molto rumore a grande scala,
# poco a scala fine). diamond_square, anche al valore di roughness piu' alto
# provato (fino a 2.0), non supera circa -2.77: NON e' un problema di
# taratura del parametro, e' un limite strutturale dell'algoritmo (i passi
# di suddivisione ricorsiva + il ricampionamento finale impongono un tetto a
# quanto puo' risultare "liscio su grande scala rispetto al dettaglio fine",
# qualunque valore di roughness si scelga). Per questo qui si genera il
# rumore frattale direttamente nel dominio di Fourier, dando pieno controllo
# sulla pendenza spettrale (beta) senza alcun tetto.

def spectral_synthesis_terrain(grid_size: int, beta: float = 3.1, seed: int | None = 0) -> np.ndarray:
    """Genera un heightfield frattale isotropo con spettro di potenza
    P(f) ~ f^-beta, normalizzato in [0, 1].

    beta: pendenza spettrale target. Valori piu' alti = piu' energia
    concentrata sulle basse frequenze (grandi macchie dolci, poco dettaglio
    fine) — l'opposto esatto del parametro roughness di diamond_square
    nella sua accezione corretta (vedi nota sopra), ma qui SENZA il tetto
    strutturale: beta=3.1 e' il valore misurato su un vero DEM lunare
    (patch di fondo vicino a Tycho, GLD100 64 px/grado) e non e' raggiungibile
    da diamond_square nemmeno al suo estremo piu' "liscio".
    seed: seed per le fasi casuali (l'ampiezza per ogni frequenza e'
    invece fissata deterministicamente da beta — solo le fasi sono casuali,
    come nella sintesi spettrale standard per superfici frattali/moto
    Browniano frazionario, Voss 1988).
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(grid_size)[:, None]
    fx = np.fft.fftfreq(grid_size)[None, :]
    freq = np.sqrt(fx ** 2 + fy ** 2)
    freq[0, 0] = 1.0  # placeholder, azzerato subito sotto (evita 0^-beta)

    amplitude = freq ** (-beta / 2.0)
    amplitude[0, 0] = 0.0  # nessuna componente continua (offset globale ininfluente)

    phases = rng.uniform(0, 2 * np.pi, size=(grid_size, grid_size))
    F = amplitude * (np.cos(phases) + 1j * np.sin(phases))

    h = np.fft.ifft2(F).real
    h -= h.min()
    h /= (h.max() + 1e-12)
    return h


# ---------------------------------------------------------------------
# 2. Crateri
# ---------------------------------------------------------------------

def _foreshorten_offsets(dx: np.ndarray, dy: np.ndarray,
                          radial_unit: tuple[float, float] | None,
                          foreshorten_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Corregge gli offset (dx, dy) — distanza dal centro del cratere, in
    pixel dell'immagine — per la prospettiva sferica, PRIMA di calcolare
    r/theta per il profilo del cratere.

    Perche' serve: un cratere e' (circa) un cerchio fisico sulla
    superficie sferica. In proiezione ortografica, un cerchio fisico
    centrato vicino al lembo NON proietta come un cerchio nell'immagine —
    proietta come un'ellisse, schiacciata nella direzione RADIALE
    (verso/dal centro del disco) di un fattore w0 = cos(theta_centro)
    (theta_centro = distanza angolare del CENTRO del cratere dal centro
    del disco, cioe' la stessa w calcolata in sphere_geometry). La
    direzione tangenziale (perpendicolare) non si accorcia affatto.
    Finora sia add_crater sia crater_fields definivano il profilo del
    cratere come funzione della sola distanza euclidea in pixel
    (r=sqrt(dx^2+dy^2)), identica in ogni direzione e indipendente dalla
    posizione sul disco — quindi un cratere vicino al lembo veniva
    disegnato circolare in pixel, non ellittico, mentre nella realta'
    (e nelle foto vere) i crateri vicino al lembo appaiono chiaramente
    schiacciati/ellittici.

    radial_unit: versore (2D, coordinate immagine) dal centro del disco
    verso il centro del cratere. foreshorten_w: cos(theta_centro), in
    (0, 1] (1 = nessuna correzione, cioe' cratere al centro del disco o
    field "flat" senza nozione di sfera — comportamento di prima).
    Se radial_unit e' None non si applica alcuna correzione (default,
    per compatibilita' con l'uso "isolato" di fit.py/main.py, che non
    posizionano il cratere su un disco).

    Derivazione: parametrizzando il punto sulla sfera lungo la direzione
    radiale in funzione di rho=sin(theta), il vettore tangente in quella
    direzione ha modulo esattamente 1/w0 (vedi il commento nel modulo
    README) — cioe' un passo di 1 pixel in direzione radiale, vicino al
    lembo, corrisponde a un tratto di superficie reale molto piu' lungo
    che al centro. Per disegnare un cratere di dimensione fisica
    costante, la componente RADIALE dell'offset va quindi DIVISA per w0
    (amplificata) prima di calcolare la distanza effettiva dal centro:
    cosi' il bordo (r_eff=radius) viene raggiunto a un offset in pixel
    piu' piccolo in direzione radiale che in direzione tangenziale — cioe'
    esattamente un'ellisse schiacciata verso il centro del disco."""
    if radial_unit is None or foreshorten_w >= 0.999999:
        return dx, dy
    rx, ry = radial_unit
    radial = dx * rx + dy * ry
    tangential = -dx * ry + dy * rx
    w0 = max(foreshorten_w, 0.05)  # stesso floor di sicurezza di sphere_geometry (w_safe)
    radial_eff = radial / w0
    # ricompone (dx, dy) equivalenti nella base originale, cosi' r/theta
    # sotto restano definiti come al solito ma con la geometria corretta
    dx_eff = radial_eff * rx - tangential * ry
    dy_eff = radial_eff * ry + tangential * rx
    return dx_eff, dy_eff


def add_crater(h: np.ndarray, cx: float, cy: float, radius: float, depth: float,
                rim_frac: float = 0.35, rim_height_frac: float = 0.35,
                irregularity: float = 0.08, seed: int | None = None,
                peak_frac: float = 0.0, peak_radius_frac: float = 0.15,
                radial_unit: tuple[float, float] | None = None,
                foreshorten_w: float = 1.0) -> np.ndarray:
    """Somma un cratere a h e ne ritorna la copia modificata.

    - (cx, cy): centro del cratere in coordinate pixel (x=colonna, y=riga)
    - radius: raggio "nominale" del bordo del cratere
    - depth: profondita' della conca (in unita' di altezza)
    - rim_frac: larghezza del bordo rialzato, come frazione del raggio
    - rim_height_frac: altezza del bordo rialzato, come frazione di depth
    - irregularity: quanto il bordo si discosta da un cerchio perfetto
      (0 = cerchio perfetto). Piccole perturbazioni angolari rendono il
      bordo piu' realistico invece che un anello geometrico esatto.
    - peak_frac: altezza del picco centrale, come frazione di depth (0 =
      nessun picco). I crateri complessi (tipicamente da ~15 km di
      diametro in su, es. Tycho) hanno spesso un rialzo roccioso al
      centro della conca, dovuto al rimbalzo elastico del suolo dopo
      l'impatto: senza questo termine, il modello puo' solo simularlo
      "sprecando" un secondo cratere sovrapposto per riprodurne l'effetto
      visivo con il proprio bordo rialzato, il che genera un cratere
      fantasma invece di un picco vero e proprio.
    - peak_radius_frac: raggio del picco centrale, come frazione di radius
      (i picchi centrali reali sono in genere il 10-25% del diametro del
      cratere, molto piu' stretti del cratere stesso).
    - radial_unit / foreshorten_w: correzione prospettica per crateri
      vicino al lembo di un disco — vedi _foreshorten_offsets sopra.
      Default None/1.0 = nessuna correzione (comportamento di prima),
      corretto per un cratere isolato non posizionato su un disco.
    """
    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    dx, dy = _foreshorten_offsets(dx, dy, radial_unit, foreshorten_w)
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    theta = np.arctan2(dy, dx)

    if irregularity > 0:
        rng = np.random.default_rng(seed if seed is not None else int(abs(cx * 977 + cy * 733)) % (2**32))
        phases = rng.uniform(0, 2 * np.pi, size=3)
        distortion = radius * irregularity * (
            np.sin(3 * theta + phases[0]) * 0.5 +
            np.sin(5 * theta + phases[1]) * 0.3 +
            np.sin(8 * theta + phases[2]) * 0.2
        )
    else:
        distortion = 0.0

    r_rim = radius + distortion

    # conca (bowl): profilo gaussiano-quartico, morbido al centro
    bowl = -depth * np.exp(-((r / (r_rim * (1 - rim_frac) + 1e-6)) ** 4))
    # bordo rialzato: anello gaussiano attorno a r_rim
    rim = (depth * rim_height_frac) * np.exp(-(((r - r_rim) / (radius * rim_frac * 0.6 + 1e-6)) ** 2))
    # picco centrale (opzionale): rialzo gaussiano stretto al centro della conca
    if peak_frac > 0:
        peak = (depth * peak_frac) * np.exp(-((r / (radius * peak_radius_frac + 1e-6)) ** 2))
    else:
        peak = 0.0

    return h + bowl + rim + peak


def assemble_terrain(base_pattern: np.ndarray, amp: float, craters: list[dict]) -> np.ndarray:
    """Combina pattern di base (scalato) + lista di crateri.
    Ogni elemento di craters e' un dict con le chiavi di add_crater
    (cx, cy, radius, depth, rim_frac, rim_height_frac, irregularity,
    peak_frac, peak_radius_frac).

    NOTA: qui i crateri sono sommati linearmente, quindi dove piu' bordi
    rialzati si sovrappongono per caso l'altezza (e la pendenza locale)
    si accumula oltre quella di ogni singolo cratere. Per un singolo
    cratere isolato (il caso d'uso di fit.py/main.py, che adattano UN
    cratere reale) questo non e' un problema. Per un campo denso di
    centinaia di crateri sovrapposti (moon_disk.py) puo' esserlo — vedi
    ``assemble_terrain_capped`` sotto."""
    h = base_pattern * amp
    for c in craters:
        h = add_crater(h, **c)
    return h


# ---------------------------------------------------------------------
# 2b. Combinazione non-lineare (max/min) per campi densi di crateri
# ---------------------------------------------------------------------
#
# Debug del problema "la falce non sembra piu' una falce" (immagini troppo
# rugose, macchie chiare sparse anche sul lato in ombra): la causa
# DOMINANTE si e' rivelata essere il terreno di fondo (spectral_synthesis_
# terrain), non i crateri — vedi la nota nel modulo su beta e sull'
# ampiezza (base_amp_frac in moon_disk.py). Ma sommare linearmente
# centinaia di crateri sovrapposti (assemble_terrain sopra) resta comunque
# fisicamente discutibile: un nuovo impatto scava il terreno che trova,
# non si limita ad aggiungere la propria profondita' sopra una depressione
# gia' esistente. Queste funzioni offrono una combinazione alternativa,
# per pixel: la conca PIU' PROFONDA fra tutti i crateri che coprono quel
# pixel (minimo) e il bordo/picco PIU' ALTO (massimo), invece della somma
# di tutti. Il risultato e' equivalente per un cratere isolato (nessun
# altro cratere a coprire lo stesso pixel) e diverso solo dove c'e'
# sovrapposizione reale.

def crater_fields(shape: tuple[int, int], cx: float, cy: float, radius: float, depth: float,
                   rim_frac: float = 0.35, rim_height_frac: float = 0.35,
                   irregularity: float = 0.08, seed: int | None = None,
                   peak_frac: float = 0.0, peak_radius_frac: float = 0.15,
                   pad_factor: float = 1.6,
                   radial_unit: tuple[float, float] | None = None,
                   foreshorten_w: float = 1.0):
    """Come add_crater, ma per UN cratere e limitato a un riquadro locale
    attorno al centro (invece che alla griglia intera): per centinaia di
    crateri, il costo di calcolare ogni volta un array NxN completo
    (come fa add_crater) e' inutile dato che un cratere ha effetto
    trascurabile a molti raggi di distanza dal proprio centro.

    radial_unit / foreshorten_w: correzione prospettica per crateri
    vicino al lembo di un disco — vedi _foreshorten_offsets sopra. Il
    riquadro locale (pad, sotto) resta dimensionato su radius "puro" (non
    ridotto): dato che la correzione puo' solo RESTRINGERE l'estensione
    in pixel nella direzione radiale (mai allargarla, foreshorten_w<=1),
    il riquadro calcolato senza correzione e' sempre sufficiente.

    Ritorna (y0, y1, x0, x1, bowl, bump): il riquadro (slice) e le due
    componenti separate — bowl (<=0, la conca) e bump (>=0, bordo +
    eventuale picco centrale) — cosi' il chiamante puo' combinarle con
    min/max invece che sommarle. bowl/bump sono None se il riquadro cade
    interamente fuori dalla griglia."""
    ny, nx = shape
    pad = radius * pad_factor + 4
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(ny, int(np.ceil(cy + pad)))
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(nx, int(np.ceil(cx + pad)))
    if y1 <= y0 or x1 <= x0:
        return y0, y1, x0, x1, None, None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dx_ = xx - cx
    dy_ = yy - cy
    dx_, dy_ = _foreshorten_offsets(dx_, dy_, radial_unit, foreshorten_w)
    r = np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-6
    theta = np.arctan2(dy_, dx_)

    if irregularity > 0:
        rng = np.random.default_rng(seed if seed is not None else int(abs(cx * 977 + cy * 733)) % (2**32))
        phases = rng.uniform(0, 2 * np.pi, size=3)
        distortion = radius * irregularity * (
            np.sin(3 * theta + phases[0]) * 0.5 +
            np.sin(5 * theta + phases[1]) * 0.3 +
            np.sin(8 * theta + phases[2]) * 0.2
        )
    else:
        distortion = 0.0

    r_rim = radius + distortion
    bowl = -depth * np.exp(-((r / (r_rim * (1 - rim_frac) + 1e-6)) ** 4))
    rim = (depth * rim_height_frac) * np.exp(-(((r - r_rim) / (radius * rim_frac * 0.6 + 1e-6)) ** 2))
    if peak_frac > 0:
        peak = (depth * peak_frac) * np.exp(-((r / (radius * peak_radius_frac + 1e-6)) ** 2))
    else:
        peak = 0.0
    bump = rim + peak
    return y0, y1, x0, x1, bowl, bump


def assemble_terrain_capped(base_pattern: np.ndarray, amp: float, craters: list[dict]) -> np.ndarray:
    """Come assemble_terrain, ma i crateri sono combinati per MASSIMO
    (bordo/picco) e MINIMO (conca) invece che per somma — vedi la nota
    sopra. Per pixel coperti da un solo cratere il risultato e' identico
    ad assemble_terrain; dove piu' crateri si sovrappongono, la
    profondita'/altezza combinata resta quella del cratere piu'
    profondo/alto li', invece di sommarsi. Piu' lento per un singolo
    cratere isolato (overhead della slice), ma per centinaia di crateri
    piccoli e' anche piu' VELOCE di assemble_terrain (ogni cratere lavora
    solo sul proprio riquadro locale, non sulla griglia intera)."""
    h = base_pattern * amp
    ny, nx = h.shape
    bowl_min = np.zeros_like(h)
    bump_max = np.zeros_like(h)
    for c in craters:
        y0, y1, x0, x1, bowl, bump = crater_fields((ny, nx), **c)
        if bowl is None:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        np.minimum(bowl_min[sl], bowl, out=bowl_min[sl])
        np.maximum(bump_max[sl], bump, out=bump_max[sl])
    return h + bowl_min + bump_max
