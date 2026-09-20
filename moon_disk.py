"""
moon_disk.py
============

Generatore di "lune artificiali": immagini sintetiche dell'intero disco
lunare, con fase casuale (non solo luna piena) e crateri sparsi secondo
una distribuzione dimensionale realistica.

E' un uso diverso da terrain.py/render.py/fit.py: quegli script servono
ad ADATTARE un modello a UNA foto reale (un ritaglio piccolo). Qui invece
generiamo lune INVENTATE, per esplorazione/divertimento — nessun fit,
nessuna immagine target.

--------------------------------------------------------------------
Come funziona: una normale sferica, "disturbata" (bump mapping) dal
rilievo locale
--------------------------------------------------------------------

Ogni punto del disco visibile, a coordinate immagine (x,y), corrisponde
a un punto su una sfera vista in proiezione ortografica: la sua normale
"pura" N0(x,y) dipende solo dalla posizione (e' quella che da' la forma
a falce/gibbosa/quarto e l'oscuramento verso il lembo). A questa normale
sferica sommiamo la perturbazione dovuta alla pendenza LOCALE del
terreno (crateri, rilievo frattale) — la stessa tecnica del "bump
mapping" in computer graphics: il terreno non sposta realmente la
superficie della sfera (sarebbe una distorsione enorme, dato che le
altezze usate qui sono volutamente esagerate per essere visibili), ma
ne inclina la normale quel tanto che basta a produrre ombreggiatura
realistica.

Il punto cruciale (questo e' il miglioramento rispetto alla prima
versione): la normale finale in ogni punto e' N0(x,y) - dh/dx * Tx(x,y)
- dh/dy * Ty(x,y), dove Tx, Ty sono i vettori tangenti LOCALI alla
sfera in quel punto (non gli assi x,y dell'immagine!) — vicino al lembo
questi vettori si allungano (l'accorciamento prospettico), quindi la
STESSA pendenza in pixel produce un effetto sulla normale diverso a
seconda di dove ci si trova sul disco. Il prodotto scalare fra questa
normale e la direzione (unica, fissa: il sole e' a distanza
"infinita") del sole da' direttamente l'altezza/azimut EFFETTIVI del
sole rispetto alla superficie in quel punto, punto per punto — non piu'
una direzione uniforme su tutto il disco con sopra una maschera globale
moltiplicativa come nella prima versione.

APPROSSIMAZIONE ONESTA RIMASTA: le ombre PORTATE (un rilievo che blocca
la vista del sole a un rilievo vicino, calcolate col ray-marching di
render.py) usano ancora una singola direzione del sole per l'intero
disco — il ray-marching a raggi di render.py e' scritto per una
direzione globale unica per griglia, ed estenderlo a raggi che seguono
la vera curvatura sferica punto per punto sarebbe un lavoro
sostanzialmente piu' grande (un vero ray-marcher 3D sulla sfera).
Per questa direzione uso l'azimut VERO del sole (cosi' le ombre puntano
dalla parte giusta), ma un'ALTEZZA fissa moderata (non quella vera, che
varierebbe punto per punto e vicino ai quarti puo' avvicinarsi a zero,
facendo esplodere il ray-marching in ombre innaturali a chiazze su tutto
il disco): dove l'illuminazione vera e' comunque quasi nulla (vicino al
terminatore) il fattore d'ombra pesa poco perche' moltiplica un
I_shading gia' quasi zero, quindi l'approssimazione si vede solo come
"rilievo dei crateri leggermente meno drammatico del reale vicino al
terminatore", non come un errore vistoso. Quindi la GRADAZIONE
complessiva di luce (dove passa il terminatore, quanto si scurisce il
lembo, come cade la luce su ogni cratere in base a dove si trova sul
disco) e' ora corretta punto per punto; solo l'intensita' delle ombre
proprie dei singoli crateri usa un'altezza del sole approssimata.

Anche la distribuzione dei crateri e' un'approssimazione: usiamo una
legge di potenza per i raggi (vedi power_law_radii) ispirata al fatto,
discusso e reale, che la distribuzione dimensionale dei crateri lunari
segue circa una legge di potenza con "dimensione frattale" fra ~1.8 e
2.0 — senza pretendere di riprodurre esattamente le statistiche di una
regione lunare specifica.

--------------------------------------------------------------------
Mari vs altopiani: perche' una maschera PROCEDURALE e non una mappa
di albedo reale scaricata da LROC/Clementine
--------------------------------------------------------------------

Fino a qui il modello era geometria pura con albedo UNIFORME: nessuna
differenza di riflettivita' fra crateri e mari, che invece nella Luna
reale e' fortissima (i mari sono lava basaltica scura, gli altopiani
roccia chiara). Il tentativo naturale sarebbe scaricare una vera mappa
di albedo (es. il mosaico Clementine 750nm o LROC WAC) e campionarla.
In pratica, nel sandbox in cui gira questo script l'accesso di rete e'
limitato ai soli registri di pacchetti (pip/npm): i domini normali del
web (Wikimedia, NASA SVS, GitHub raw, ecc.) sono bloccati a livello di
policy di rete, quindi scaricare un'immagine reale non e' fattibile da
qui.

Ma ripensandoci, per QUESTO script (un generatore di lune INVENTATE, a
seed casuale) usare la vera mappa del lato visibile reale della Luna
sarebbe comunque concettualmente storto: legherebbe ogni luna generata
alla stessa identica geografia reale, cambiando solo l'illuminazione —
il contrario dell'idea di "lune casuali diverse ogni volta". Quindi la
soluzione qui e' generare una regionalizzazione mari/altopiani
PROCEDURALE (vedi generate_maria_mask), diversa per ogni seed come il
resto del terreno, ma fisicamente motivata: i mari reali sono bacini
da impatto GIGANTI rimasti topograficamente BASSI e poi allagati da
lava (per questo sono scuri E piatti E con pochi crateri grandi
superstiti, tutte cose correlate fra loro, non indipendenti). La
maschera nasce quindi come l'unione di 1-3 DISCHI (la sezione di un
bacino da impatto circolare, con un bordo leggermente irregolare, non
un cerchio perfetto — idea discussa insieme: "probabilmente hanno una
forma vicina a un disco, o all'unione di due dischi") invece che dalle
zone piu' basse di un campo frattale generico (che dava contorni a
"macchia di leopardo" senza relazione con un'origine da impatto). Le
zone dentro quei dischi diventano "mare": piu' scure (albedo_contrast),
allagate dalla lava fino a un livello fisso PIU' BASSO della media del
terreno circostante — un bacino, non un altopiano — con la rugosita'
fine del terreno di fondo fortemente attenuata li' (maria_smoothing) e
con meno crateri grandi superstiti (maria_crater_suppression scarta
probabilisticamente i crateri grandi che cadono li', simulando
l'allagamento lavico che ha cancellato i bacini piu' vecchi — i piccoli
restano, come nella realta').
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as ndi_label, distance_transform_edt

from terrain import assemble_terrain_capped, spectral_synthesis_terrain, _foreshorten_offsets
from render import cast_shadows


# Raggio lunare medio, in km — l'unica assunzione fisica del file, usata
# solo per convertire "voglio crateri fino a X km" in una frazione del
# raggio del disco in pixel.
MOON_RADIUS_KM = 1737.4


# ---------------------------------------------------------------------
# 1. Crateri sparsi con distribuzione dimensionale a legge di potenza
# ---------------------------------------------------------------------

def power_law_radii(n: int, r_min: float, r_max: float, slope: float, rng: np.random.Generator) -> np.ndarray:
    """Campiona n raggi da una densita' di probabilita' p(r) ~ r^-slope
    su [r_min, r_max], via inversione della CDF."""
    u = rng.uniform(0, 1, size=n)
    if abs(slope - 1.0) < 1e-9:
        # caso degenere p(r) ~ 1/r: CDF logaritmica
        return r_min * (r_max / r_min) ** u
    a = r_min ** (1 - slope)
    b = r_max ** (1 - slope)
    return (a + u * (b - a)) ** (1.0 / (1 - slope))


def crater_depth_over_diameter(D_km: float) -> float:
    """Rapporto profondita'/diametro (d/D) in funzione del diametro reale,
    da leggi di scala di tipo Pike (1974/1977): i crateri SEMPLICI (a
    scodella, sotto la transizione) hanno d/D quasi costante (~0.15-0.2),
    quelli COMPLESSI (sopra la transizione: pareti terrazzate, fondo
    piatto, spesso un picco centrale) sono molto piu' bassi in proporzione
    — la fisica e' che oltre una certa energia d'impatto il collasso
    gravitazionale delle pareti "allarga e appiattisce" il cratere invece
    di scavarlo piu' a fondo."""
    Dc = 18.0  # diametro di transizione semplice->complesso sulla Luna (~15-20 km in letteratura)
    if D_km < Dc:
        d_km = 0.196 * D_km ** 1.010
    else:
        d_km = 1.044 * D_km ** 0.301
    return d_km / D_km


def crater_rim_over_diameter(D_km: float) -> float:
    """Rapporto altezza-del-bordo/diametro, stessa famiglia di leggi di
    scala (Pike). Usata come approssimazione anche per i crateri complessi:
    la scala del bordo li' e' meno vincolata in letteratura, ma l'errore
    qui pesa meno di quello sulla profondita'."""
    h_rim_km = 0.036 * D_km ** 1.014
    return h_rim_km / D_km


def disk_radial_geometry(cx: float, cy: float, grid_size: int,
                          strength: float = 1.0) -> tuple[tuple[float, float], float]:
    """Ritorna (radial_unit, foreshorten_w) per un punto (cx, cy) in
    coordinate pixel — la stessa normalizzazione usata da sphere_geometry
    (centro griglia, raggio grid_size/2) — da passare a terrain.add_crater/
    crater_fields (correzione ellittica vicino al lembo) per un elemento
    circolare (cratere, bacino-mare, ...) centrato li'. Usata sia da
    scatter_craters sia da generate_maria_mask.

    strength (default 1.0): quanto applicare la correzione prospettica
    fisicamente esatta (w0 = cos(theta_centro), vedi
    terrain._foreshorten_offsets). 1.0 = correzione piena (un cratere
    esattamente al lembo, theta_centro=90 gradi, diventa uno spicchio
    schiacciato quasi a zero — fisicamente corretto in proiezione
    ortografica pura, ma l'utente lo ha trovato esteticamente troppo
    marcato: "mi pare che facevamo i crateri un po' ellittici"). Con
    strength<1 si interpola fra "nessuna correzione" (w0=1, cerchio
    perfetto ovunque) e quella fisica: w0_eff = 1 - strength*(1-w0), che
    ammorbidisce l'effetto vicino al lembo mantenendolo nullo al centro
    del disco (dove w0 e' gia' ~1) e diverso da zero (mai un cerchio
    degenere) anche esattamente al lembo. Non e' un compromesso fisico
    (una vera proiezione ortografica non ha una 'via di mezzo'), ma la
    fotografia reale della Luna non e' mai proiezione ortografica pura
    fino all'estremo lembo geometrico (illuminazione radente, risoluzione,
    atmosfera in altri corpi) quindi un ammorbidimento e' difendibile
    quanto la correzione esatta."""
    u0 = (cx - (grid_size - 1) / 2.0) / (grid_size / 2.0)
    v0 = (cy - (grid_size - 1) / 2.0) / (grid_size / 2.0)
    rho0 = float(np.hypot(u0, v0))
    foreshorten_w = float(np.sqrt(max(0.0, 1.0 - rho0 * rho0)))
    if strength < 1.0:
        foreshorten_w = 1.0 - strength * (1.0 - foreshorten_w)
    radial_unit = (u0 / rho0, v0 / rho0) if rho0 > 1e-6 else (1.0, 0.0)
    return radial_unit, foreshorten_w


def _crater_grid_insert(grid: dict, cx: float, cy: float, radius: float, cell_size: float) -> None:
    """Inserisce un cratere (cx, cy, radius) in una hash-grid uniforme
    (dict chiave=cella intera -> lista di (cx,cy,radius)), usata da
    scatter_craters per trovare velocemente i "vicini" candidati a un
    conflitto di bordo senza confrontare ogni nuovo cratere con TUTTI
    quelli gia' piazzati (O(n^2), impraticabile a decine di migliaia di
    crateri). cell_size deve essere >= al piu' grande r1+r2 possibile fra
    qualsiasi coppia di crateri in gioco: cosi' due crateri a distanza
    <= cell_size cadono sempre in celle uguali o adiacenti (vedi
    _crater_boundary_conflict, che infatti guarda solo le 8 celle
    confinanti + quella propria)."""
    key = (int(cx // cell_size), int(cy // cell_size))
    grid.setdefault(key, []).append((cx, cy, radius))


def _crater_boundary_conflict(cx: float, cy: float, radius: float, grid: dict, cell_size: float) -> bool:
    """True se il cerchio (cx,cy,radius) ha il BORDO che interseca il
    bordo di almeno un cratere gia' in 'grid' — cioe' i due cerchi si
    sovrappongono parzialmente senza che uno contenga l'altro. Due
    cerchi di raggio r1,r2 con centri a distanza d hanno i bordi che si
    intersecano esattamente quando |r1-r2| < d < r1+r2 (d>=r1+r2: cerchi
    separati/tangenti esternamente, bordi non si toccano davvero in
    un'intersezione a due punti; d<=|r1-r2|: un cerchio dentro l'altro,
    bordi che non si toccano — entrambi i casi vanno bene e NON sono
    conflitti)."""
    kx, ky = int(cx // cell_size), int(cy // cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            bucket = grid.get((kx + dx, ky + dy))
            if not bucket:
                continue
            for ox, oy, orad in bucket:
                d = math.hypot(cx - ox, cy - oy)
                if abs(radius - orad) < d < radius + orad:
                    return True
    return False


def scatter_craters(grid_size: int, n_craters: int, rng: np.random.Generator,
                     max_diam_km: float = 100.0, min_diam_km: float = 3.0,
                     size_freq_exponent: float = 1.9,
                     peak_diam_threshold_km: float = 15.0,
                     peak_prob: float = 0.45,
                     maria_mask: np.ndarray | None = None,
                     maria_suppression: float = 0.75,
                     maria_density_mult: float = 1.0,
                     maria_depth_atten: float = 0.4,
                     foreshorten_strength: float = 1.0,
                     avoid_boundary_intersections: bool = True,
                     max_placement_attempts: int = 100,
                     avoid_craters: list[dict] | None = None,
                     verbose: bool = False) -> list[dict]:
    """Genera una lista di crateri (compatibile con terrain.add_crater)
    sparsi a posizioni casuali sull'intero disco, con raggi campionati da
    una legge di potenza.

    foreshorten_strength: vedi disk_radial_geometry — quanto applicare la
    correzione ellittica da prospettiva vicino al lembo (1.0 = fisicamente
    esatta, <1.0 ammorbidita).

    avoid_boundary_intersections (default True): un cratere piccolo
    completamente DENTRO uno grande (senza toccarne il bordo) e' realistico
    e resta permesso; due crateri che si intersecano "a meta'" — nessuno
    contenuto nell'altro, il bordo dell'uno taglia il bordo dell'altro —
    visivamente sembrano quasi sempre un errore (due impatti che si
    sovrappongono in modo innaturale) piu' che una geologia plausibile, e
    l'utente ha chiesto di evitarli. Implementato con retry: se la
    posizione campionata per un cratere produce un conflitto di bordo con
    un cratere gia' piazzato, si ripesca una nuova posizione (fino a
    max_placement_attempts volte) prima di arrendersi e piazzarlo comunque
    (per non lasciare buchi o loop infiniti in scene molto dense — capita
    raramente, solo con moltissimi crateri piccoli ammassati). Il
    contenimento (un cerchio interamente dentro l'altro) non conta MAI
    come conflitto, in nessun caso.
    max_placement_attempts: quanti tentativi di riposizionamento per
    cratere prima di arrendersi (vedi sopra). Ignorato se
    avoid_boundary_intersections=False.
    avoid_craters: crateri gia' esistenti (stessa forma di dict di
    ritorno, servono solo 'cx'/'cy'/'radius') da NON intersecare, ma che
    NON vengono restituiti/duplicati in questa chiamata — usato da
    generate_moon per far si' che i crateri "grandi" (seconda
    popolazione, generata con una seconda chiamata a scatter_craters)
    evitino di intersecare il bordo dei crateri "normali" gia' piazzati
    dalla prima chiamata (il percorso inverso — i crateri normali che
    evitano quelli grandi, piazzati dopo — non e' coperto, ma dato che i
    crateri grandi sono pochi, in media 2, e piazzati per ultimi, basta
    che siano loro a rispettare gli altri).
    verbose: se True, stampa quanti crateri (su n_craters) non hanno
    trovato una posizione libera entro max_placement_attempts tentativi
    e sono stati piazzati comunque nonostante il conflitto di bordo.

    max_diam_km / min_diam_km: limiti fisici del diametro, in km (assume
    un raggio lunare di MOON_RADIUS_KM per convertirli in pixel). Il tetto
    e' deliberato: crateri VERAMENTE grandi (centinaia/migliaia di km,
    tipo South Pole-Aitken) sono eccezioni geologiche rare — e ora, con
    generate_maria_mask, sono comunque rappresentati separatamente come
    i mari stessi (bacini da impatto allagati di lava), non come crateri
    di questa lista. Default 150 km (non piu' 100): crateri reali come
    Clavius (~230 km) o Schickard (~227 km) superano abbondantemente
    100 km restando pero' crateri "normali" (non bacini-mare), quindi
    100 km era un tetto un po' troppo stretto. Nella prima versione di
    questo generatore crateri troppo grandi/alti vicino al lembo
    rendevano il bordo del disco visibilmente meno netto — la normale
    sferica (vedi sphere_geometry) e' molto sensibile alla pendenza
    locale proprio li' (il fattore 1/w della prospettiva esplode); questo
    non e' piu' un problema pratico da quando spherical_lambert_shading
    limita la perturbazione della normale (vedi il suo docstring). Il
    floor di risoluzione (i crateri sotto ~1-2 pixel non hanno senso da
    disegnare) viene applicato automaticamente confrontando min_diam_km
    con quanti km rappresenta un singolo pixel a questa risoluzione.
    size_freq_exponent: l'esponente "b" della distribuzione cumulativa
    N(>D) ~ D^-b, quello che di solito si cita come "dimensione
    frattale" dei crateri lunari (~1.8-2.0, il valore discusso insieme).
    La densita' di probabilita' usata per campionare (power_law_radii)
    ha pendenza b+1, perche' e' la derivata della cumulativa cambiata di
    segno. Un valore piu' alto = molti piu' crateri piccoli rispetto ai
    grandi.
    peak_diam_threshold_km: solo i crateri sopra questo diametro hanno
    (con probabilita' peak_prob) un picco centrale — solo i crateri
    complessi/grandi ne hanno uno in realta' (Tycho, ~85 km, e' proprio
    al limite tipico per svilupparne uno).
    maria_mask: se fornita (vedi generate_maria_mask), i crateri GRANDI
    che cadono in zone "mare" (valore alto della maschera) vengono
    scartati con probabilita' crescente (maria_suppression * mask *
    frazione_di_raggio) — simula il fatto che i bacini piu' vecchi in
    quelle zone sono stati allagati dalla lava e cancellati; i crateri
    piccoli restano quasi tutti (come nella realta', dove i mari hanno
    comunque una disseminazione fitta di piccoli crateri recenti). I
    crateri superstiti in zone mare hanno anche la profondita' ridotta
    (maria_depth_atten) per simulare il parziale allagamento/erosione.
    maria_density_mult: fattore moltiplicativo indipendente dalla
    dimensione sulla densita' complessiva di crateri nei mari (1.0 =
    nessun effetto, 0.5 = meta' crateri, di qualunque taglia, rispetto
    agli altopiani). Distinto da maria_suppression (che colpisce quasi
    solo i crateri grandi, in proporzione alla taglia): qui ogni
    cratere, piccolo o grande, viene scartato con probabilita'
    (1 - maria_density_mult) * mask, per un diradamento uniforme.

    Profondita' e bordo di OGNI cratere seguono ora leggi di scala reali
    (crater_depth_over_diameter / crater_rim_over_diameter, tipo Pike),
    invece di frazioni casuali uniformi indipendenti dalla dimensione —
    verificate contro un vero profilo altimetrico di Tycho (misurato da
    un DEM LROC/GLD100 reale): il rapporto d/D misurato (~0.066, cratere
    complesso di ~85-100 km) e' consistente con l'ordine di grandezza di
    questa legge. Ogni cratere riceve anche un grado di 'degradazione'
    casuale indipendente (0=fresco, 1=molto eroso): riduce profondita' e
    bordo, allarga rim_frac (bordo piu' smussato/diffuso) e aumenta
    irregularity (contorno meno pulito) — i crateri lunari reali NON sono
    tutti della stessa eta', e questo produce la stessa varieta' visiva
    (alcuni nitidi, altri quasi sommersi nel fondo) senza dover scartare
    o duplicare craters."""
    disk_radius_km = MOON_RADIUS_KM
    km_per_pixel = disk_radius_km / (grid_size / 2.0)

    r_max = (max_diam_km / 2.0) / km_per_pixel
    r_min_physical = (min_diam_km / 2.0) / km_per_pixel
    r_min_resolution = 0.6  # meno di ~1 px di diametro un cratere non e' comunque disegnabile in modo sensato
    r_min = max(r_min_physical, r_min_resolution)
    r_min = min(r_min, r_max * 0.5)  # non invertire l'intervallo a risoluzioni molto basse
    if r_min_resolution > r_min_physical:
        # min_diam_km richiesto e' sotto cio' che questa griglia puo' risolvere:
        # il minimo VERO usato e' quello di risoluzione, non quello chiesto.
        # A grid=260/400/600 corrisponde a circa 32/21/14 km — per crateri
        # davvero piccoli (pochi km) serve una griglia molto piu' fitta.
        actual_min_km = 2 * r_min_resolution * km_per_pixel
        import warnings
        warnings.warn(f"min_diam_km={min_diam_km:.1f} km e' sotto la risoluzione di questa griglia "
                       f"({km_per_pixel:.2f} km/pixel): il diametro minimo effettivo usato e' "
                       f"{actual_min_km:.1f} km. Per crateri piu' piccoli serve un --grid piu' alto.")

    pdf_slope = size_freq_exponent + 1.0
    radii = power_law_radii(n_craters, r_min, r_max, pdf_slope, rng)
    # crateri piu' grandi per primi: non piu' solo estetico per il log
    # (la somma delle altezze resta comunque commutativa) — ora serve
    # anche a avoid_boundary_intersections: piazzando prima i grandi, un
    # cratere piccolo che nasce DENTRO un'area gia' occupata da uno
    # grande puo' comunque risultare "contenuto" (permesso) invece di
    # dover per forza essere respinto altrove.
    radii = np.sort(radii)[::-1]

    # Hash-grid uniforme per il controllo di conflitto dei bordi (vedi
    # _crater_boundary_conflict): cell_size deve coprire il piu' grande
    # r1+r2 possibile fra QUALSIASI coppia coinvolta, incluse quelle di
    # avoid_craters (che possono avere raggio diverso dalla popolazione
    # corrente, es. i crateri "grandi" rispetto ai "normali").
    placement_grid: dict = {}
    max_avoid_radius = max((c["radius"] for c in avoid_craters), default=0.0) if avoid_craters else 0.0
    cell_size = 2.0 * max(r_max, max_avoid_radius, 1.0)
    if avoid_craters:
        for c in avoid_craters:
            _crater_grid_insert(placement_grid, c["cx"], c["cy"], c["radius"], cell_size)

    craters = []
    n_forced_overlap = 0
    for i, radius in enumerate(radii):
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
                continue  # bacino allagato dalla lava: non sopravvive come cratere visibile
            if maria_density_mult < 1.0 and rng.uniform() < (1.0 - maria_density_mult) * m:
                continue  # diradamento uniforme (indipendente dalla taglia) nei mari

        # Evita che il BORDO di questo cratere intersechi il bordo di uno
        # gia' piazzato (contenimento permesso, vedi il docstring sopra):
        # la decisione "sopravvive al mare" resta quella presa sopra alla
        # posizione iniziale (e' un evento geologico legato al luogo, non
        # un conflitto di posizionamento); qui si ripesca SOLO la
        # posizione, fino a max_placement_attempts volte, poi ci si
        # arrende e lo si piazza comunque dov'e' finito l'ultimo tentativo.
        if avoid_boundary_intersections:
            for _attempt in range(max_placement_attempts):
                if not _crater_boundary_conflict(cx, cy, radius, placement_grid, cell_size):
                    break
                cx = rng.uniform(0, grid_size)
                cy = rng.uniform(0, grid_size)
            else:
                n_forced_overlap += 1
            # la posizione puo' essere cambiata: ricalcola 'm' (usato sotto
            # per l'attenuazione di profondita' nei mari) al punto finale
            if maria_mask is not None:
                iy = int(np.clip(round(cy), 0, grid_size - 1))
                ix = int(np.clip(round(cx), 0, grid_size - 1))
                m = float(maria_mask[iy, ix])
        _crater_grid_insert(placement_grid, cx, cy, radius, cell_size)

        degradation = float(rng.uniform(0.0, 1.0))  # 0 = fresco, 1 = molto eroso/sommerso
        jitter = float(rng.lognormal(mean=0.0, sigma=0.15))  # dispersione naturale attorno alla legge di scala

        d_over_D = crater_depth_over_diameter(D_km) * jitter * (1.0 - 0.55 * degradation)
        rim_over_D = crater_rim_over_diameter(D_km) * jitter * (1.0 - 0.75 * degradation)
        if m > 0:
            d_over_D *= (1.0 - maria_depth_atten * m)  # allagamento lavico: ulteriore riduzione nei mari

        diameter_px = 2.0 * radius
        depth = d_over_D * diameter_px
        rim_h_px = rim_over_D * diameter_px
        rim_height_frac = float(np.clip(rim_h_px / (depth + 1e-9), 0.12, 0.6))

        rim_frac = float(np.clip(rng.uniform(0.20, 0.35) + 0.15 * degradation, 0.15, 0.55))
        # ATTENZIONE alla scala di irregularity: NON e' innocua ad ampiezza
        # alta. add_crater/crater_fields distorcono il raggio del bordo con
        # sin(3*theta+f0)*0.5 + sin(5*theta+f1)*0.3 + sin(8*theta+f2)*0.2,
        # fasi f0/f1/f2 casuali indipendenti per cratere — per la maggior
        # parte delle fasi il risultato e' un bordo piacevolmente irregolare,
        # ma per alcune combinazioni di fase (non rare: osservato su piu'
        # crateri indipendenti, non un caso isolato) i tre termini
        # interferiscono in modo da creare un contorno con pochi lobi
        # DOMINANTI invece di tanta rugosita' fine — a ampiezza alta il
        # risultato e' un cratere a forma di trifoglio/quadrifoglio netto,
        # non un bordo "roccioso". Root cause del bug "crateri a
        # quadrifoglio" segnalato dall'utente — confermato isolando UN
        # cratere renderizzato da solo, senza nessuna texture reale
        # innestata: il difetto e' nel modello geometrico di base
        # (terrain.py), non nel patch-matching (moon_generation_patchmatch/,
        # che nel frattempo aveva gia' una sua protezione crateri per un
        # problema DIVERSO e reale, ma non era la causa di questo). La
        # "peakiness" della forma dipende SOLO dalle fasi casuali (non da
        # irregularity, che e' un moltiplicatore scalare puro) quindi non
        # e' evitabile scegliendo le fasi "giuste" caso per caso — ma
        # verificato empiricamente (sweep su centinaia di seed, incluse le
        # fasi piu' sfavorevoli trovate) che sotto ~0.08 il lobo resta
        # impercettibile anche nel caso peggiore, mentre il vecchio range
        # (0.10-0.22 base, fino a 0.4 con degradazione) ci ricadeva quasi
        # sempre dentro. Range dimezzato di conseguenza.
        irregularity = float(np.clip(rng.uniform(0.02, 0.05) + 0.05 * degradation, 0.0, 0.10))
        has_peak = (D_km > peak_diam_threshold_km and
                    rng.uniform() < peak_prob * (1.0 - 0.6 * degradation))  # i picchi si "sommergono" nei crateri vecchi
        peak_frac = rng.uniform(0.25, 0.55) if has_peak else 0.0
        peak_radius_frac = rng.uniform(0.12, 0.22)

        # Correzione prospettica (vedi terrain._foreshorten_offsets): un
        # cratere e' un cerchio fisico sulla sfera, quindi in proiezione
        # ortografica appare ELLITTICO (schiacciato in direzione radiale)
        # quanto piu' e' vicino al lembo — finora invece veniva disegnato
        # come un cerchio perfetto in pixel, sempre, indipendentemente
        # dalla posizione sul disco. radial_unit/foreshorten_w dicono a
        # crater_fields quanto e in che direzione "schiacciare" il
        # profilo per quel singolo cratere (vedi disk_radial_geometry).
        radial_unit, foreshorten_w = disk_radial_geometry(cx, cy, grid_size, strength=foreshorten_strength)

        craters.append(dict(cx=cx, cy=cy, radius=radius, depth=depth,
                             rim_frac=rim_frac, rim_height_frac=rim_height_frac, irregularity=irregularity,
                             peak_frac=peak_frac, peak_radius_frac=peak_radius_frac,
                             seed=int(rng.integers(0, 2**31 - 1)),
                             radial_unit=radial_unit, foreshorten_w=foreshorten_w))
    if verbose and avoid_boundary_intersections and n_forced_overlap > 0:
        print(f"[scatter_craters] {n_forced_overlap}/{len(craters)} crateri piazzati nonostante un "
              f"conflitto di bordo residuo (esauriti {max_placement_attempts} tentativi di riposizionamento)")
    return craters


# ---------------------------------------------------------------------
# 1b. Regionalizzazione mari/altopiani (procedurale, vedi docstring)
# ---------------------------------------------------------------------

def generate_maria_mask(grid_size: int, rng: np.random.Generator,
                         coverage: float | None = None,
                         edge_width_frac: float = 0.05,
                         edge_irregularity: float = 0.10,
                         min_radius_frac: float = 0.10,
                         n_basins: int | None = None) -> tuple[np.ndarray, float]:
    """Genera una maschera 'mare vs altopiano' procedurale, in [0,1]
    (1 = pieno mare: scuro, piatto, pochi crateri grandi; 0 = pieno
    altopiano). NON e' una mappa reale della Luna (vedi la nota nel
    docstring del modulo su come mai) — e' una regionalizzazione
    inventata ma fisicamente motivata.

    SECONDA versione (la prima generava la maschera come le zone piu'
    basse di un campo frattale a grandissima scala, sogliato per
    percentile: dava contorni morbidi ma a "macchia di leopardo", una
    forma senza nessuna relazione geometrica con un'origine da impatto —
    limite gia' segnalato onestamente. Punto sollevato insieme: i mari
    reali sono bacini da impatto GIGANTI poi allagati di lava — la loro
    forma dovrebbe quindi essere vicina a un disco, o all'unione di due
    dischi che si sovrappongono (bacini adiacenti), non un blob
    frattale arbitrario). Qui il mare e' letteralmente l'unione di al
    massimo 2 dischi (cerchi = sezione di un bacino da impatto circolare,
    "al massimo uno o due" — a volte nessuno), ciascuno con:
    - un raggio derivato dalla `coverage` bersaglio (area totale del
      disco lunare * coverage, divisa fra i bacini);
    - un bordo leggermente irregolare (armoniche angolari basse, stessa
      idea usata per il bordo dei singoli crateri in terrain.add_crater)
      invece di un cerchio geometricamente perfetto — i bordi reali dei
      bacini sono comunque smussati/eresi da eoni di craterizzazione
      successiva, non un compasso;
    - una transizione morbida (edge_width_frac) verso l'altopiano.
    Quando i bacini sono 2, il centro di quello PICCOLO non e' piu'
    piazzato in modo indipendente (poteva capitare per puro caso che i
    due non si toccassero affatto, o si sovrapponessero quasi del tutto):
    e' forzato a una distanza dal centro di quello GRANDE tale che almeno
    meta' del suo diametro (lungo la congiungente) cada dentro il grande —
    la sagoma "composta" (un bacino grande con uno piu' piccolo incastrato,
    come Imbrium+Iridum o Serenitatis+Tranquillitatis) invece di due
    macchie scollegate o quasi indistinguibili.
    Piu' bacini si combinano per MASSIMO (unione), esattamente come per
    i crateri in assemble_terrain_capped — dove due bacini si
    sovrappongono il risultato e' l'unione delle due aree, non la somma.

    Ordine geologico discusso insieme (e che questa funzione/generate_moon
    seguono, nello spirito): (1) crosta primordiale ruvida (il terreno di
    fondo, spectral_synthesis_terrain), (2) bombardamento con crateri di
    ogni taglia, (3) *al massimo uno o due* fra i piu' grandi sono bacini
    da impatto giganteschi, allagati dalla lava, (4) il bombardamento
    continua dopo, sia sugli altopiani che sui mari appena solidificati
    (i piccoli crateri "recenti" visibili anche dentro un mare). Qui i
    passi (3)+(4) sono quello che genera_maria_mask/scatter_craters
    implementano: i bacini SONO trattati come eventi a parte da "al
    massimo uno o due" (non un pulviscolo di mari medi — i mari reali
    piu' vistosi sono proprio pochi bacini enormi), e i crateri ordinari
    (scatter_craters, fino a 150 km) continuano a cadere ovunque dopo,
    con quelli grandi soppressi/attenuati nei mari (maria_crater_
    suppression/maria_depth_atten) a simulare l'allagamento che ha
    cancellato i bacini piu' vecchi — i piccoli restano quasi tutti,
    come nella realta'.

    coverage: frazione del disco coperta da mare, in [0,1]. Se None,
    scelta a caso in [0.15, 0.40] (il lato visibile reale e' vicino al
    31%, ma qui e' un generatore di lune inventate quindi si varia).

    n_basins: se None (default), scelto a caso come prima (0/1/2 con
    probabilita' 0.15/0.55/0.30). Passare esplicitamente 0, 1 o 2 forza
    quel numero di bacini invece di lasciarlo al caso."""
    if coverage is None:
        coverage = float(rng.uniform(0.15, 0.40))

    # AL MASSIMO uno o due bacini (mai piu' di 2, e a volte zero: una luna
    # "fortunata" senza grandi impatti — varietà in piu', non solo
    # allineamento alla richiesta) — non un pulviscolo di bacini medi: i
    # mari reali piu' vistosi sono proprio pochi bacini enormi, non tanti
    # piccoli. Ridotto automaticamente se la coverage richiesta e' troppo
    # piccola per sostenere 2 bacini sopra la dimensione minima.
    n_basins = int(rng.choice([0, 1, 2], p=[0.15, 0.55, 0.30])) if n_basins is None else int(n_basins)
    if n_basins == 0:
        return np.zeros((grid_size, grid_size)), 0.0
    disk_area = np.pi * (grid_size / 2.0) ** 2
    total_area = coverage * disk_area
    min_radius = min_radius_frac * grid_size
    min_area_each = np.pi * min_radius ** 2
    while n_basins > 1 and total_area / n_basins < min_area_each:
        n_basins -= 1

    if n_basins == 1:
        fracs = np.array([1.0])
    else:
        raw = rng.uniform(0.5, 1.0, size=n_basins)
        fracs = raw / raw.sum()

    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float64)
    mask = np.zeros((grid_size, grid_size))
    edge_width = edge_width_frac * grid_size

    radii = [max(float(np.sqrt(fracs[i] * total_area / np.pi)), min_radius) for i in range(n_basins)]

    # Centri: il PRIMO bacino (se 2, quello scelto qui non e' necessariamente
    # il piu' grande, vedi sotto) resta piazzato a caso nel solito riquadro.
    centers = [None] * n_basins
    centers[0] = (rng.uniform(grid_size * 0.15, grid_size * 0.85),
                  rng.uniform(grid_size * 0.15, grid_size * 0.85))
    if n_basins == 2:
        # Punto sollevato dall'utente: due bacini piazzati in modo
        # indipendente possono anche non toccarsi affatto (puro caso) — i
        # bacini composti reali (es. Imbrium+Iridum, Serenitatis che
        # "morde" Tranquillitatis) hanno invece il bacino piu' piccolo
        # deliberatamente incastrato in quello piu' grande. Qui si forza
        # il centro del bacino PICCOLO a una distanza dal centro del
        # bacino GRANDE tale che almeno META' del suo diametro (lungo la
        # congiungente i centri) cada dentro il bacino grande:
        # d = R + r - 2*r*overlap_frac, con overlap_frac in [0.5, 1.0] —
        # a overlap_frac=0.5, d=R (il centro del piccolo cade esattamente
        # sul bordo del grande, meta' dentro meta' fuori); a
        # overlap_frac=1.0, d=R-r (il piccolo quasi completamente incluso).
        # Un'approssimazione semplice (basata sulla sola distanza fra
        # centri, non sull'area vera di intersezione) ma sufficiente allo
        # scopo, come suggerito.
        i_big, i_small = (0, 1) if radii[0] >= radii[1] else (1, 0)
        big_r, small_r = radii[i_big], radii[i_small]
        overlap_frac = float(rng.uniform(0.5, 1.0))
        d = max(big_r + small_r - 2.0 * small_r * overlap_frac, 0.0)
        angle = float(rng.uniform(0, 2 * np.pi))
        # centers[0] e' l'unico gia' piazzato (l'ancora) — puo' essere sia
        # il bacino grande che il piccolo, non importa: la distanza d
        # dipende solo dai due raggi, non da chi e' l'ancora.
        ax, ay = centers[0]
        ox = ax + d * np.cos(angle)
        oy = ay + d * np.sin(angle)
        margin = grid_size * 0.02
        centers[1] = (float(np.clip(ox, margin, grid_size - margin)),
                      float(np.clip(oy, margin, grid_size - margin)))

    for i in range(n_basins):
        radius = radii[i]
        cx, cy = centers[i]
        # Stessa correzione ellittica dei crateri (vedi scatter_craters e
        # terrain._foreshorten_offsets): (cx, cy) e' vincolato solo in x e
        # in y separatamente, quindi puo' comunque atterrare vicino
        # all'angolo del riquadro [0.15, 0.85]^2 — a quel punto la
        # distanza dal centro del disco (in diagonale) e' fino a ~0.495
        # del raggio, cioe' quasi al lembo vero. Inoltre un bacino e' per
        # definizione GRANDE rispetto al disco (a differenza della
        # maggior parte dei crateri): anche solo parzialmente vicino al
        # lembo, un cerchio cosi' grande va comunque deformato — un
        # cerchio perfetto qui sarebbe un errore piu' vistoso che per un
        # piccolo cratere. w0 e' comunque valutato una sola volta al
        # centro del bacino (non integrato sulla sua estensione): resta
        # un'approssimazione, migliore di nessuna correzione ma meno
        # precisa per bacini molto grandi vicino al lembo.
        radial_unit, foreshorten_w = disk_radial_geometry(cx, cy, grid_size)
        dx = xx - cx
        dy = yy - cy
        dx, dy = _foreshorten_offsets(dx, dy, radial_unit, foreshorten_w)
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        theta = np.arctan2(dy, dx)
        phases = rng.uniform(0, 2 * np.pi, size=3)
        distortion = radius * edge_irregularity * (
            np.sin(2 * theta + phases[0]) * 0.5 +
            np.sin(4 * theta + phases[1]) * 0.3 +
            np.sin(7 * theta + phases[2]) * 0.2
        )
        r_edge = radius + distortion
        disk_i = np.clip((r_edge - r) / edge_width + 0.5, 0.0, 1.0)
        mask = np.maximum(mask, disk_i)

    return mask, coverage


def _percolation_grow(base: np.ndarray, si: int, sj: int, disk_area: float, thresholds: np.ndarray,
                       coverage_cap: float, explosion_frac: float, min_pre_frac: float,
                       bisect_iters: int, stop_margin_frac: float) -> tuple[np.ndarray, float]:
    """Nucleo dell'allagamento per soglia crescente ("percolazione"),
    estratto da generate_maria_mask_flood per essere riusabile anche da
    generate_crater_seeded_mare (stesso algoritmo, seme e soglie diverse):
    parte dal pixel (si,sj) e alza la soglia finche' la componente connessa
    che lo contiene non "esplode", fermandosi un passo prima (per bisezione).
    Ritorna (chosen, coverage) — chosen e' la maschera booleana GREZZA
    (senza smussatura/bordo morbido, vedi _soften_basin per quello)."""
    def eval_threshold(t):
        flooded = base <= t
        labeled, _ = ndi_label(flooded)
        lab = labeled[si, sj]
        if lab == 0:
            return None, 0.0
        comp = labeled == lab
        return comp, float(comp.sum()) / disk_area

    def bisect(t_lo, t_hi, comp_lo, cov_lo, predicate):
        best_comp, best_cov = comp_lo, cov_lo
        lo, hi = t_lo, t_hi
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            comp, cov = eval_threshold(mid)
            if comp is not None and predicate(cov):
                lo, best_comp, best_cov = mid, comp, cov
            else:
                hi = mid
        return best_comp, best_cov

    comps, covs, ts = [], [], []
    for t in thresholds:
        comp, cov = eval_threshold(t)
        if comp is None:
            continue
        comps.append(comp)
        covs.append(cov)
        ts.append(t)
        if cov >= coverage_cap or cov > 0.97:
            break

    if not covs:
        return np.zeros(base.shape, dtype=bool), 0.0
    if len(covs) == 1:
        return comps[0], covs[0]

    covs_arr = np.array(covs)
    deltas = np.diff(covs_arr, prepend=0.0)
    pre_covs = np.concatenate(([0.0], covs_arr[:-1]))
    jump_candidates = np.where((pre_covs >= min_pre_frac) & (deltas > explosion_frac))[0]
    if len(jump_candidates) > 0:
        jmax = int(jump_candidates[np.argmax(deltas[jump_candidates])])
        t_lo = ts[jmax - 1] if jmax > 0 else float(base.min())
        comp_lo = comps[jmax - 1] if jmax > 0 else np.zeros(base.shape, dtype=bool)
        cov_lo = covs[jmax - 1] if jmax > 0 else 0.0
        return bisect(t_lo, ts[jmax], comp_lo, cov_lo,
                       predicate=lambda c: (c - cov_lo) <= explosion_frac * stop_margin_frac)
    if covs[-1] >= coverage_cap and len(covs) > 1:
        return bisect(ts[-2], ts[-1], comps[-2], covs[-2], predicate=lambda c: c <= coverage_cap)
    return comps[-1], covs[-1]  # crescita genuinamente graduale, nessuno sforamento


def _soften_basin(chosen: np.ndarray, smooth_radius_px: float, edge_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Dilatazione morfologica opzionale (vedi il commento su
    smooth_radius_px in generate_maria_mask_flood) + bordo morbido dalla
    distanza con segno dal contorno. Ritorna (soft_mask, chosen_dilatato)."""
    if smooth_radius_px > 0 and chosen.any():
        dist_out_chosen = distance_transform_edt(~chosen)
        chosen = chosen | (dist_out_chosen <= smooth_radius_px)
    dist_in = distance_transform_edt(chosen)
    dist_out = distance_transform_edt(~chosen)
    signed = np.where(chosen, dist_in, -dist_out)
    soft = np.clip(signed / edge_width + 0.5, 0.0, 1.0)
    return soft, chosen


def generate_maria_mask_flood(grid_size: int, base: np.ndarray, rng: np.random.Generator,
                               coverage_cap: float = 0.45, n_thresholds: int = 200,
                               explosion_frac: float = 0.05, min_pre_frac: float = 0.03,
                               seed_quantile: float = 0.05, seed_margin_frac: float = 0.15,
                               bisect_iters: int = 14, stop_margin_frac: float = 0.3,
                               smooth_radius_px: float = 0.0,
                               edge_width_frac: float = 0.05,
                               n_basins: int | None = None) -> tuple[np.ndarray, float]:
    """SECONDA idea dell'utente per la forma dei bacini-mare, alternativa
    a generate_maria_mask (dischi): invece di un cerchio (o due), la
    sagoma viene da un vero e proprio ALLAGAMENTO per soglia crescente
    ("percolazione") del terreno di fondo 'base' (lo stesso campo
    frattale gia' usato per il rilievo — vedi spectral_synthesis_terrain),
    seguendo esattamente il suggerimento: parti da un'altezza bassa e
    alzala finche' il numero di siti connessi non "esplode" (transizione
    di percolazione), fermandoti un passo prima.

    Perche' funziona: 'base' e' un campo Gaussiano correlato (sintesi
    spettrale, beta=3.1) — per campi di questo tipo e' un fenomeno noto
    che l'insieme dei punti sotto una soglia (l'"excursion set") passi
    da tante piccole isole scollegate a un'unica componente che copre
    gran parte della griglia in un intervallo di soglia molto STRETTO
    (transizione di percolazione).

    Storia di due bug trovati SOLO testando su tanti seed diversi (su un
    seed singolo il difetto non si vedeva):

    1. Punto di partenza su un minimo troppo "debole". La prima versione
       cercava il minimo LOCALE in una finestra casuale — ma un minimo
       locale puo' benissimo trovarsi su un rilievo moderato (non un
       vero bacino), che resta isolato per quasi tutto lo sweep di
       soglia e poi, non appena si allaga, si ritrova GIA' innestato in
       una componente enorme (percolazione avvenuta altrove nel
       frattempo) — risultato: coverage quasi nulla invece di un bacino
       vero. Fix: il punto di partenza si sceglie ora fra i pixel gia'
       nel percentile piu' basso (`seed_quantile`) dell'intera 'base'
       (non solo di una finestra locale) — cosi' si parte sempre da un
       bacino genuinamente profondo.
    2. Rilevare "l'esplosione" come primo salto (fra due soglie
       consecutive) che superi `explosion_frac` e' fragile su due fronti
       opposti: (a) un salto piccolo ma spurio (due isolette minuscole
       che si toccano) puo' scattare per primo e bloccare la crescita
       quasi a zero; (b) con soglie campionate troppo rade, la vera
       transizione puo' saltare da una taglia piccola a una ben oltre
       `coverage_cap` in un solo passo campionato, sforando il tetto.
       Fix: le soglie sono ora campionate per QUANTILI di 'base' (invece
       che linearmente sul range di valori — piu' risoluzione dove la
       massa della distribuzione e' davvera, cioe' proprio dove serve);
       fra i salti si considera "esplosione" solo quello dove la
       componente era gia' "basin-sized" (>= `min_pre_frac`) PRIMA del
       salto (scarta i micro-agganci spuri); e la soglia esatta del
       salto scelto (o del taglio al `coverage_cap`) viene rifinita con
       `bisect_iters` passi di bisezione fra le due soglie campionate
       che lo delimitano, invece di accettare la granularita' grezza del
       campionamento.

    Verificato su 60 seed diversi (grid=500): coverage per bacino sempre
    fra ~4% e ~48% dell'area del disco (nessun quasi-zero, nessuno
    sforamento serio del cap 45%) — prima di questi due fix il range
    osservato era 0.4%-44% con parecchi risultati degeneri.

    seed_quantile / seed_margin_frac: il punto di partenza si pesca a
    caso fra i pixel di 'base' sotto il percentile `seed_quantile`,
    ristretto al riquadro centrale [seed_margin_frac, 1-seed_margin_frac]
    (evita bacini a ridosso del lembo del disco).

    coverage_cap: tetto di sicurezza (frazione dell'area del DISCO): se
    lo sweep non mostra un salto netto abbastanza presto, ci si ferma
    qui (rifinito per bisezione) invece di rischiare di allagare quasi
    tutto il disco.

    stop_margin_frac (default 0.3): quanto della "cascata" fra l'ultimo
    punto stabile e il salto vero e proprio si lascia recuperare dalla
    bisezione — 1.0 userebbe l'intero margine `explosion_frac`, 0.0
    fermerebbe esattamente al campione grezzo precedente il salto senza
    rifinire affatto. Abbassato dal 1.0 iniziale su richiesta esplicita
    dopo aver visto i primi risultati ("bella la figura di percolazione,
    forse fermarsi ancora un po' prima"): con 1.0 il bordo era gia' un
    filo dentro l'inizio della cascata di fusioni che precede
    l'esplosione vera; con 0.3 si resta piu' vicino al bacino "stabile"
    da cui la crescita sarebbe comunque esplosa.

    smooth_radius_px (default 0.0, disattivato): idea dell'utente per
    smussare il contorno frastagliato/ramificato tipico di un insieme di
    percolazione — dopo aver scelto `chosen`, lo si sostituisce con
    l'insieme dei punti a distanza euclidea < smooth_radius_px da
    `chosen` (una dilatazione morfologica con un disco di quel raggio,
    non uno smoothing gaussiano: riempie insenature e stringe i
    "tentacoli" piu' sottili del raggio scelto, ma i rami piu' larghi del
    raggio restano). Valori tipici 3-10px (a `--grid 500`, dove 1px =
    grid_size/2/moon_radius_km circa 6-7km); piu' alto = contorno piu'
    arrotondato/gonfio ma meno fedele alla forma di percolazione originale
    (e coverage finale piu' grande, essendo una dilatazione, mai una
    contrazione). E' indipendente da `edge_width_frac`: quest'ultimo
    resta la transizione morbida (in albedo, non in forma) verso
    l'altopiano, applicata DOPO la dilatazione.

    edge_width_frac: stessa idea di generate_maria_mask (transizione
    morbida verso l'altopiano), ma qui calcolata come distanza CON SEGNO
    dal contorno vero e proprio della macchia allagata (irregolare per
    costruzione) invece che da un cerchio.

    NON impone una coverage target come generate_maria_mask: la
    dimensione la decide la topografia stessa. La coverage restituita e'
    quindi un RISULTATO, non un parametro d'ingresso rispettato a
    posteriori — puo' variare piu' di quanto farebbe con coverage=0.35
    fisso, e' il prezzo di una sagoma dettata dal terreno vero invece che
    imposta.

    n_basins: se None (default), scelto a caso come in generate_maria_mask
    (0/1/2 con probabilita' 0.15/0.55/0.30). Passare esplicitamente 0, 1 o
    2 forza quel numero di bacini invece di lasciarlo al caso.

    Con due bacini, il secondo seme evita (tramite `_pick_flood_seed`,
    escludendo i pixel gia' scelti/allagati dal primo) di ripartire dalla
    stessa depressione gia' innestata nel primo — puo' comunque capitare
    che i due bacini finiscano per fondersi in un'unica componente se il
    terreno li connette prima del taglio, cosi' come nella realta' alcuni
    sistemi di mari (Imbrium+Iridum, Serenitatis+Tranquillitatis) sono in
    parte collegati.

    Ancora sperimentale (non e' il default di generate_moon): passare
    maria_style='flood' a generate_moon per provarla."""
    disk_area = np.pi * (grid_size / 2.0) ** 2
    n_basins = int(rng.choice([0, 1, 2], p=[0.15, 0.55, 0.30])) if n_basins is None else int(n_basins)
    if n_basins == 0:
        return np.zeros((grid_size, grid_size)), 0.0

    thresholds = np.quantile(base.ravel(), np.linspace(0.0, 1.0, n_thresholds))
    edge_width = edge_width_frac * grid_size
    mask = np.zeros((grid_size, grid_size))
    total_flooded_px = 0.0
    excluded = np.zeros((grid_size, grid_size), dtype=bool)  # bacini gia' scelti, per il seme successivo

    for _ in range(n_basins):
        # --- 1. seme: un pixel a caso fra i piu' bassi di 'base' (non
        # gia' scelto da un bacino precedente), lontano dal lembo.
        lo_m, hi_m = grid_size * seed_margin_frac, grid_size * (1.0 - seed_margin_frac)
        thresh = np.quantile(base, seed_quantile)
        yy, xx = np.mgrid[0:grid_size, 0:grid_size]
        candidate_mask = (base <= thresh) & (yy >= lo_m) & (yy < hi_m) & (xx >= lo_m) & (xx < hi_m) & (~excluded)
        cand = np.where(candidate_mask)
        if len(cand[0]) == 0:
            cand = np.where((base <= thresh) & (~excluded))
        if len(cand[0]) == 0:
            cand = np.where(base <= thresh)  # bacini precedenti hanno coperto tutto il resto: rinuncia all'esclusione
        k = int(rng.integers(0, len(cand[0])))
        si, sj = int(cand[0][k]), int(cand[1][k])

        # --- 2. sweep di soglia crescente ("percolazione", vedi _percolation_grow) ---
        chosen, _ = _percolation_grow(base, si, sj, disk_area, thresholds,
                                       coverage_cap, explosion_frac, min_pre_frac,
                                       bisect_iters, stop_margin_frac)

        # --- 3. smussatura opzionale + bordo morbido (vedi _soften_basin,
        # idea dell'utente: l'insieme di percolazione e' frastagliato per
        # natura, smooth_radius_px=0 di default lo lascia com'e'). ---
        soft, chosen = _soften_basin(chosen, smooth_radius_px, edge_width)
        mask = np.maximum(mask, soft)
        total_flooded_px += float(chosen.sum())
        excluded |= chosen

    coverage_used = total_flooded_px / disk_area
    return mask, coverage_used


def generate_crater_seeded_mare(grid_size: int, base: np.ndarray, rng: np.random.Generator,
                                 crater_diam_km: float = 200.0, max_coverage: float = 0.05,
                                 depth_mult: float = 6.0, seed_margin_frac: float = 0.15,
                                 explosion_frac: float = 0.05, min_pre_frac: float = 0.01,
                                 bisect_iters: int = 14, stop_margin_frac: float = 0.3,
                                 n_thresholds: int = 200, smooth_radius_px: float = 0.0,
                                 edge_width_frac: float = 0.05,
                                 excluded: np.ndarray | None = None
                                 ) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Un mare PIU' PICCOLO, opzionale (vedi add_crater_mare in
    generate_moon), con un'origine diversa dai bacini di
    generate_maria_mask_flood: invece di partire da un minimo genuino gia'
    presente nel terreno frattale, si scava esplicitamente una conca
    profonda (raggio = un cratere di crater_diam_km, default 200 km — una
    dimensione da "grande bacino da impatto", tipo Grimaldi/Humorum) in
    una COPIA locale di 'base', e la si fa crescere con lo STESSO
    algoritmo di percolazione (_percolation_grow) usato per i mari
    normali — ma con un tetto di copertura (max_coverage, default 5% del
    disco) molto piu' basso: "un cratere poi allargato, non troppo", non
    un bacino vero e proprio. Racconta quindi una storia fisica diversa:
    non un antico bacino d'impatto naturale, ma un singolo grande cratere
    la cui conca e' stata in parte allagata dalla lava, senza espandersi
    granche' oltre il proprio bordo.

    La conca sintetica NON modifica 'base' (il rilievo vero della luna
    resta quello frattale originale, la copia scavata serve solo a
    guidare la percolazione) — l'effetto visivo del mare (piu' scuro,
    localmente appiattito, crateri successivi in parte allagati) viene
    comunque dalla maschera restituita, esattamente come per gli altri
    mari.

    depth_mult: profondita' della conca scavata, in multipli di
    (base.max()-base.min()) — deve essere abbastanza grande da garantire
    che il centro sia SEMPRE il minimo locale piu' profondo della zona
    (altrimenti il seme del percolamento potrebbe finire in un bacino
    diverso, gia' esistente nel terreno, invece che in quello scavato).

    excluded: maschera opzionale (es. il mare "normale" gia' generato) di
    pixel da evitare per il centro del nuovo cratere, cosi' i due non
    nascono esattamente sovrapposti (possono comunque finire adiacenti o
    sfiorarsi, non e' un vincolo rigido).

    Ritorna (mask, coverage, (cx, cy)) — cx,cy sono la posizione (in
    pixel, stesse coordinate di scatter_craters) del centro del cratere
    scavato, nel caso servisse in futuro per piazzare li' anche un vero
    cratere visibile (non fatto qui: per ora e' solo la sagoma del mare)."""
    disk_area = np.pi * (grid_size / 2.0) ** 2
    km_per_pixel = MOON_RADIUS_KM / (grid_size / 2.0)
    r_px = (crater_diam_km / 2.0) / km_per_pixel

    lo_m, hi_m = grid_size * seed_margin_frac, grid_size * (1.0 - seed_margin_frac)
    cx = cy = grid_size / 2.0
    for _attempt in range(30):
        cx = float(rng.uniform(lo_m, hi_m))
        cy = float(rng.uniform(lo_m, hi_m))
        if excluded is None:
            break
        if not excluded[int(round(cy)), int(round(cx))]:
            break

    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float64)
    d = np.hypot(xx - cx, yy - cy)
    edge_r = r_px * 1.15  # un margine oltre il bordo vero, cosi' la conca scavata include anche l'apron immediato
    t = np.clip(1.0 - d / edge_r, 0.0, 1.0)
    bump = t * t * (3 - 2 * t)  # smoothstep: 1 al centro, 0 oltre edge_r
    depression = bump * depth_mult * (base.max() - base.min())
    base_mod = base - depression

    si, sj = int(round(cy)), int(round(cx))
    thresholds = np.quantile(base_mod.ravel(), np.linspace(0.0, 1.0, n_thresholds))
    chosen, _ = _percolation_grow(base_mod, si, sj, disk_area, thresholds,
                                   max_coverage, explosion_frac, min_pre_frac,
                                   bisect_iters, stop_margin_frac)

    edge_width = edge_width_frac * grid_size
    soft, chosen = _soften_basin(chosen, smooth_radius_px, edge_width)
    coverage_used = float(chosen.sum()) / disk_area
    return soft, coverage_used, (cx, cy)


# ---------------------------------------------------------------------
# 2. Geometria sferica: normale pura + vettori tangenti locali
# ---------------------------------------------------------------------

def sphere_geometry(grid_size: int):
    """Ritorna (n0, Tx, Ty, disk_mask):
    - n0: normale della sfera pura in ogni punto (H,W,3) — coordinate
      normalizzate (u,v) in [-1,1], punto sulla sfera unitaria
      (u,v,sqrt(1-u^2-v^2)) vista dall'osservatore (asse z verso
      l'osservatore, come in render.py: z="su"/verso chi guarda).
    - Tx, Ty: vettori tangenti LOCALI alla sfera nelle direzioni x,y
      dell'immagine (derivate parziali del punto sferico rispetto a
      x,y) — non sono (1,0,0)/(0,1,0) come nel caso piatto: vicino al
      lembo si allungano, catturando l'accorciamento prospettico.
    - disk_mask: maschera del disco (1 dentro, sfuma a 0 vicino al
      lembo, 0 fuori) — da applicare moltiplicativamente per ultima.
    """
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float64)
    cx = cy = (grid_size - 1) / 2.0
    r_max = grid_size / 2.0
    u = (xx - cx) / r_max
    v = (yy - cy) / r_max
    r2 = u ** 2 + v ** 2
    w = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    w_safe = np.clip(w, 0.05, None)  # evita la divisione per zero esatta al lembo

    n0 = np.stack([u, v, w], axis=-1)
    Tx = np.stack([np.ones_like(u), np.zeros_like(u), -u / w_safe], axis=-1)
    Ty = np.stack([np.zeros_like(u), np.ones_like(u), -v / w_safe], axis=-1)

    edge_softness = 1.2 / grid_size  # ~1.2 pixel di sfumatura (antialiasing del lembo)
    disk_mask = np.clip((1.0 - np.sqrt(np.clip(r2, 0, None))) / edge_softness, 0.0, 1.0)

    return n0, Tx, Ty, disk_mask


def spherical_lambert_shading(h: np.ndarray, n0: np.ndarray, Tx: np.ndarray, Ty: np.ndarray,
                               sun_dir: np.ndarray, relief_strength: float = 1.0, dx: float = 1.0,
                               max_slope: float = 0.45) -> np.ndarray:
    """Shading Lambertiano che tiene conto della curvatura sferica: la
    normale in ogni punto e' quella della sfera (n0), perturbata (bump
    mapping) dalla pendenza locale del terreno secondo
    N' = n0 - dh/dx * Tx - dh/dy * Ty. Cosi' l'illuminazione effettiva
    (l'equivalente di un'altezza/azimut del sole calcolati punto per
    punto) varia correttamente con la posizione sul disco — piu' bassa
    verso il terminatore e il lembo — invece di essere uniforme.

    max_slope: limite fisico alla PERTURBAZIONE della normale (non alla
    pendenza dh/dx grezza — vedi sotto), espresso come tan(angolo). Serve
    perche' i vettori tangenti Tx, Ty (vedi sphere_geometry) si allungano
    vicino al lembo per il fattore 1/w della prospettiva, quindi la
    STESSA pendenza, anche se moderata, produce una perturbazione della
    normale molto piu' grande li' che al centro del disco — limitare solo
    dh/dx grezzo non basta: cio' che conta per il confronto con n0 e' la
    NORMA del vettore di perturbazione finale (dopo Tx/Ty), che e' quella
    limitata qui (preservandone la direzione).

    Diagnosi del bug "la falce non sembra piu' una falce" (immagini
    troppo rugose, macchie chiare sparse anche sul lato in ombra): la
    causa non era, come si pensava all'inizio, la somma lineare di
    crateri sovrapposti (vedi assemble_terrain_capped in terrain.py, che
    resta comunque una correzione utile) — un confronto diretto (stesso
    terreno CON e SENZA crateri) mostrava statistiche di pendenza quasi
    identiche, provando che il contributo dominante era il terreno di
    FONDO (spectral_synthesis_terrain). Per uno spettro P(f)~f^-beta con
    beta<4 (qui beta=3.1, il valore misurato su un vero DEM), la varianza
    della PENDENZA (che pesa lo spettro per f^2 rispetto all'altezza) non
    e' dominata dalle basse frequenze come lo e' l'altezza stessa: resta
    spalmata fin quasi al Nyquist della griglia, cosi' anche un'altezza
    che "sembra" dolce puo' avere un gradiente punto-per-punto molto piu'
    ripido di quanto la sua ampiezza complessiva suggerisca — un artefatto
    della sintesi (uno vero DEM reale e' gia' fisicamente band-limited
    dallo strumento/processing, quindi non ha questo problema alla stessa
    ampiezza). Il rimedio principale e' quindi in generate_moon
    (base_amp_frac ridotto per dare al terreno di fondo una pendenza
    MEDIA realistica, non solo uno spettro di forma realistica); questo
    max_slope resta comunque come limite fisico di sicurezza — un valore
    piu' basso (0.45 = tan(24°) circa, contro il vecchio 0.70=tan(35°))
    perche' con centinaia di crateri anche pendenze "solo" moderate,
    amplificate dal fattore 1/w vicino al lembo, potevano ancora
    accumularsi abbastanza da capovolgere l'illuminazione oltre il
    terminatore. Un residuo di schiarimento molto vicino al terminatore
    (non nel resto del lato in ombra) resta e va bene cosi': e' lo stesso
    effetto per cui, nelle foto reali, cime e bordi di crateri catturano
    la luce radente prima/dopo il resto del terreno, dando un terminatore
    leggermente frastagliato invece che una linea geometrica perfetta.

    sun_dir: vettore 3D unitario, fisso per tutta l'immagine (il sole e'
    a distanza "infinita" quindi la sua direzione non dipende dal punto
    sulla Luna, solo dalla fase)."""
    dh_dy, dh_dx = np.gradient(h, dx)
    perturbation = (relief_strength * dh_dx[..., None] * Tx
                     + relief_strength * dh_dy[..., None] * Ty)
    p_norm = np.linalg.norm(perturbation, axis=-1, keepdims=True)
    clamp = np.clip(max_slope / np.clip(p_norm, 1e-9, None), 0.0, 1.0)
    perturbation = perturbation * clamp

    n = n0 - perturbation
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(norm, 1e-9, None)
    I = np.einsum('...k,k->...', n, sun_dir)
    return np.clip(I, 0.0, 1.0)


# ---------------------------------------------------------------------
# 3. Generazione completa
# ---------------------------------------------------------------------

def generate_moon(grid_size: int = 400, seed: int | None = None, phase_deg: float | None = None,
                   n_craters: tuple[int, int] = (10000, 10500), terrain_beta: float = 3.1,
                   base_amp_frac: float = 0.003, crater_relief_scale: float = 0.75, relief_strength: float = 1.4,
                   max_crater_diam_km: float = 150.0, min_crater_diam_km: float = 5.0,
                   size_freq_exponent: float = 1.9,
                   n_large_craters_mean: float = 2.0,
                   large_crater_min_km: float | None = None, large_crater_max_km: float = 300.0,
                   ambient: float = 0.05, with_shadows: bool = True, shadow_step: float = 1.0,
                   enable_maria: bool = True, maria_coverage: float | None = None,
                   maria_smoothing: float = 0.6, maria_crater_suppression: float = 0.75,
                   maria_density_mult: float = 1.0,
                   maria_depth_atten: float = 0.4, albedo_contrast: float = 0.45,
                   highland_roughness_mult: float = 2.0, maria_style: str = "disks",
                   maria_flood_smooth_px: float = 0.0, crater_foreshorten_strength: float = 1.0,
                   avoid_crater_boundary_intersections: bool = True,
                   add_crater_mare: bool = False, crater_mare_diam_km: float = 200.0,
                   crater_mare_max_coverage: float = 0.05, maria_n_basins: int | None = None,
                   maria_edge_width_frac: float = 0.05):
    """Genera un'immagine di una luna artificiale.

    phase_deg: se None, viene scelta a caso in [0, 360) — cioe' una fase
    casuale qualsiasi, non solo luna piena. 0/360=piena, 90=ultimo
    quarto/primo quarto (a seconda del lato), 180=nuova. Determina la
    direzione FISSA del sole (a distanza infinita); l'illuminazione
    punto per punto viene poi calcolata correttamente in base alla
    normale sferica locale, vedi spherical_lambert_shading.
    n_craters: (min, max) del numero di crateri sparsi, scelto a caso in
    questo intervallo. Default ora **(10000, 10500)**, non piu' (250, 500)
    — vedi la nota "luna quasi liscia riempita di crateri" piu' sotto nel
    modulo/README: confrontando la rugosita' fine reale (misurata da un
    vero DEM lunare, GLD100) con quella del terreno sintetico, e' emerso
    che quasi tutta la "grana" degli altopiani reali e' in realta'
    crateri piccoli saturi, non rumore diffuso senza struttura — quindi
    ora la densita' di crateri fa il lavoro che prima faceva soprattutto
    base_amp_frac (vedi sotto). scatter_craters usa gia' un riquadro
    locale per cratere (assemble_terrain_capped/crater_fields), non la
    griglia intera: migliaia di crateri costano relativamente poco (vedi
    il confronto di tempi nel README — a --grid 1200, 10000+ crateri
    aggiungono solo ~1s sopra il costo base, non craters-dependent, del
    resto della pipeline), quindi il default piu' alto resta comunque
    veloce anche alla risoluzione di anteprima (--grid 400: ~3.7s).
    max_crater_diam_km / min_crater_diam_km / size_freq_exponent: vedi
    scatter_craters — controllano la distribuzione dimensionale reale
    dei crateri "normali" (tetto a 150 km di default, e l'esponente
    ~1.8-2.0 di cui parlavamo). NOTA: min_crater_diam_km e' spesso
    irraggiungibile alla risoluzione scelta (vedi scatter_craters) — a
    --grid bassi la griglia non puo' comunque disegnare crateri sotto
    ~15-30 km, quindi con "solo" poche centinaia di crateri capped a
    150 km il risultato puo' sembrare "a puntini isolati" invece che
    densamente craterizzato: e' un limite di risoluzione, non del
    modello. Il rimedio preferito ora e' alzare n_craters (sopra) e/o
    --grid, non piu' alzare base_amp_frac (vedi sotto) — quella era una
    scorciatoia per "riempire" visivamente gli spazi vuoti con rumore
    invece che con altri crateri veri, che il confronto coi dati reali
    suggerisce essere meno fedele.

    n_large_craters_mean / large_crater_min_km / large_crater_max_km:
    seconda popolazione di crateri, SEPARATA da quella "normale" sopra
    (stessa legge di potenza/size_freq_exponent e stesso meccanismo di
    soppressione nei mari, solo un diametro tipicamente molto piu'
    grande e un conteggio proprio invece di condividere n_craters).
    Prima di questi tre parametri non c'era modo di controllare quanti
    crateri grandi comparissero se non indirettamente, alzando
    max_crater_diam_km per TUTTA la popolazione (il che significava
    anche allargare il tetto dei crateri piccoli, non solo aggiungerne
    qualcuno di grande) — qui il numero di crateri grandi per luna e'
    invece un conteggio Poisson di media n_large_craters_mean (quindi
    "in media" quel tanti, ma diverso ogni luna: 0 in una, 4 in
    un'altra, per varieta' naturale), pescati fra large_crater_min_km e
    large_crater_max_km. Default: media 2.0 crateri, fra 150 km (lo
    stesso valore di max_crater_diam_km di default: le due popolazioni
    sono contigue, senza sovrapposizione ne' buco) e 300 km — abbastanza
    per rappresentare crateri complessi grandi come Clavius (~230 km) o
    Schickard (~227 km), che sono comunque crateri "normali" (non
    bacini-mare, quelli restano generate_maria_mask). large_crater_min_km
    default None = usa automaticamente max_crater_diam_km come confine,
    cosi' cambiando quest'ultimo il confine fra le due popolazioni si
    sposta insieme invece di lasciare un buco o una sovrapposizione.
    n_large_craters_mean=0 disattiva del tutto questa seconda popolazione.

    crater_foreshorten_strength: vedi disk_radial_geometry. I crateri sono
    cerchi fisici sulla sfera, quindi in proiezione ortografica vicino al
    lembo del disco appaiono ellittici (schiacciati in direzione radiale)
    — un effetto reale, visibile anche nelle foto vere. Un primo tentativo
    (poi corretto) aveva ridotto il default a 0.6 pensando che l'utente si
    riferisse a QUESTA ellitticita' da proiezione; chiarito poi che
    l'ellitticita' vicino al lembo va bene com'e' (e' quella fisicamente
    vera) — quello che andava ridotto era invece l'eventuale ellitticita'
    "intrinseca" del cratere nel proprio piano tangente, indipendente dalla
    posizione sul disco (vedi irregularity/scatter_craters e il README).
    Default quindi di nuovo **1.0** = correzione fisicamente esatta.
    0.0 = crateri sempre circolari in proiezione, indipendentemente dalla
    posizione sul disco (nessuna correzione prospettica) — utile solo per
    debug/confronto, non per l'uso normale.

    avoid_crater_boundary_intersections: vedi scatter_craters. Default
    True: due crateri che si intersecano a meta' (nessuno dei due
    contenuto nell'altro) vengono evitati ripescando la posizione;
    un cratere piccolo interamente dentro uno grande resta permesso.
    Metti a False solo per confrontare col vecchio comportamento o per
    velocizzare iterazioni rapide con pochissimi crateri.

    crater_relief_scale: fattore MOLTIPLICATIVO applicato DOPO che ogni
    cratere e' gia' stato dimensionato con leggi di scala realistiche
    (vedi scatter_craters/crater_depth_over_diameter) — 1.0 = "tieni le
    proporzioni reali cosi' come sono". Default ora **0.75**, non 1.0:
    feedback diretto ("crateri piu' marcati ma anche piu' piatti") dopo
    aver visto che con le proporzioni reali intere il profilo vero e'
    piu' ripido di quanto serva per un buon aspetto visivo — qui si
    riduce la profondita' VERA e si compensa con relief_strength (sotto)
    per il contrasto nello shading, invece di scegliere fra "realistico
    ma ripido" e "piatto ma spento". Il rilievo di fondo (base_amp_frac/
    terrain_beta) resta comunque un contributo SEPARATO e indipendente:
    scala solo i crateri, non il terreno fra un cratere e l'altro.
    relief_strength: leva separata, solo estetica: quanto la pendenza
    locale del terreno perturba la normale nello SHADING (contrasto),
    SENZA cambiare l'heightfield vero ne' quindi le ombre portate, che
    infatti lo ignorano. Default ora **1.4**, non 1.0, per compensare
    in contrasto la minore profondita' vera di crater_relief_scale=0.75
    (vedi sopra) — crateri che risaltano di piu' nello shading a parita'
    (anzi, a minor) rilievo fisico reale.

    enable_maria / maria_coverage / maria_smoothing / maria_crater_suppression
    / maria_depth_atten / albedo_contrast: vedi generate_maria_mask e la
    nota nel docstring del modulo — regionalizzazione procedurale
    mari/altopiani (riflettivita', rugosita' di fondo, densita' di
    crateri grandi, tutte correlate come nella realta'). enable_maria=False
    ripristina il vecchio comportamento (albedo/rugosita' uniformi).

    maria_style: 'disks' (default, vedi generate_maria_mask — cerchi, o
    la coppia grande+piccolo forzata a sovrapporsi per almeno meta') o
    'flood' (sperimentale, vedi generate_maria_mask_flood — sagoma da un
    allagamento a soglia crescente del terreno di fondo vero, fermato
    appena prima della transizione di percolazione: risultato piu'
    organico/imprevedibile, coverage non piu' un target rispettato ma un
    risultato). Ignorato se enable_maria=False. maria_coverage viene
    ignorato con maria_style='flood' (la dimensione la decide il
    terreno, non un target imposto).

    maria_flood_smooth_px (default 0.0, solo con maria_style='flood'):
    smussa il contorno frastagliato dell'allagamento per percolazione
    dilatandolo di questo raggio in pixel — vedi smooth_radius_px in
    generate_maria_mask_flood per i dettagli.

    highland_roughness_mult: la rugosita' fine residua dei mari
    (mare_residual_frac, sopra) e' stata giudicata "perfetta" cosi' com'e' —
    questo parametro fissa quella dei mari come RIFERIMENTO e chiede quanto
    piu' accentuata (in ampiezza) debba essere la stessa texture fine sul
    resto del terreno, cioe' la zona che non e' ne' cratere ne' mare
    (l'altopiano "nudo" fra un cratere e l'altro). Default 2.0 = il doppio
    dell'ampiezza dei mari. Prima di questo parametro l'altopiano usava
    l'intera ampiezza del campo di fondo (base_amp_frac) senza alcuna
    riduzione — con maria_smoothing=0.6 (default) questo equivaleva a
    circa 6x l'ampiezza dei mari, molto piu' del rapporto 2x che si vede
    qui: la stessa formula usata per i mari (campo di fondo riportato a
    un'ampiezza ridotta, ma qui SENZA lo spostamento verso il basso che
    invece simula l'allagamento lavico — l'altopiano non e' un bacino)
    viene ora applicata anche fuori dai mari, solo con un residuo doppio
    invece che uguale. Nessun effetto se enable_maria=False (in quel caso
    non c'e' un riferimento "mare" a cui essere relativi, e si torna al
    vecchio comportamento uniforme).

    add_crater_mare / crater_mare_diam_km / crater_mare_max_coverage:
    OPZIONALE (default disattivato): aggiunge un mare PIU' PICCOLO in
    piu', con un'origine diversa dal/dai mare/i normali — invece di un
    bacino che nasce da un minimo genuino del terreno frattale, parte da
    una conca scavata apposta delle dimensioni di un grande cratere
    (crater_mare_diam_km, default 200 km) e la fa crescere con lo STESSO
    metodo di percolazione, ma con un tetto di copertura molto piu' basso
    (crater_mare_max_coverage, default 5% del disco — contro il 4-48%
    tipico dei mari normali): "un cratere allagato, allargato ma non
    troppo", non un bacino vero e proprio. Vedi generate_crater_seeded_mare
    per i dettagli. Funziona con qualunque maria_style (la parola
    "percolazione" nella richiesta si riferisce a QUESTO mare aggiuntivo,
    non al mare principale, che puo' restare a dischi). Nessun effetto se
    enable_maria=False.

    maria_n_basins: se None (default), il NUMERO di mari principali resta
    casuale (0/1/2, vedi generate_maria_mask/generate_maria_mask_flood).
    Passare esplicitamente 0, 1 o 2 forza quel numero — es. maria_n_basins=2
    per avere sempre due mari. Non conta il mare da cratere (add_crater_mare),
    che e' sempre in piu' e indipendente da questo conteggio.

    maria_edge_width_frac (default 0.05): quanto e' GRADUALE (sfumata,
    "smussata") la transizione di albedo/rilievo al bordo di un mare,
    invece di un salto netto — e' lo stesso `edge_width_frac` di
    generate_maria_mask/generate_maria_mask_flood/generate_crater_seeded_mare
    (finora fissato al loro default, MAI esposto qui: era hardcoded, non
    un buco nella copertura di generate_moon rispetto a moon_ot.py -
    semplicemente non era ancora un parametro libero da nessuna parte).
    E' una FRAZIONE di grid_size (non pixel assoluti), quindi lo stesso
    valore da' una sfumatura proporzionalmente uguale a qualunque
    risoluzione. Usato per TUTTI i mari (principali e quello da
    cratere, se attivo) con lo stesso valore.

    Ritorna (img, meta) dove meta e' un dict con tutti i parametri usati
    (utile per rigenerare la stessa luna passando lo stesso seed/fase)."""
    rng = np.random.default_rng(seed)
    if phase_deg is None:
        phase_deg = float(rng.uniform(0, 360))
    n = int(rng.integers(n_craters[0], n_craters[1] + 1))

    # Terreno di fondo via sintesi spettrale (vedi terrain.spectral_synthesis_terrain):
    # diamond_square/make_base_pattern non riesce a riprodurre la pendenza spettrale
    # misurata su un vero DEM lunare (limite strutturale dell'algoritmo, non di
    # taratura — vedi il commento nel modulo terrain.py); terrain_beta=3.1 e' il
    # valore misurato su un ritaglio reale GLD100 vicino a Tycho.
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

    crater_mare_coverage_used, crater_mare_pos = 0.0, None
    if enable_maria and add_crater_mare:
        # Mare aggiuntivo, opzionale, "nato" da un grande cratere invece
        # che da un minimo naturale del terreno — vedi generate_crater_seeded_mare.
        # excluded=maria_mask>0.5 evita solo che il NUOVO cratere nasca
        # esattamente dentro il mare gia' scelto; puo' comunque finirgli
        # accanto o sfiorarlo (fuso via np.maximum, come fra due bacini
        # normali in generate_maria_mask_flood).
        crater_mare_mask, crater_mare_coverage_used, crater_mare_pos = generate_crater_seeded_mare(
            grid_size, base, rng, crater_diam_km=crater_mare_diam_km,
            max_coverage=crater_mare_max_coverage, edge_width_frac=maria_edge_width_frac,
            excluded=(maria_mask > 0.5) if maria_mask is not None else None)
        maria_mask = np.maximum(maria_mask, crater_mare_mask) if maria_mask is not None else crater_mare_mask
        maria_coverage_used = float(maria_coverage_used) + crater_mare_coverage_used

    craters = scatter_craters(grid_size, n, rng, max_diam_km=max_crater_diam_km,
                               min_diam_km=min_crater_diam_km, size_freq_exponent=size_freq_exponent,
                               maria_mask=maria_mask, maria_suppression=maria_crater_suppression,
                               maria_density_mult=maria_density_mult,
                               maria_depth_atten=maria_depth_atten,
                               foreshorten_strength=crater_foreshorten_strength,
                               avoid_boundary_intersections=avoid_crater_boundary_intersections)
    # Seconda popolazione, separata: crateri "grandi" (vedi il docstring sopra
    # per il perche' — controllo diretto ed esplicito del conteggio, invece
    # di doverlo ottenere indirettamente allargando max_crater_diam_km per
    # TUTTI i crateri). Conteggio Poisson: n_large_craters_mean e' una MEDIA,
    # il numero vero varia da luna a luna (puo' anche capitare 0). Stessa
    # funzione scatter_craters, stesso rng (quindi comunque riproducibile a
    # parita' di seed), solo un intervallo dimensionale diverso e contiguo
    # a quello "normale" (large_crater_min_km di default = max_crater_diam_km).
    n_large = int(rng.poisson(n_large_craters_mean)) if n_large_craters_mean > 0 else 0
    if n_large > 0:
        effective_large_min_km = large_crater_min_km if large_crater_min_km is not None else max_crater_diam_km
        large_craters = scatter_craters(grid_size, n_large, rng,
                                         max_diam_km=large_crater_max_km,
                                         min_diam_km=effective_large_min_km,
                                         size_freq_exponent=size_freq_exponent,
                                         maria_mask=maria_mask, maria_suppression=maria_crater_suppression,
                                         maria_density_mult=maria_density_mult,
                                         maria_depth_atten=maria_depth_atten,
                                         foreshorten_strength=crater_foreshorten_strength,
                                         avoid_boundary_intersections=avoid_crater_boundary_intersections,
                                         avoid_craters=craters,
                                         # i crateri "grandi" nascono DOPO decine di migliaia di crateri
                                         # normali gia' piazzati (avoid_craters=craters sopra): in un campo
                                         # gia' saturo trovare un punto dove un cerchio grande non sfiori
                                         # il bordo di NESSUNO dei tanti piccoli vicini puo' richiedere molti
                                         # tentativi. Costa comunque pochissimo in assoluto: sono in media
                                         # solo n_large_craters_mean crateri (default 2), quindi anche
                                         # migliaia di tentativi qui pesano meno di un secondo in totale
                                         # (misurato: anche a 20000 tentativi il tempo totale non cambia
                                         # percepibilmente). Budget molto piu' alto che per la popolazione
                                         # normale di conseguenza.
                                         max_placement_attempts=20000)
        craters = craters + large_craters
    # base e crateri assemblati SEPARATAMENTE cosi' crater_relief_scale puo'
    # scalare solo il contributo dei crateri, lasciando la texture di fondo
    # (che riempie gli spazi fra un cratere e l'altro) al suo livello naturale
    h_base = base * (grid_size * base_amp_frac)
    if maria_mask is not None:
        # SECONDA versione dei mari (la prima, un massimo puntuale con un
        # livello di allagamento fisso VICINO AL MASSIMO del rilievo di
        # fondo, aveva due problemi concreti segnalati dopo averla vista in
        # azione: (1) dove il rumore frattale di fondo superava gia' quel
        # livello anche dentro al mare — che capita spesso, non solo alle
        # cime vere, perche' e' un confronto punto per punto con un rumore
        # fine — quel punto restava "non allagato" cosi' com'era: il
        # confine fra "allagato/piatto" e "non allagato" seguiva quindi il
        # contorno FRASTAGLIATO del rumore stesso, non il bordo morbido
        # della maschera mare, disseminando isolotti dai bordi netti dentro
        # al mare — proprio la rugosita' residua che sembrava ancora
        # perturbare la sagoma della falce; (2) un livello fisso VICINO AL
        # MASSIMO e' piu' alto della media del terreno circostante (~meta'
        # dell'escursione): il mare finiva per essere un altopiano rialzato,
        # non un bacino — il contrario della realta'.
        #
        # Qui invece il campo usato nel mare e' lo STESSO campo frattale di
        # fondo (nessuna nuova discontinuita' introdotta), solo riportato a
        # un livello fisso PIU' BASSO della media del terreno circostante
        # (mare_level_frac, sotto 0.5 = un bacino, non un altopiano) e con
        # l'ampiezza delle proprie fluttuazioni ridotta drasticamente
        # (mare_residual_frac) invece che azzerata o lasciata a tratti
        # intatta: resta una texture fine molto tenue (le vere dorsali/
        # pieghe laviche dei mari reali), non un'isola improvvisa. La
        # transizione fra mare e altopiano resta quella del bordo morbido
        # gia' presente in maria_mask, non un contorno casuale del rumore.
        # --maria-smoothing (stesso parametro/range [0,1] di prima, significato
        # aggiornato alla nuova formula): quanta rugosita' fine del campo di
        # fondo sopravvive nel mare. 0=quasi tutta (allagamento minimo),
        # 1=quasi azzerata (mare quasi perfettamente piatto, solo dorsali
        # lievissime). Default 0.6 -> residuo 0.16, vicino al minimo che
        # ancora si vede come texture invece che sparire del tutto.
        mare_level_frac = 0.30
        mare_residual_frac = float(np.clip(0.40 * (1.0 - maria_smoothing), 0.03, 0.40))
        mare_level = mare_level_frac * (grid_size * base_amp_frac)
        h_base_mare = mare_level + (base - 0.5) * (grid_size * base_amp_frac) * mare_residual_frac

        # Rugosita' dell'altopiano (fuori da crateri e mari) RELATIVA a
        # quella dei mari, non piu' l'intera ampiezza di base_amp_frac senza
        # riduzione: la rugosita' fine vista nei mari e' stata giudicata
        # "perfetta" cosi' com'e' — qui si applica la STESSA formula
        # (stesso campo frattale di fondo, ampiezza ridotta) anche
        # all'altopiano, solo con un residuo highland_roughness_mult volte
        # (default 2x) quello dei mari invece che uguale. A differenza dei
        # mari pero' NON c'e' alcuno spostamento verso un "livello" fisso
        # piu' basso (mare_level): l'altopiano non e' un bacino allagato,
        # resta centrato sulla stessa quota media di sempre (0.5*ampiezza).
        # Prima di questo parametro l'altopiano teneva l'intera ampiezza
        # (base*amp, equivalente a un "residuo" di 1.0) — con
        # maria_smoothing=0.6 questo era circa 6x l'ampiezza dei mari,
        # molto piu' del rapporto 2x visto qui.
        highland_residual_frac = float(np.clip(highland_roughness_mult * mare_residual_frac, 0.0, 1.0))
        h_base_highland = 0.5 * (grid_size * base_amp_frac) + \
            (base - 0.5) * (grid_size * base_amp_frac) * highland_residual_frac
        h_base = h_base_highland * (1.0 - maria_mask) + h_base_mare * maria_mask

    # Levigatura generale del terreno di fondo (fuori dai crateri): anche
    # dopo aver ridotto base_amp_frac (vedi sopra), il rilievo puo' ancora
    # "sembrare" ruvido perche' la rugosita' fine (pixel-a-pixel) resta ad
    # alta frequenza anche se piccola in ampiezza — sotto luce radente lo
    # shading Lambertiano la rende comunque visibile come una grana
    # "a carta vetrata" uniforme su tutto il disco. Ridurre SOLO l'ampiezza
    # non basta a togliere questa grana (la attenua, non la elimina);
    # sfocare leggermente il campo (qui, PRIMA di aggiungere i crateri, che
    # restano quindi nitidi) la elimina invece direttamente. Non tocca
    # crateri (sommati dopo) ne' la loro nitidezza — solo il "riempitivo"
    # fra un cratere e l'altro, che e' esattamente cio' che si voleva
    # appiattire lasciando i crateri come le uniche strutture marcate.
    #
    # sigma=2.0 (primo tentativo) era troppo: il feedback successivo
    # ("l'immagine non e' nitida") aveva ragione — a quel punto anche i
    # bordi dei crateri piccoli e il confine dei mari iniziavano a apparire
    # sfocati, non solo il rumore di fondo. sigma=0.8 toglie comunque la
    # grana piu' fine pixel-a-pixel (il problema originale) mantenendo
    # nitidi i bordi/crateri piu' piccoli — un confronto diretto a sigma
    # 0/0.7/1.0/1.5 ha mostrato che oltre 1.0 la nitidezza peggiora in modo
    # visibile senza un beneficio proporzionale sulla grana residua.
    h_base = gaussian_filter(h_base, sigma=0.8)
    # assemble_terrain_capped invece di assemble_terrain: con centinaia di
    # crateri sparsi, sommare linearmente ogni bordo rialzato dove piu'
    # crateri si sovrappongono per caso produceva pendenze locali composte
    # fisicamente impossibili (vedi la nota su "falce rotta" piu' sotto e
    # in spherical_lambert_shading) — qui ogni pixel prende la conca PIU'
    # PROFONDA e il bordo/picco PIU' ALTO fra i crateri che lo coprono,
    # invece della somma di tutti (identico per un cratere isolato).
    h_craters = assemble_terrain_capped(np.zeros_like(base), amp=0.0, craters=craters)
    h = h_base + h_craters * crater_relief_scale

    phase_rad = np.deg2rad(phase_deg)
    sun_dir = np.array([np.sin(phase_rad), 0.0, np.cos(phase_rad)])

    n0, Tx, Ty, disk_mask = sphere_geometry(grid_size)
    I_shading = spherical_lambert_shading(h, n0, Tx, Ty, sun_dir, relief_strength=relief_strength)

    if with_shadows:
        # Per le ombre PORTATE (ray-marching, direzione unica per l'intera
        # griglia) uso l'azimut vero del sole (proiezione di sun_dir sul
        # piano immagine — cosi' le ombre puntano dalla parte giusta) ma
        # un'ALTEZZA fissa moderata, non quella vera del punto
        # sub-osservatore. Motivo: la vera altezza locale del sole varia
        # da punto a punto (e' proprio quello che spherical_lambert_shading
        # calcola correttamente sopra); usarne UNA sola per tutta la
        # griglia e' necessario per il ray-marching, e se si sceglie quella
        # del centro disco, vicino a fasi di quarto quel valore puo' essere
        # quasi zero o negativo — con tan(alt) vicino a zero il
        # ray-marching blocca quasi tutta la griglia, producendo ombre
        # innaturali a chiazze su tutto il disco invece che solo vicino ai
        # crateri. Un'altezza fissa moderata (buona per il rilievo dei
        # crateri) evita il problema; dove la vera illuminazione e' comunque
        # vicina a zero (terminatore/lato notte) il fattore d'ombra conta
        # comunque poco perche' moltiplica un I_shading gia' quasi nullo.
        # una leggera dipendenza dalla fase: vicino a luna piena (fase~0)
        # il sole e' alto quasi ovunque sul lato visibile -> pochissime
        # ombre proprie (come nella realta', dove le foto a luna piena
        # appaiono "piatte"); vicino ai quarti/crescente le ombre sono
        # piu' drammatiche. Il pavimento a 15 gradi evita comunque la
        # patologia del ray-marching a altezza quasi zero.
        local_alt = 15.0 + 70.0 * max(0.0, float(np.cos(phase_rad)))
        local_az = float(np.degrees(np.arctan2(sun_dir[1], sun_dir[0])))
        # cast_shadows marcia a passi FISSI in pixel lungo una diagonale che
        # e' invece lunga hypot(grid,grid) PIXEL — cioe' proporzionale alla
        # risoluzione. Con uno shadow_step fisso (com'era prima), il numero
        # di passi cresce linearmente con --grid E ogni passo costa gia'
        # O(grid^2) (campiona l'intera griglia): il costo totale del
        # ray-marching cresce quindi come O(grid^3), non O(grid^2) come il
        # resto della pipeline — misurato: 420->800->1200 px, ~3s->22s->78s,
        # una crescita molto piu' che quadratica. Scalando lo step con
        # --grid (qui, calibrato cosi' che grid=400 dia lo stesso step di
        # sempre) il numero di passi resta CIRCA COSTANTE al variare della
        # risoluzione — stessa densita' di campionamento RELATIVA lungo il
        # raggio, non la stessa assoluta in pixel — riportando il costo del
        # ray-marching a O(grid^2), utile soprattutto per provare griglie
        # grandi (--grid 1000+).
        effective_shadow_step = shadow_step * (grid_size / 400.0)
        shadow = cast_shadows(h, local_az, local_alt, step=effective_shadow_step)
    else:
        shadow = 1.0

    img = ambient + (1.0 - ambient) * I_shading * shadow

    if maria_mask is not None:
        # riflettivita' (albedo) piu' bassa nei mari — l'ingrediente di
        # realismo che mancava del tutto nella versione a albedo uniforme:
        # senza questo, mari e altopiani sono indistinguibili a parita' di
        # pendenza/illuminazione, il che non e' come appare la Luna reale.
        albedo_field = 1.0 - albedo_contrast * maria_mask
        img = img * albedo_field

    img = np.clip(img, 0.0, 1.0) * disk_mask

    meta = dict(grid_size=grid_size, seed=seed, phase_deg=phase_deg, n_craters=len(craters),
                n_large_craters=n_large,
                terrain_beta=terrain_beta, ambient=ambient, crater_relief_scale=crater_relief_scale,
                relief_strength=relief_strength, craters=craters, h=h,
                enable_maria=enable_maria, maria_coverage=maria_coverage_used,
                albedo_contrast=albedo_contrast,
                add_crater_mare=add_crater_mare, crater_mare_coverage=crater_mare_coverage_used,
                crater_mare_pos=crater_mare_pos,
                # with_shadows/shadow_step/maria_mask: aggiunti per poter
                # rirenderizzare fedelmente un heightfield modificato DOPO
                # generate_moon (es. moon_generation_patchmatch, che innesta
                # texture reale in h e poi richiama la stessa pipeline di
                # illuminazione) senza dover rigenerare crateri/mari da zero.
                with_shadows=with_shadows, shadow_step=shadow_step, maria_mask=maria_mask)
    return img, meta


def phase_name(phase_deg: float) -> str:
    p = phase_deg % 360
    if p < 11.25 or p >= 348.75:
        return "luna piena"
    if p < 78.75:
        return "gibbosa calante" if p < 45 else "ultimo quarto"
    if p < 101.25:
        return "ultimo quarto"
    if p < 168.75:
        return "falce calante"
    if p < 191.25:
        return "luna nuova"
    if p < 258.75:
        return "falce crescente"
    if p < 281.25:
        return "primo quarto"
    return "gibbosa crescente"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genera lune artificiali con fase casuale.")
    parser.add_argument("--grid", type=int, default=400)
    parser.add_argument("--n", type=int, default=6, help="quante lune generare in una griglia di anteprima")
    parser.add_argument("--seed", type=int, default=None, help="seed globale (None = casuale ad ogni run)")
    parser.add_argument("--phase", type=float, default=None,
                         help="fase fissa in gradi (0=piena, 180=nuova). Se omesso, ogni luna generata ha fase casuale indipendente.")
    parser.add_argument("--n-craters-min", type=int, default=10000,
                         help="default alzato da 250 a 10000: vedi la nota 'luna quasi liscia riempita di crateri' "
                              "nel README — la rugosita' fine reale e' quasi tutta crateri piccoli saturi, non "
                              "rumore diffuso, quindi ora la densita' di crateri fa il lavoro che prima faceva "
                              "soprattutto --base-amp-frac. Costa relativamente poco (crateri lavorano su un "
                              "riquadro locale, non sull'intera griglia)")
    parser.add_argument("--n-craters-max", type=int, default=10500)
    parser.add_argument("--max-crater-km", type=float, default=150.0,
                         help="diametro massimo dei crateri 'normali' (vedi --n-large-craters-mean per la popolazione "
                              "separata di crateri grandi), in km. Default 150. E' anche, di default, il confine "
                              "inferiore della popolazione dei crateri grandi (vedi --large-crater-min-km)")
    parser.add_argument("--min-crater-km", type=float, default=5.0,
                         help="diametro minimo desiderato per i crateri 'normali', in km (default 5, non piu' 3: gia' "
                              "sotto la risoluzione tipica di questo generatore — vedi scatter_craters, sotto ~1-2 "
                              "pixel un cratere non e' comunque disegnabile in modo sensato)")
    parser.add_argument("--n-large-craters-mean", type=float, default=2.0,
                         help="numero MEDIO di crateri 'grandi' per luna (popolazione separata da quella normale "
                              "sopra, stessa legge di potenza ma conteggio proprio, pescato da una distribuzione di "
                              "Poisson: il numero vero varia da luna a luna attorno a questa media, puo' anche essere "
                              "0). Default 2.0. Metti 0 per disattivare del tutto questa seconda popolazione (torna "
                              "al comportamento di prima: crateri grandi solo come coda della legge di potenza "
                              "'normale', fino a --max-crater-km)")
    parser.add_argument("--large-crater-min-km", type=float, default=None,
                         help="diametro minimo dei crateri 'grandi', in km. Default: uguale a --max-crater-km (le due "
                              "popolazioni restano contigue, senza sovrapposizione ne' buco, anche se cambi "
                              "--max-crater-km)")
    parser.add_argument("--large-crater-max-km", type=float, default=300.0,
                         help="diametro massimo dei crateri 'grandi', in km. Default 300 — abbastanza per crateri "
                              "complessi grandi come Clavius (~230 km) o Schickard (~227 km), restando comunque "
                              "crateri 'normali' e non bacini-mare (quelli sono generate_maria_mask)")
    parser.add_argument("--exponent", type=float, default=1.9,
                         help="esponente b della distribuzione cumulativa N(>D)~D^-b (la 'dimensione frattale' "
                              "dei crateri lunari, tipicamente 1.8-2.0): piu' alto = molti piu' crateri piccoli")
    parser.add_argument("--crater-foreshorten-strength", type=float, default=1.0,
                         help="quanto applicare la correzione ellittica da prospettiva ai crateri vicino al lembo "
                              "del disco (vedi disk_radial_geometry/generate_moon): 1.0 = fisicamente esatta "
                              "(default, e' un effetto reale), 0.0 = crateri sempre circolari in proiezione "
                              "indipendentemente dalla posizione sul disco (solo per debug/confronto)")
    parser.add_argument("--allow-crater-overlap", action="store_true",
                         help="disattiva l'evitamento dei crateri che si intersecano a meta' (vedi "
                              "scatter_craters/avoid_boundary_intersections) — un cratere piccolo interamente "
                              "dentro uno grande resta comunque permesso in ogni caso, questo flag riguarda solo "
                              "l'intersezione parziale dei bordi. Utile solo per confronto o per velocizzare "
                              "iterazioni rapide")
    parser.add_argument("--relief-scale", type=float, default=0.75,
                         help="moltiplicatore artistico SOPRA la profondita' gia' fisicamente proporzionata di ogni "
                              "cratere (vedi crater_depth_over_diameter): 1.0 = tieni le proporzioni reali, valori "
                              "piu' bassi = crateri deliberatamente piu' dolci/piatti del reale. Default 0.75 (non "
                              "piu' 1.0): crateri col profilo VERO leggermente attenuato, compensato in resa da "
                              "--relief-strength piu' alto — 'crateri piu' marcati ma piu' piatti', vedi sotto")
    parser.add_argument("--base-amp-frac", type=float, default=0.003,
                         help="ampiezza del rilievo di fondo (frazione di --grid): e' il rumore diffuso SENZA "
                              "struttura che riempie gli spazi fra un cratere risolto e l'altro, indipendente da "
                              "--relief-scale. Default ora 0.003 (non piu' 0.010): confrontato con un vero DEM lunare "
                              "(GLD100), quasi tutta la rugosita' fine reale degli altopiani risulta essere crateri "
                              "piccoli saturi, non rumore senza struttura — quindi ora il grosso della texture fine "
                              "viene da --n-craters-min/max (default molto piu' alto, 2000-2500) invece che da questo "
                              "parametro, che resta comunque utile per non lasciare completamente piatte le zone "
                              "sotto la risoluzione minima disegnabile di crateri. Il vecchio default 0.010 dava una "
                              "pendenza MEDIA di fondo di circa 0.05-0.06 (~3 gradi); quello ancora precedente, 0.05, "
                              "dava ~0.33 (quasi 18 gradi ovunque) — vedi il commento in spherical_lambert_shading per "
                              "la storia di quella correzione")
    parser.add_argument("--terrain-beta", type=float, default=3.1,
                         help="pendenza spettrale del terreno di fondo (sintesi spettrale, vedi terrain.py): valore "
                              "piu' alto = piu' energia su grande scala, meno grana fine. 3.1 e' il valore misurato "
                              "su un vero DEM lunare (GLD100, patch vicino a Tycho) — non e' lo stesso parametro "
                              "della vecchia 'roughness' di diamond_square (quella aveva un tetto strutturale e la "
                              "direzione dell'effetto era, tra l'altro, documentata al contrario)")
    parser.add_argument("--no-maria", action="store_true",
                         help="disabilita la regionalizzazione mari/altopiani (torna ad albedo e rugosita' uniformi)")
    parser.add_argument("--maria-coverage", type=float, default=None,
                         help="frazione del disco coperta da 'mare', in [0,1]. Se omesso, casuale in [0.15,0.40] per ogni luna. Ignorato con --maria-style flood")
    parser.add_argument("--maria-style", type=str, default="disks", choices=["disks", "flood"],
                         help="'disks' (default): cerchi (il piccolo forzato a sovrapporre almeno meta' del grande, se sono 2). "
                              "'flood' (sperimentale): sagoma da un allagamento a soglia crescente del terreno di fondo vero, "
                              "fermato appena prima della transizione di percolazione — piu' organico, coverage non piu' un "
                              "target ma un risultato (vedi generate_maria_mask_flood)")
    parser.add_argument("--maria-flood-smooth", type=float, default=0.0,
                         help="solo con --maria-style flood: raggio (in pixel) della dilatazione morfologica che smussa "
                              "il contorno frastagliato/ramificato del flood (vedi smooth_radius_px in "
                              "generate_maria_mask_flood). 0=disattivato (default), valori tipici 3-10")
    parser.add_argument("--albedo-contrast", type=float, default=0.45,
                         help="quanto sono piu' scuri i mari rispetto agli altopiani, in [0,1] (0=nessuna differenza)")
    parser.add_argument("--add-crater-mare", action="store_true",
                         help="opzionale: aggiunge un mare piu' piccolo in piu', 'nato' da un grande cratere "
                              "(vedi generate_crater_seeded_mare) invece che da un minimo naturale del terreno")
    parser.add_argument("--crater-mare-diam-km", type=float, default=200.0,
                         help="solo con --add-crater-mare: diametro (km) del cratere da cui parte la conca scavata")
    parser.add_argument("--crater-mare-max-coverage", type=float, default=0.05,
                         help="solo con --add-crater-mare: tetto di copertura (frazione del disco) di quanto puo' "
                              "'allargarsi' oltre il cratere stesso — molto piu' basso del cap dei mari normali")
    parser.add_argument("--maria-n-basins", type=int, default=None, choices=[0, 1, 2],
                         help="forza il numero di mari PRINCIPALI (0, 1 o 2) invece di lasciarlo casuale. "
                              "Non conta il mare da cratere (--add-crater-mare), sempre in piu'")
    parser.add_argument("--maria-edge-width", type=float, default=0.05,
                         help="quanto e' graduale/sfumata (invece che netta) la transizione di albedo al bordo di "
                              "un mare, come frazione di --grid. Vale per tutti i mari (principali e quello da "
                              "cratere). Valori piu' alti = bordo piu' smussato/sfumato")
    parser.add_argument("--maria-smoothing", type=float, default=0.6,
                         help="quanta rugosita' fine del terreno di fondo sopravvive nei mari, in [0,1] MA invertito: "
                              "0=quasi tutta (mare appena piu' calmo dell'altopiano), 1=quasi azzerata (mare quasi "
                              "perfettamente piatto). I mari sono simulati come bacini allagati dalla lava fino a un "
                              "livello FISSO PIU' BASSO della media del terreno circostante (non un altopiano "
                              "rialzato), con la texture fine dello stesso campo di fondo solo fortemente attenuata "
                              "(non sostituita da un contorno rumoroso: niente piu' 'isolotti' dai bordi netti dentro "
                              "al mare)")
    parser.add_argument("--relief-strength", type=float, default=1.4,
                         help="leva SOLO estetica (contrasto nello shading, non cambia l'heightfield ne' le ombre "
                              "portate): quanto la pendenza locale perturba la normale nel calcolo dell'illuminazione. "
                              "Valori piu' alti fanno risaltare di piu' i crateri (utile se sembrano 'poco marcati') "
                              "senza toccare le loro proporzioni fisiche reali — a differenza di --relief-scale, che "
                              "invece scala la profondita' vera. Default 1.4 (non piu' 1.0), in coppia con "
                              "--relief-scale 0.75: compensa in contrasto la minore profondita' vera")
    parser.add_argument("--maria-crater-suppression", type=float, default=0.75,
                         help="probabilita' massima (per i crateri piu' grandi, in piena zona mare) che un cratere venga 'allagato' e scartato")
    parser.add_argument("--maria-density-mult", type=float, default=1.0,
                         help="fattore sulla densita' complessiva di crateri nei mari, indipendente dalla taglia "
                              "(1.0 = nessun effetto, 0.5 = meta' crateri rispetto agli altopiani, in piena zona mare)")
    parser.add_argument("--highland-roughness-mult", type=float, default=2.0,
                         help="rugosita' fine dell'altopiano (fuori da crateri e mari), come multiplo di quella dei "
                              "mari (giudicata gia' 'perfetta' cosi' com'e'). Default 2.0 = il doppio dei mari. Prima "
                              "di questo parametro l'altopiano teneva l'intera ampiezza di --base-amp-frac senza "
                              "alcuna riduzione (circa 6x i mari, coi default). Nessun effetto con --no-maria")
    parser.add_argument("--shadow-step", type=float, default=1.0,
                         help="moltiplicatore del passo di campionamento del ray-marching per le ombre portate "
                              "(vedi il commento in generate_moon: lo step effettivo si adatta gia' automaticamente "
                              "a --grid per evitare un costo O(grid^3) — questo e' un fattore ULTERIORE sopra quello, "
                              "1.0=default. Valori piu' alti = ombre piu' grezze ma piu' veloci da calcolare, utile "
                              "per provare rapidamente griglie molto grandi")
    parser.add_argument("--out", type=str, default="out_moons")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(args.n, 3)
    nrows = int(np.ceil(args.n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(args.n):
        seed_i = int(master_rng.integers(0, 2**31 - 1))
        img, meta = generate_moon(grid_size=args.grid, seed=seed_i, phase_deg=args.phase,
                                   n_craters=(args.n_craters_min, args.n_craters_max),
                                   max_crater_diam_km=args.max_crater_km,
                                   min_crater_diam_km=args.min_crater_km,
                                   size_freq_exponent=args.exponent,
                                   n_large_craters_mean=args.n_large_craters_mean,
                                   large_crater_min_km=args.large_crater_min_km,
                                   large_crater_max_km=args.large_crater_max_km,
                                   crater_relief_scale=args.relief_scale,
                                   crater_foreshorten_strength=args.crater_foreshorten_strength,
                                   avoid_crater_boundary_intersections=not args.allow_crater_overlap,
                                   base_amp_frac=args.base_amp_frac,
                                   terrain_beta=args.terrain_beta,
                                   enable_maria=not args.no_maria,
                                   maria_coverage=args.maria_coverage,
                                   maria_style=args.maria_style,
                                   maria_flood_smooth_px=args.maria_flood_smooth,
                                   maria_smoothing=args.maria_smoothing,
                                   maria_crater_suppression=args.maria_crater_suppression,
                                   maria_density_mult=args.maria_density_mult,
                                   shadow_step=args.shadow_step,
                                   albedo_contrast=args.albedo_contrast,
                                   relief_strength=args.relief_strength,
                                   highland_roughness_mult=args.highland_roughness_mult,
                                   add_crater_mare=args.add_crater_mare,
                                   crater_mare_diam_km=args.crater_mare_diam_km,
                                   crater_mare_max_coverage=args.crater_mare_max_coverage,
                                   maria_n_basins=args.maria_n_basins,
                                   maria_edge_width_frac=args.maria_edge_width)
        Image.fromarray((img * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"moon_{i:02d}_seed{seed_i}_phase{meta['phase_deg']:.0f}.png"))
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        maria_info = f"mare {meta['maria_coverage']*100:.0f}%" if meta['enable_maria'] else "no mari"
        axes[i].set_title(f"seed={seed_i}  fase={meta['phase_deg']:.0f}° ({phase_name(meta['phase_deg'])})\n"
                           f"{meta['n_craters']} crateri ({meta['n_large_craters']} grandi), {maria_info}", fontsize=9)
        axes[i].axis("off")
        print(f"[{i}] seed={seed_i}  fase={meta['phase_deg']:.1f} ({phase_name(meta['phase_deg'])})  "
              f"n_crateri={meta['n_craters']} (di cui {meta['n_large_craters']} grandi)")

    for j in range(args.n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.1)
    grid_path = os.path.join(args.out, "griglia_anteprima.png")
    plt.savefig(grid_path, dpi=130)
    print(f"\n[out] Anteprima a griglia salvata in: {grid_path}")
    print(f"[out] Immagini singole salvate in: {args.out}/moon_*.png")


if __name__ == "__main__":
    main()
