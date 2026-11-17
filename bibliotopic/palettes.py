"""Paletas de color de las figuras: una sola paleta por artículo, elegida automáticamente según la temática.

Cada paleta tiene 8 colores categóricos validados (orden fijo; banda de luminosidad OKLCH 0.43–0.77, croma
≥ 0.10, separación entre colores contiguos ΔE ≥ 8 en OKLab ×100 con simulación de protanopía y deuteranopía
de Machado et al., 2009, y ΔE ≥ 15 con visión normal). De ellos se derivan los papeles que usan todas las
figuras: color principal y secundario, polos divergentes (creciente / decreciente) y escalas secuenciales de
un solo tono (claro → oscuro, luminosidad monótona). Así, todas las figuras de un artículo comparten paleta y
cada tema conserva su color en todas ellas.

La paleta se elige por las palabras de la búsqueda y las palabras clave más frecuentes; la variante (qué
color es el principal) depende del propio corpus, de modo que dos artículos de temática parecida no quedan
idénticos. Puede fijarse a mano en la interfaz o en config.json («palette»)."""
from __future__ import annotations

import hashlib
import math
import re

PALETTES = {
    "earth": dict(name="Tierra", topic="agricultura, alimentos, suelos, bosques y ecología",
                  cat=["#277539", "#db9e2e", "#b44e2e", "#8dab54", "#06658e", "#df84a8", "#6b4690", "#39bab4"]),
    "ocean": dict(name="Océano", topic="agua, mares, pesca, clima y ciencias de la Tierra",
                  cat=["#2063b0", "#31c3b5", "#d55948", "#564597", "#d6a62e", "#17895b", "#bf5da2", "#49abd6"]),
    "health": dict(name="Salud", topic="medicina, salud pública, nutrición y biociencias",
                   cat=["#009390", "#e7624b", "#305eb7", "#65c98c", "#853e80", "#e4a339", "#bc3f53", "#50a9d9"],
                   alt=["#305eb7", "#853e80", "#50a9d9", "#bc3f53", "#65c98c", "#009390", "#e4a339", "#e7624b"]),
    "tech": dict(name="Tecnología", topic="ingeniería, energía, computación y materiales",
                 cat=["#2e5297", "#e87f25", "#25aabe", "#b63039", "#886acf", "#d2b223", "#298646", "#da7baa"]),
    "social": dict(name="Sociedad", topic="educación, economía, gestión, turismo y ciencias sociales",
                   cat=["#6e4199", "#d4a73e", "#088f77", "#d97230", "#2e69b2", "#da709d", "#65a556", "#904529"]),
    "classic": dict(name="Clásica", topic="neutra (cualquier temática)",
                    cat=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
}
ORDER = ["earth", "ocean", "health", "tech", "social", "classic"]
NEUTRAL = "#b5b5b5"      # sin tendencia, outliers, enlaces entre grupos
NEUTRAL_DARK = "#555555"  # trayectorias, barras de error, flechas

# Raíces (inglés y español) que asocian un corpus a cada paleta. Se cuentan como coincidencias de palabra
# completa o prefijo en las ecuaciones de búsqueda, los términos de búsqueda y las palabras clave principales.
THEMES = {
    "earth": ["agri", "agro", "crop", "farm", "soil", "plant", "forest", "coffee", "cocoa", "cacao", "maize",
              "corn", "rice", "wheat", "banana", "fruit", "seed", "fertiliz", "pest", "harvest", "yield", "cultiv",
              "livestock", "cattle", "food", "biodivers", "ecolog", "ecosystem", "land use", "tree", "shade",
              "organic", "compost", "rust", "fung", "weed", "cultivo", "suelo", "bosque", "café", "cafe", "semilla",
              "ganad", "aliment", "agroforest", "pollinat", "insect", "bee", "wildlife"],
    "ocean": ["water", "marine", "ocean", "sea", "fish", "aquacult", "coast", "river", "lake", "wetland",
              "hydro", "rain", "drought", "flood", "climate", "glacier", "coral", "mangrove", "sediment",
              "groundwater", "irrigat", "watershed", "basin", "agua", "mar", "pesca", "río", "rio", "clima",
              "microplastic", "plastic", "pollut", "estuar", "algae", "plankton"],
    "health": ["health", "medic", "clinic", "patient", "disease", "cancer", "covid", "sars", "virus", "infect",
               "hospital", "nurs", "therap", "drug", "pharma", "diabet", "obes", "nutrition", "diet",
               "mental", "depress", "epidemi", "vaccin", "surg", "cardio", "gene", "protein", "cell", "bacteri",
               "antibiot", "salud", "enferm", "médic", "medic", "paciente", "nutrici"],
    "tech": ["engineer", "energy", "solar", "wind", "battery", "electric", "power", "machine learning",
             "deep learning", "artificial intelligence", "neural", "algorithm", "comput", "software", "robot",
             "sensor", "iot", "internet of things", "blockchain", "material", "nano", "polymer", "manufactur",
             "industry 4", "digital", "network", "optimiz", "fuel", "hydrogen", "catalys", "energía", "ingenier",
             "inteligencia artificial"],
    "social": ["education", "student", "teach", "learning", "school", "universit", "econom", "financ", "market",
               "business", "management", "entrepreneur", "touris", "policy", "governance", "politic", "social",
               "community", "poverty", "gender", "employ", "labor", "consumer", "marketing", "innovation",
               "sustainab", "law", "culture", "educa", "estudiant", "turism", "empresa", "gestión", "pobreza"],
}


# ------------------------------------------------------------------ color (OKLCH)
def _hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _enc(x):
    x = max(0.0, min(1.0, x))
    return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


def hex2oklch(h):
    r, g, b = (_lin(c) for c in _hex2rgb(h))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def _oklch2lin(L, C, h):
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def oklch2hex(L, C, h):
    """OKLCH -> hex, reduciendo el croma hasta entrar en la gama sRGB."""
    c = C
    while True:
        rgb = _oklch2lin(L, c, h)
        if all(-1e-4 <= v <= 1 + 1e-4 for v in rgb) or c <= 0:
            break
        c = max(0.0, c - 0.002)
    return "#" + "".join(f"{round(_enc(v) * 255):02x}" for v in rgb)


def ramp(base_hex, n=5, light=0.74, dark=0.30):
    """Escala secuencial de un solo tono: luminosidad monótona de clara a oscura; el croma crece hacia los
    tonos medios y se reduce en los extremos para seguir en la gama sRGB."""
    _, C, h = hex2oklch(base_hex)
    C = max(C, 0.10)
    out = []
    for i in range(n):
        t = i / (n - 1)
        L = light + (dark - light) * t
        out.append(oklch2hex(L, C * (0.12 + 0.88 * math.sin(math.pi * min(1.0, 0.15 + 0.85 * t) * 0.62)), h))
    return out


def tint(base_hex, L=0.95, k=0.25):
    _, C, h = hex2oklch(base_hex)
    return oklch2hex(L, C * k, h)


# ------------------------------------------------------------------ paleta efectiva
class Palette:
    def __init__(self, key="classic", variant=0):
        p = PALETTES.get(key, PALETTES["classic"])
        self.key, self.name, self.variant = key if key in PALETTES else "classic", p["name"], int(variant) % 2
        base = list(p["cat"])
        # variante 1: el tercer color pasa a ser el principal (rotación de dos posiciones o, si la rotación
        # no supera la validación, un orden alternativo validado)
        self.cat = (p.get("alt") or base[2:] + base[:2]) if self.variant else base
        self.primary, self.secondary, self.third = self.cat[0], self.cat[1], self.cat[2]
        warm = [c for c in self.cat if (hex2oklch(c)[2] < 100 or hex2oklch(c)[2] > 330)]
        cool = [c for c in self.cat if 150 <= hex2oklch(c)[2] <= 300]
        self.increase = warm[0] if warm else self.secondary  # tendencia creciente (cálido)
        self.decrease = cool[0] if cool else self.primary    # tendencia decreciente (frío)
        self.seq = ramp(self.primary)
        self.seq2 = ramp(self.secondary)
        self.neutral, self.neutral_dark = NEUTRAL, NEUTRAL_DARK
        self.box_in, self.box_out = tint(self.primary), tint(self.secondary)

    def color(self, i):
        """Color categórico i (orden fijo); con más de 8 categorías se repite el ciclo (las figuras rotulan
        cada categoría directamente)."""
        return self.cat[int(i) % len(self.cat)]

    def info(self):
        return {"key": self.key, "name": self.name, "variant": self.variant, "categorical": self.cat,
                "primary": self.primary, "secondary": self.secondary, "increase": self.increase,
                "decrease": self.decrease, "sequential": self.seq}


def _count(text, stem):
    return len(re.findall(r"(?<![a-záéíóúñ])" + re.escape(stem), text))


def choose(query_texts=(), keywords=(), weights=None):
    """Paleta según la temática. query_texts: ecuaciones y términos de búsqueda (peso 3); keywords: palabras
    clave más frecuentes (con su frecuencia si se da weights). Devuelve (clave, variante, puntuaciones)."""
    q = " ".join(t for t in query_texts if t).lower()
    kws = [str(k).lower() for k in keywords]
    w = list(weights) if weights is not None else [1] * len(kws)
    tot = sum(w) or 1
    scores = {}
    for key, stems in THEMES.items():
        s = 3.0 * sum(_count(q, st) > 0 for st in stems)
        s += 10.0 * sum(wi for k, wi in zip(kws, w) if any(_count(k, st) for st in stems)) / tot
        scores[key] = round(s, 3)
    best = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else "classic"
    sig = "|".join(kws[:20]) or q or "corpus"
    variant = int(hashlib.md5(sig.encode()).hexdigest(), 16) % 2
    return best, variant, scores


def catalogue():
    """Para la interfaz: clave, nombre, temática y colores de cada paleta (variante 0)."""
    return [{"key": k, "name": PALETTES[k]["name"], "topic": PALETTES[k]["topic"], "cat": PALETTES[k]["cat"],
             "alt": PALETTES[k].get("alt") or PALETTES[k]["cat"][2:] + PALETTES[k]["cat"][:2]} for k in ORDER]
