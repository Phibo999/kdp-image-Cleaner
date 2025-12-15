from flask import Flask, request, jsonify, send_file
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import cairosvg
import re

app = Flask(__name__)

# Configuration des formats KDP (en pixels à 300 DPI)
KDP_FORMATS = {
    "6x9": {"width": 1800, "height": 2700},
    "8x10": {"width": 2400, "height": 3000},
    "8.5x11": {"width": 2550, "height": 3300},
    "5x8": {"width": 1500, "height": 2400},
    "5.5x8.5": {"width": 1650, "height": 2550},
    "6x6": {"width": 1800, "height": 1800},
    "8x8": {"width": 2400, "height": 2400},
}
def download_bytes(url: str) -> bytes:
    """Télécharge du contenu binaire depuis une URL (SVG)"""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def parse_svg_ratio(svg_bytes: bytes) -> float:
    """
    Essaie de déduire le ratio largeur/hauteur du SVG.
    Priorité: viewBox, sinon width/height si présents, sinon 1.0
    """
    try:
        s = svg_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return 1.0

    # viewBox="minx miny w h"
    m = re.search(r'viewBox\s*=\s*"[^"]*"', s, flags=re.IGNORECASE)
    if m:
        # extraire 4 nombres du viewBox
        nums = re.findall(r"[-+]?\d*\.?\d+", m.group(0))
        if len(nums) >= 4:
            w = float(nums[2])
            h = float(nums[3])
            if h > 0:
                return max(w / h, 0.0001)

    # width="123px" height="456px"
    mw = re.search(r'width\s*=\s*"([^"]+)"', s, flags=re.IGNORECASE)
    mh = re.search(r'height\s*=\s*"([^"]+)"', s, flags=re.IGNORECASE)

    def to_px(val: str) -> float:
        # garde juste le nombre, ignore unités
        n = re.findall(r"[-+]?\d*\.?\d+", val)
        return float(n[0]) if n else 0.0

    if mw and mh:
        w = to_px(mw.group(1))
        h = to_px(mh.group(1))
        if h > 0 and w > 0:
            return max(w / h, 0.0001)

    return 1.0

def kdp_target_pixels(format_kdp: str, dpi: int) -> tuple[int, int]:
    """Calcule largeur/hauteur cible en pixels pour un format KDP à un DPI donné."""
    # fallback si format inconnu
    fmt = (format_kdp or "8.5x11").strip()
    if fmt in KDP_FORMATS:
        # Tu avais KDP_FORMATS en pixels à 300 DPI. On convertit proprement selon dpi.
        base_w = KDP_FORMATS[fmt]["width"]
        base_h = KDP_FORMATS[fmt]["height"]
        scale = float(dpi) / 300.0
        return int(round(base_w * scale)), int(round(base_h * scale))

    # Sinon, tente de parser "8.5x11"
    m = re.match(r"^\s*(\d+(\.\d+)?)\s*x\s*(\d+(\.\d+)?)\s*$", fmt)
    if m:
        w_in = float(m.group(1))
        h_in = float(m.group(3))
        return int(round(w_in * dpi)), int(round(h_in * dpi))

    # fallback final
    base_w = KDP_FORMATS["8.5x11"]["width"]
    base_h = KDP_FORMATS["8.5x11"]["height"]
    scale = float(dpi) / 300.0
    return int(round(base_w * scale)), int(round(base_h * scale))


def download_image(url):
    """Télécharge une image depuis une URL"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def clean_image(img, cleaning_strength="Medium"):
    """Nettoie l'image : supprime les gris, le bruit, etc."""
    
    # Convertir en numpy array
    img_array = np.array(img.convert('RGB'))
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Définir les seuils selon la force de nettoyage
    thresholds = {
        "Light": 200,
        "Medium": 180,
        "Strong": 160,
        "Extreme": 140
    }
    threshold = thresholds.get(cleaning_strength, 180)
    
    # Appliquer le seuillage pour obtenir du noir et blanc pur
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Supprimer le bruit avec des opérations morphologiques
    kernel = np.ones((2, 2), np.uint8)
    
    # Ouverture pour supprimer les petits points
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fermeture pour combler les petits trous dans les lignes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Supprimer les petits artefacts isolés
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(cleaned), connectivity=8
    )
    
    min_size = 50  # Taille minimale des composants à garder
    mask = np.ones(cleaned.shape, dtype=np.uint8) * 255
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 0
    
    cleaned = mask
    
    return Image.fromarray(cleaned)

def resize_for_kdp(img, format_kdp, margins_mm=5):
    """Redimensionne l'image pour le format KDP avec marges"""
    
    if format_kdp not in KDP_FORMATS:
        format_kdp = "8.5x11"  # Format par défaut
    
    target = KDP_FORMATS[format_kdp]
    
    # Calculer les marges en pixels (300 DPI)
    margin_pixels = int(margins_mm * 300 / 25.4)
    
    # Zone utilisable
    usable_width = target["width"] - (2 * margin_pixels)
    usable_height = target["height"] - (2 * margin_pixels)
    
    # Calculer le ratio pour que l'image tienne dans la zone utilisable
    img_ratio = img.width / img.height
    target_ratio = usable_width / usable_height
    
    if img_ratio > target_ratio:
        # Image plus large : ajuster par la largeur
        new_width = usable_width
        new_height = int(usable_width / img_ratio)
    else:
        # Image plus haute : ajuster par la hauteur
        new_height = usable_height
        new_width = int(usable_height * img_ratio)
    
    # Redimensionner l'image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Créer l'image finale avec fond blanc
    final_img = Image.new('L', (target["width"], target["height"]), 255)
    
    # Centrer l'image
    x = (target["width"] - new_width) // 2
    y = (target["height"] - new_height) // 2
    
    final_img.paste(img_resized, (x, y))
    
    return final_img

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({"status": "healthy", "service": "KDP Image Cleaner"})

@app.route('/clean-kdp-image', methods=['POST'])
def clean_kdp_image():
    """
    Endpoint principal pour nettoyer les images KDP
    
    Body JSON attendu:
    {
        "input_url": "https://...",
        "format": "8.5x11",
        "cleaning": "Medium",
        "margins_mm": 5,
        "vectorize": false
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        input_url = data.get('input_url')
        format_kdp = data.get('format', '8.5x11')
        cleaning_strength = data.get('cleaning', 'Medium')
        margins_mm = data.get('margins_mm', 5)
        
        if not input_url:
            return jsonify({"error": "input_url is required"}), 400
        
        # 1. Télécharger l'image
        img = download_image(input_url)
        
        # 2. Nettoyer l'image
        cleaned_img = clean_image(img, cleaning_strength)
        
        # 3. Redimensionner pour KDP
        final_img = resize_for_kdp(cleaned_img, format_kdp, margins_mm)
        
        # 4. Préparer la réponse (image en binaire)
        img_buffer = BytesIO()
        final_img.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/formats', methods=['GET'])
def list_formats():
    """Liste les formats KDP disponibles"""
    return jsonify({
        "formats": list(KDP_FORMATS.keys()),
        "cleaning_levels": ["Light", "Medium", "Strong", "Extreme"]
    })

@app.route('/svg_to_png', methods=['POST'])
def svg_to_png():
    """
    Convertit un SVG en PNG KDP-ready (Make compatible)
    Reçoit JSON, renvoie PNG binaire
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        svg_url = data.get("svg_url")
        format_kdp = data.get("format", "8.5x11")
        dpi = int(data.get("dpi", 300))
        margins_mm = int(data.get("margins_mm", 0))

        if not svg_url:
            return jsonify({"error": "svg_url is required"}), 400

        # Télécharger le SVG
        r = requests.get(svg_url, timeout=60)
        r.raise_for_status()
        svg_data = r.content

        # Dimensions KDP
        if format_kdp not in KDP_FORMATS:
            return jsonify({"error": "Invalid format"}), 400

        target = KDP_FORMATS[format_kdp]
        width_px = target["width"]
        height_px = target["height"]

        # Marges
        margin_px = int(margins_mm * dpi / 25.4)
        usable_w = width_px - (2 * margin_px)
        usable_h = height_px - (2 * margin_px)

        # Conversion SVG → PNG
        png_bytes = cairosvg.svg2png(
            bytestring=svg_data,
            output_width=usable_w,
            output_height=usable_h,
            dpi=dpi,
            background_color="white"
        )

        # Charger image
        img = Image.open(BytesIO(png_bytes)).convert("L")

        # Canvas final KDP
        final_img = Image.new("L", (width_px, height_px), 255)
        x = (width_px - img.width) // 2
        y = (height_px - img.height) // 2
        final_img.paste(img, (x, y))

        buf = BytesIO()
        final_img.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
def download_svg(url: str) -> bytes:
    """Télécharge un SVG depuis une URL et retourne les bytes."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def _normalize_dpi(dpi):
    try:
        dpi = int(dpi)
    except Exception:
        dpi = 300
    return max(72, min(dpi, 1200))  # bornes raisonnables

def _normalize_margins_mm(margins_mm):
    try:
        margins_mm = float(margins_mm)
    except Exception:
        margins_mm = 0.0
    return max(0.0, min(margins_mm, 25.0))  # bornes raisonnables

def svg_bytes_to_pil(svg_bytes: bytes, dpi: int) -> Image.Image:
    """
    Convertit un SVG en PIL.Image via CairoSVG.
    Retourne une image RGBA.
    """
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi)
    return Image.open(BytesIO(png_bytes)).convert("RGBA")

def fit_on_kdp_canvas(img_rgba: Image.Image, format_kdp: str, dpi: int, margins_mm: float) -> Image.Image:
    """
    Place l'image rendue (RGBA) au centre d'une page KDP (fond blanc),
    en respectant format + dpi + marges, en conservant le ratio.
    Retourne une image L (niveaux de gris) prête à binariser.
    """
    # Dimensions page en px pour le dpi demandé
    # On part de tes formats définis en 300dpi, puis on scale au dpi voulu.
    if format_kdp not in KDP_FORMATS:
        format_kdp = "8.5x11"

    base = KDP_FORMATS[format_kdp]
    scale = dpi / 300.0
    page_w = int(round(base["width"] * scale))
    page_h = int(round(base["height"] * scale))

    margin_px = int(round(margins_mm * dpi / 25.4))
    usable_w = max(1, page_w - 2 * margin_px)
    usable_h = max(1, page_h - 2 * margin_px)

    # Aplatir sur fond blanc
    white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(white_bg, img_rgba).convert("RGB")

    # Fit dans zone utilisable
    img_ratio = flat.width / max(1, flat.height)
    zone_ratio = usable_w / max(1, usable_h)

    if img_ratio > zone_ratio:
        new_w = usable_w
        new_h = int(round(usable_w / img_ratio))
    else:
        new_h = usable_h
        new_w = int(round(usable_h * img_ratio))

    resized = flat.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)

    # Canvas final
    canvas = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    x = (page_w - resized.width) // 2
    y = (page_h - resized.height) // 2
    canvas.paste(resized, (x, y))

    return canvas.convert("L")

def binarize_strict(img_l: Image.Image, threshold: int = 245) -> Image.Image:
    """
    Binarisation agressive pour supprimer l’anti-aliasing (gris) issu du rendu SVG.
    0 = noir, 255 = blanc.
    """
    arr = np.array(img_l)
    arr = np.where(arr < threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

@app.route('/svg_to_png', methods=['POST'])
def svg_to_png():
    """
    Convertit un SVG (URL) en PNG binaire, KDP-ready.

    Body JSON attendu:
    {
      "svg_url": "https://drive.google.com/uc?id=...&export=download",
      "format": "8.5x11",
      "dpi": 300,
      "margins_mm": 0,
      "binarize": true,
      "threshold": 245
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        svg_url = data.get("svg_url")
        format_kdp = data.get("format", "8.5x11")
        dpi = _normalize_dpi(data.get("dpi", 300))
        margins_mm = _normalize_margins_mm(data.get("margins_mm", 0))
        binarize = bool(data.get("binarize", True))
        threshold = int(data.get("threshold", 245))

        if not svg_url:
            return jsonify({"error": "svg_url is required"}), 400

        # 1) download svg
        svg_bytes = download_svg(svg_url)

        # 2) render svg -> PIL (RGBA)
        rendered = svg_bytes_to_pil(svg_bytes, dpi=dpi)

        # 3) place on KDP canvas
        page_l = fit_on_kdp_canvas(rendered, format_kdp=format_kdp, dpi=dpi, margins_mm=margins_mm)

        # 4) strict binarize (remove grays)
        if binarize:
            page_l = binarize_strict(page_l, threshold=threshold)

        # 5) return PNG binary
        buf = BytesIO()
        page_l.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=False)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download SVG: {str(e)}"}), 400
    except Exception as e:
        # utile pour debug Make
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
