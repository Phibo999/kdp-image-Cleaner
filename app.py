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
    Convertit un SVG (URL) en PNG binaire, redimensionné au format KDP (dpi) + marges.

    Body JSON attendu (compatible Make):
    {
      "svg_url": "https://....svg",
      "format": "8.5x11",
      "dpi": 300,
      "margins_mm": 0
    }

    Retour: image/png (binaire)
    """
    try:
        data = request.get_json(silent=True) or {}
        svg_url = data.get("svg_url") or data.get("input_url")  # tolérance si tu réutilises un champ
        format_kdp = data.get("format", "8.5x11")
        dpi = int(data.get("dpi", 300) or 300)
        margins_mm = float(data.get("margins_mm", 0) or 0)

        if not svg_url:
            return jsonify({"error": "svg_url is required"}), 400

        # bornes de sécurité
        if dpi < 72:
            dpi = 72
        if dpi > 600:
            dpi = 600
        if margins_mm < 0:
            margins_mm = 0

        target_w, target_h = kdp_target_pixels(format_kdp, dpi)

        # marges en pixels
        margin_px = int(round(margins_mm * dpi / 25.4))
        usable_w = max(1, target_w - 2 * margin_px)
        usable_h = max(1, target_h - 2 * margin_px)

        # download svg
        svg_bytes = download_bytes(svg_url)

        # calc ratio pour "contain"
        ratio = parse_svg_ratio(svg_bytes)
        target_ratio = usable_w / usable_h

        if ratio > target_ratio:
            render_w = usable_w
            render_h = max(1, int(round(usable_w / ratio)))
        else:
            render_h = usable_h
            render_w = max(1, int(round(usable_h * ratio)))

        # render SVG -> PNG (RGBA)
        rendered_png = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=render_w,
            output_height=render_h
        )

        # compose sur un fond blanc KDP
        fg = Image.open(BytesIO(rendered_png)).convert("RGBA")
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        x = (target_w - fg.width) // 2
        y = (target_h - fg.height) // 2

        # alpha composite
        canvas.paste(fg, (x, y), fg)

        out = BytesIO()
        canvas.save(out, format="PNG", optimize=True)
        out.seek(0)

        return send_file(out, mimetype="image/png", as_attachment=False)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download svg: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"svg_to_png processing error: {str(e)}"}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
