from flask import Flask, request, jsonify, send_file
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import cairosvg

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
    Convertit un SVG en PNG prêt pour KDP

    Body JSON attendu :
    {
        "svg_url": "https://...",
        "format": "8.5x11",
        "dpi": 300,
        "margins_mm": 0
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        svg_url = data.get("svg_url")
        format_kdp = data.get("format", "8.5x11")
        dpi = int(data.get("dpi", 300))
        margins_mm = float(data.get("margins_mm", 0))

        if not svg_url:
            return jsonify({"error": "svg_url is required"}), 400

        if format_kdp not in KDP_FORMATS:
            return jsonify({"error": f"Unsupported format {format_kdp}"}), 400

        # Dimensions cible KDP
        target = KDP_FORMATS[format_kdp]
        width_px = target["width"]
        height_px = target["height"]

        # Marges en pixels
        margin_px = int(margins_mm * dpi / 25.4)
        usable_width = width_px - (2 * margin_px)
        usable_height = height_px - (2 * margin_px)

        if usable_width <= 0 or usable_height <= 0:
            return jsonify({"error": "Margins too large for selected format"}), 400

        # Télécharger le SVG
        svg_response = requests.get(svg_url, timeout=60)
        svg_response.raise_for_status()
        svg_bytes = svg_response.content

        # Rasterisation SVG → PNG (zone utile)
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=usable_width,
            output_height=usable_height,
            dpi=dpi,
            background_color="white"
        )

        # Charger PNG rasterisé
        raster_img = Image.open(BytesIO(png_bytes)).convert("L")

        # Image finale KDP (fond blanc)
        final_img = Image.new("L", (width_px, height_px), 255)

        x = (width_px - raster_img.width) // 2
        y = (height_px - raster_img.height) // 2
        final_img.paste(raster_img, (x, y))

        # Export PNG
        buffer = BytesIO()
        final_img.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="image/png",
            as_attachment=False
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download SVG: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"SVG to PNG processing error: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
