from flask import Flask, request, jsonify, send_file
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import cairosvg
import re
import base64
from typing import Tuple, Dict, Optional
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OBSIDIAN")

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


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE OBSIDIAN PRO CLEANUP - Pipeline Professionnel 10 étapes
# ═══════════════════════════════════════════════════════════════════════════════

class ObsidianProCleanup:
    """
    Pipeline de nettoyage professionnel pour livres de coloriage KDP
    
    Étapes du pipeline:
    1. Auto-crop (suppression bordures parasites)
    2. Deskew (correction inclinaison)
    3. Denoising (débruitage)
    4. Binarization (seuillage adaptatif)
    5. Line detection & removal (lignes parasites)
    6. Morphological cleanup (ouverture/fermeture)
    7. Despeckle (suppression micro-artefacts)
    8. Line boost (épaississement traits)
    9. Border cleanup (suppression cadres ajoutés)
    10. Final validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.default_config()
        
    @staticmethod
    def default_config() -> Dict:
        return {
            # Auto-crop
            "crop_margin_px": 20,
            "crop_white_threshold": 250,
            
            # Deskew
            "deskew_enabled": True,
            "deskew_max_angle": 5.0,
            
            # Denoising
            "denoise_strength": 3,
            "denoise_template_window": 7,
            "denoise_search_window": 21,
            
            # Binarization
            "binarize_method": "adaptive",  # "otsu", "adaptive", "sauvola"
            "binarize_block_size": 35,
            "binarize_c_constant": 10,
            
            # Line detection (parasites)
            "line_min_length": 200,
            "line_max_thickness": 8,
            "line_removal_enabled": True,
            
            # Morphology
            "morph_kernel_size": 2,
            "morph_iterations": 1,
            
            # Despeckle
            "despeckle_min_area": 10,
            "despeckle_max_area": 100,
            
            # Line boost
            "line_boost_px": 0,  # 0 = désactivé par défaut
            "line_boost_kernel": "ellipse",
            
            # Border cleanup
            "border_check_width": 50,
            "border_black_threshold": 0.7,
            
            # Validation
            "corner_check_size": 30,
            "max_black_ratio": 0.15,
        }
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Exécute le pipeline complet sur une image.
        
        Args:
            image: Image en niveaux de gris (numpy array)
            
        Returns:
            Tuple (image traitée, métriques)
        """
        metrics = {}
        
        # S'assurer que l'image est en niveaux de gris
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        original_shape = image.shape
        metrics["original_size"] = f"{original_shape[1]}x{original_shape[0]}"
        
        # 1. AUTO-CROP
        image, crop_info = self._auto_crop(image)
        metrics["crop"] = crop_info
        
        # 2. DESKEW
        if self.config["deskew_enabled"]:
            image, angle = self._deskew(image)
            metrics["deskew_angle"] = round(angle, 2)
        
        # 3. DENOISING
        image = self._denoise(image)
        metrics["denoised"] = True
        
        # 4. BINARIZATION
        image = self._binarize(image)
        metrics["binarized"] = True
        
        # 5. LINE REMOVAL (lignes parasites)
        if self.config["line_removal_enabled"]:
            image, lines_removed = self._remove_parasitic_lines(image)
            metrics["lines_removed"] = lines_removed
        
        # 6. MORPHOLOGICAL CLEANUP
        image = self._morphological_cleanup(image)
        metrics["morphology"] = True
        
        # 7. DESPECKLE
        image, specks_removed = self._despeckle(image)
        metrics["specks_removed"] = specks_removed
        
        # 8. LINE BOOST
        if self.config["line_boost_px"] > 0:
            image = self._line_boost(image)
            metrics["line_boost_px"] = self.config["line_boost_px"]
        
        # 9. BORDER CLEANUP
        image, border_cleaned = self._border_cleanup(image)
        metrics["border_cleaned"] = border_cleaned
        
        # 10. VALIDATION
        validation = self._validate(image)
        metrics["validation"] = validation
        
        metrics["final_size"] = f"{image.shape[1]}x{image.shape[0]}"
        
        return image, metrics
    
    def _auto_crop(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Étape 1: Supprime les bordures parasites (cadres noirs, marges excessives).
        """
        info = {"applied": False}
        
        # Inverser pour trouver le contenu (noir devient blanc)
        thresh = cv2.threshold(
            image, 
            self.config["crop_white_threshold"], 
            255, 
            cv2.THRESH_BINARY_INV
        )[1]
        
        # Trouver les contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            info["reason"] = "no_content_found"
            return image, info
        
        # Bounding box du contenu
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Ajouter marge de sécurité
        margin = self.config["crop_margin_px"]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Vérifier si le crop est significatif (> 5% de réduction)
        original_area = image.shape[0] * image.shape[1]
        new_area = (x2 - x1) * (y2 - y1)
        reduction = 1 - (new_area / original_area)
        
        if reduction > 0.05:
            cropped = image[y1:y2, x1:x2]
            info["applied"] = True
            info["reduction_pct"] = round(reduction * 100, 1)
            info["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
            return cropped, info
        
        info["reason"] = "minimal_reduction"
        return image, info
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Étape 2: Corrige l'inclinaison de l'image.
        """
        # Binariser pour la détection
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Détection des lignes avec Hough
        lines = cv2.HoughLinesP(
            binary, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image, 0.0
        
        # Calculer les angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Ne garder que les angles proches de l'horizontal ou vertical
                if abs(angle) < self.config["deskew_max_angle"]:
                    angles.append(angle)
                elif abs(abs(angle) - 90) < self.config["deskew_max_angle"]:
                    angles.append(angle - 90 if angle > 0 else angle + 90)
        
        if not angles:
            return image, 0.0
        
        # Angle médian
        median_angle = np.median(angles)
        
        if abs(median_angle) < 0.1:
            return image, 0.0
        
        # Rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )
        
        return rotated, median_angle
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Étape 3: Débruitage avec préservation des contours.
        """
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=self.config["denoise_strength"],
            templateWindowSize=self.config["denoise_template_window"],
            searchWindowSize=self.config["denoise_search_window"]
        )
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Étape 4: Binarisation (conversion noir/blanc pur).
        """
        method = self.config["binarize_method"]
        
        if method == "otsu":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config["binarize_block_size"],
                self.config["binarize_c_constant"]
            )
            
        elif method == "sauvola":
            # Implémentation Sauvola simplifiée
            block_size = self.config["binarize_block_size"]
            k = 0.2  # Paramètre Sauvola
            
            # Moyenne locale
            mean = cv2.blur(image.astype(np.float64), (block_size, block_size))
            
            # Écart-type local
            sq_mean = cv2.blur(image.astype(np.float64) ** 2, (block_size, block_size))
            std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
            
            # Seuil Sauvola
            R = 128  # Dynamique des niveaux de gris
            threshold = mean * (1 + k * (std / R - 1))
            
            binary = np.where(image > threshold, 255, 0).astype(np.uint8)
            
        else:
            # Fallback: seuil simple
            _, binary = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _remove_parasitic_lines(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Étape 5: Détecte et supprime les lignes droites parasites 
        (jointures de tiles Midjourney).
        """
        removed_info = {"horizontal": 0, "vertical": 0}
        
        # Inverser (traits noirs → blanc)
        inverted = cv2.bitwise_not(image)
        
        min_length = self.config["line_min_length"]
        max_thick = self.config["line_max_thickness"]
        
        # Détection lignes horizontales
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        h_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel)
        
        # Filtrer par épaisseur
        h_kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, max_thick))
        h_lines_thick = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel_thick)
        h_parasites = cv2.subtract(h_lines_thick, cv2.dilate(h_lines, np.ones((max_thick, 1), np.uint8)))
        
        # Compter pixels supprimés
        removed_info["horizontal"] = int(np.sum(h_lines > 0))
        
        # Détection lignes verticales
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
        v_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel)
        
        # Filtrer par épaisseur
        v_kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (max_thick, min_length))
        v_lines_thick = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel_thick)
        v_parasites = cv2.subtract(v_lines_thick, cv2.dilate(v_lines, np.ones((1, max_thick), np.uint8)))
        
        removed_info["vertical"] = int(np.sum(v_lines > 0))
        
        # Masque combiné des lignes parasites
        lines_mask = cv2.add(h_lines, v_lines)
        
        # Dilatation légère pour couvrir les antialiasing
        lines_mask = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Supprimer les lignes (remettre en blanc)
        result = image.copy()
        result[lines_mask > 0] = 255
        
        return result, removed_info
    
    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """
        Étape 6: Nettoyage morphologique (ouverture puis fermeture).
        """
        kernel_size = self.config["morph_kernel_size"]
        iterations = self.config["morph_iterations"]
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Ouverture (supprime petits points)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Fermeture (comble petits trous)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return cleaned
    
    def _despeckle(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Étape 7: Suppression des micro-artefacts isolés (composantes connexes).
        """
        # Inverser pour analyser les composantes noires
        inverted = cv2.bitwise_not(image)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        
        specks_removed = 0
        min_area = self.config["despeckle_min_area"]
        max_area = self.config["despeckle_max_area"]
        
        # Créer masque des composantes à supprimer
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                # Petite composante isolée → probablement du bruit
                mask[labels == i] = 255
                specks_removed += 1
        
        # Supprimer les specs (remettre en blanc)
        result = image.copy()
        result[mask > 0] = 255
        
        return result, specks_removed
    
    def _line_boost(self, image: np.ndarray) -> np.ndarray:
        """
        Étape 8: Épaississement des traits pour KDP.
        """
        boost_px = self.config["line_boost_px"]
        kernel_type = self.config["line_boost_kernel"]
        
        # Inverser (traits noirs → blancs pour dilatation)
        inverted = cv2.bitwise_not(image)
        
        # Kernel
        kernel_size = boost_px * 2 + 1
        if kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Dilatation
        dilated = cv2.dilate(inverted, kernel, iterations=1)
        
        # Ré-inverser
        return cv2.bitwise_not(dilated)
    
    def _border_cleanup(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Étape 9: Vérifie et supprime les cadres noirs parasites sur les bords.
        """
        h, w = image.shape
        check_width = self.config["border_check_width"]
        threshold = self.config["border_black_threshold"]
        
        cleaned = False
        result = image.copy()
        
        # Vérifier chaque bord
        borders = {
            "top": result[:check_width, :],
            "bottom": result[-check_width:, :],
            "left": result[:, :check_width],
            "right": result[:, -check_width:]
        }
        
        for border_name, border_region in borders.items():
            # Calculer le ratio de pixels noirs
            black_ratio = np.sum(border_region < 128) / border_region.size
            
            if black_ratio > threshold:
                # Bord trop noir → le blanchir
                if border_name == "top":
                    result[:check_width, :] = 255
                elif border_name == "bottom":
                    result[-check_width:, :] = 255
                elif border_name == "left":
                    result[:, :check_width] = 255
                elif border_name == "right":
                    result[:, -check_width:] = 255
                cleaned = True
                logger.info(f"Border cleanup: {border_name} cleared (black ratio: {black_ratio:.2%})")
        
        return result, cleaned
    
    def _validate(self, image: np.ndarray) -> Dict:
        """
        Étape 10: Validation finale de l'image.
        """
        h, w = image.shape
        corner_size = self.config["corner_check_size"]
        
        # Vérifier les coins (doivent être blancs)
        corners = {
            "top_left": image[:corner_size, :corner_size],
            "top_right": image[:corner_size, -corner_size:],
            "bottom_left": image[-corner_size:, :corner_size],
            "bottom_right": image[-corner_size:, -corner_size:]
        }
        
        corners_white = all(
            np.mean(corner) > 240
            for corner in corners.values()
        )
        
        # Ratio noir/blanc global
        black_ratio = np.sum(image < 128) / image.size
        black_ok = black_ratio <= self.config["max_black_ratio"]
        
        # Verdict
        if corners_white and black_ok:
            verdict = "GO"
        elif not corners_white:
            verdict = "REVIEW"
        else:
            verdict = "REVIEW"
        
        return {
            "corners_white": corners_white,
            "black_ratio": round(black_ratio * 100, 2),
            "verdict": verdict
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

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
    fmt = (format_kdp or "8.5x11").strip()
    if fmt in KDP_FORMATS:
        base_w = KDP_FORMATS[fmt]["width"]
        base_h = KDP_FORMATS[fmt]["height"]
        scale = float(dpi) / 300.0
        return int(round(base_w * scale)), int(round(base_h * scale))

    m = re.match(r"^\s*(\d+(\.\d+)?)\s*x\s*(\d+(\.\d+)?)\s*$", fmt)
    if m:
        w_in = float(m.group(1))
        h_in = float(m.group(3))
        return int(round(w_in * dpi)), int(round(h_in * dpi))

    base_w = KDP_FORMATS["8.5x11"]["width"]
    base_h = KDP_FORMATS["8.5x11"]["height"]
    scale = float(dpi) / 300.0
    return int(round(base_w * scale)), int(round(base_h * scale))


def download_image(url):
    """Télécharge une image depuis une URL"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION : Récupération image (base64 OU URL)
# ═══════════════════════════════════════════════════════════════════════════════

def get_image_from_request(data: dict) -> Image.Image:
    """
    Récupère l'image soit depuis base64 (prioritaire), soit depuis une URL.
    
    Paramètres attendus dans data:
    - input_base64: string base64 de l'image (PRIORITAIRE)
    - input_url: URL de l'image (fallback)
    
    Retourne:
    - PIL.Image
    """
    
    # Option 1 : Base64 (prioritaire - plus fiable, pas de blocage Google)
    if data.get('input_base64'):
        try:
            # Nettoyer le base64 si préfixe data:image/...
            b64_data = data['input_base64']
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            
            # Décoder
            image_bytes = base64.b64decode(b64_data)
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Erreur décodage base64: {str(e)}")
    
    # Option 2 : URL (fallback)
    elif data.get('input_url'):
        try:
            response = requests.get(data['input_url'], timeout=60)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Erreur téléchargement URL: {str(e)}")
    
    else:
        raise ValueError("Aucune source d'image fournie (input_base64 ou input_url requis)")


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION : Récupération SVG (base64 OU URL)
# ═══════════════════════════════════════════════════════════════════════════════

def get_svg_from_request(data: dict) -> bytes:
    """
    Récupère le SVG soit depuis base64 (prioritaire), soit depuis une URL.
    
    Paramètres attendus dans data:
    - svg_base64: string base64 du SVG (PRIORITAIRE)
    - svg_url: URL du SVG (fallback)
    
    Retourne:
    - bytes du contenu SVG
    """
    
    # Option 1 : Base64 (prioritaire - plus fiable, pas de blocage Google)
    if data.get('svg_base64'):
        try:
            # Nettoyer le base64 si préfixe data:image/...
            b64_data = data['svg_base64']
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            
            # Décoder
            svg_bytes = base64.b64decode(b64_data)
            
            # Vérifier que c'est bien du SVG
            svg_str = svg_bytes.decode('utf-8', errors='ignore').lower()
            if '<svg' not in svg_str:
                raise ValueError("Le contenu base64 ne semble pas être un SVG valide")
            
            return svg_bytes
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Erreur décodage base64 SVG: {str(e)}")
    
    # Option 2 : URL (fallback)
    elif data.get('svg_url'):
        try:
            response = requests.get(data['svg_url'], timeout=60)
            response.raise_for_status()
            svg_bytes = response.content
            
            # Vérifier que c'est bien du SVG (pas une page HTML Google)
            content_start = svg_bytes[:200].decode('utf-8', errors='ignore').lower()
            
            if '<html' in content_start or '<!doctype html' in content_start:
                raise ValueError(
                    "L'URL retourne une page HTML au lieu du SVG. "
                    "Google Drive bloque le téléchargement direct. "
                    "Utilisez svg_base64 à la place."
                )
            
            if '<svg' not in content_start and '<?xml' not in content_start:
                raise ValueError("Le contenu téléchargé ne semble pas être un SVG valide")
            
            return svg_bytes
            
        except ValueError:
            raise
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Erreur téléchargement SVG: {str(e)}")
    
    else:
        raise ValueError("Aucune source SVG fournie (svg_base64 ou svg_url requis)")


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION DE NETTOYAGE LEGACY (ancien endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_image_legacy(img, cleaning_strength="Medium", denoise_level=0, line_boost_px=0):
    """
    Nettoie l'image : supprime les gris, le bruit, etc.
    VERSION LEGACY - conservée pour compatibilité /clean-kdp-image
    
    Paramètres:
    -----------
    - img: Image PIL
    - cleaning_strength: "Light", "Medium", "Strong", "Extreme"
    - denoise_level: 0-10 (0 = désactivé, 10 = maximum)
    - line_boost_px: 0-10 pixels d'épaississement (0 = désactivé)
    
    Retourne:
    ---------
    - Image PIL nettoyée (mode L, niveaux de gris)
    """
    
    # Convertir en numpy array
    img_array = np.array(img.convert('RGB'))
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # ÉTAPE 1 : DÉBRUITAGE AVANCÉ
    if denoise_level > 0:
        h = 3 + (denoise_level * 2.7)  # 3 à 30
        gray = cv2.fastNlMeansDenoising(
            gray, 
            None, 
            h=h, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
    
    # ÉTAPE 2 : BINARISATION
    thresholds = {
        "Light": 200,
        "Medium": 180,
        "Strong": 160,
        "Extreme": 140
    }
    threshold = thresholds.get(cleaning_strength, 180)
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # ÉTAPE 3 : NETTOYAGE MORPHOLOGIQUE
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # ÉTAPE 4 : SUPPRESSION DES ARTEFACTS ISOLÉS
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(cleaned), connectivity=8
    )
    
    min_size = 50
    mask = np.ones(cleaned.shape, dtype=np.uint8) * 255
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 0
    
    cleaned = mask
    
    # ÉTAPE 5 : ÉPAISSISSEMENT DES TRAITS
    if line_boost_px > 0:
        line_boost_px = max(1, min(10, line_boost_px))
        inverted = cv2.bitwise_not(cleaned)
        kernel_size = line_boost_px * 2 + 1
        kernel_boost = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(inverted, kernel_boost, iterations=1)
        cleaned = cv2.bitwise_not(dilated)
    
    return Image.fromarray(cleaned)


def resize_for_kdp(img, format_kdp, margins_mm=5):
    """Redimensionne l'image pour le format KDP avec marges"""
    
    if format_kdp not in KDP_FORMATS:
        format_kdp = "8.5x11"
    
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
        new_width = usable_width
        new_height = int(usable_width / img_ratio)
    else:
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "healthy", 
        "service": "KDP Image Cleaner",
        "version": "3.0.0",
        "endpoints": [
            "/clean-kdp-image (legacy)",
            "/clean-kdp-image-pro (NEW - 10 étapes)",
            "/svg_to_png",
            "/formats"
        ]
    })


@app.route('/clean-kdp-image', methods=['POST'])
def clean_kdp_image():
    """
    Endpoint LEGACY pour nettoyer les images KDP
    Conservé pour compatibilité avec les scénarios Make.com existants.
    
    Pour le nouveau pipeline pro, utilisez /clean-kdp-image-pro
    
    Body JSON attendu:
    {
        "input_base64": "iVBORw0KGgo...",  // PRIORITAIRE
        "input_url": "https://...",         // Fallback si pas de base64
        "format": "8.5x11",
        "cleaning": "Medium",
        "margins_mm": 5,
        "denoise_level": 2,
        "line_boost_px": 0
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        format_kdp = data.get('format', '8.5x11')
        cleaning_strength = data.get('cleaning', 'Medium')
        margins_mm = data.get('margins_mm', 5)
        denoise_level = int(data.get('denoise_level', 0))
        line_boost_px = int(data.get('line_boost_px', 0))
        
        # Borner les valeurs
        denoise_level = max(0, min(10, denoise_level))
        line_boost_px = max(0, min(10, line_boost_px))
        
        if not data.get('input_base64') and not data.get('input_url'):
            return jsonify({"error": "input_base64 ou input_url requis"}), 400
        
        # 1. Récupérer l'image
        img = get_image_from_request(data)
        
        # 2. Nettoyer l'image (legacy)
        cleaned_img = clean_image_legacy(
            img, 
            cleaning_strength,
            denoise_level=denoise_level,
            line_boost_px=line_boost_px
        )
        
        # 3. Redimensionner pour KDP
        final_img = resize_for_kdp(cleaned_img, format_kdp, margins_mm)
        
        # 4. Préparer la réponse
        img_buffer = BytesIO()
        final_img.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False
        )
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/clean-kdp-image-pro', methods=['POST'])
def clean_kdp_image_pro():
    """
    NOUVEAU Endpoint professionnel pour nettoyer les images KDP
    Pipeline 10 étapes basé sur les standards de l'industrie prepress.
    
    Résout:
    - Bordures noires parasites (auto-crop + border cleanup)
    - Lignes verticales Midjourney (line removal)
    - Gris résiduels (binarisation adaptative)
    
    Body JSON attendu:
    {
        "input_base64": "iVBORw0KGgo...",  // PRIORITAIRE
        "input_url": "https://...",         // Fallback
        "format": "8.5x11",
        "margins_mm": 5,
        "return_metrics": true,             // Retourne les métriques JSON
        "config": {                         // Configuration optionnelle
            "deskew_enabled": true,
            "line_removal_enabled": true,
            "line_boost_px": 2,
            "binarize_method": "adaptive"   // "otsu", "adaptive", "sauvola"
        }
    }
    
    Réponse:
    - Si return_metrics=false (défaut): PNG binaire
    - Si return_metrics=true: JSON avec image base64 + métriques
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        format_kdp = data.get('format', '8.5x11')
        margins_mm = data.get('margins_mm', 5)
        return_metrics = data.get('return_metrics', False)
        user_config = data.get('config', {})
        
        if not data.get('input_base64') and not data.get('input_url'):
            return jsonify({"error": "input_base64 ou input_url requis"}), 400
        
        # 1. Récupérer l'image
        pil_img = get_image_from_request(data)
        
        # 2. Convertir en numpy grayscale
        img_array = np.array(pil_img.convert('L'))
        
        # 3. Créer le processeur avec config custom
        config = ObsidianProCleanup.default_config()
        config.update(user_config)
        processor = ObsidianProCleanup(config)
        
        # 4. Exécuter le pipeline pro
        cleaned_array, metrics = processor.process(img_array)
        
        # 5. Convertir en PIL
        cleaned_pil = Image.fromarray(cleaned_array, mode='L')
        
        # 6. Redimensionner pour KDP
        final_img = resize_for_kdp(cleaned_pil, format_kdp, margins_mm)
        
        # 7. Réponse
        if return_metrics:
            # Mode JSON avec métriques
            img_buffer = BytesIO()
            final_img.save(img_buffer, format='PNG', optimize=True)
            img_buffer.seek(0)
            img_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            return jsonify({
                "success": True,
                "image_base64": img_b64,
                "metrics": metrics,
                "format": format_kdp,
                "pipeline": "ObsidianProCleanup v3.0"
            })
        else:
            # Mode binaire (compatibilité Make.com)
            img_buffer = BytesIO()
            final_img.save(img_buffer, format='PNG', optimize=True)
            img_buffer.seek(0)
            
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=False
            )
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in clean-kdp-image-pro: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/formats', methods=['GET'])
def list_formats():
    """Liste les formats KDP disponibles et les options"""
    return jsonify({
        "formats": list(KDP_FORMATS.keys()),
        "endpoints": {
            "/clean-kdp-image": {
                "description": "Legacy cleaner (simple)",
                "cleaning_levels": ["Light", "Medium", "Strong", "Extreme"],
                "options": {
                    "denoise_level": "0-10",
                    "line_boost_px": "0-10"
                }
            },
            "/clean-kdp-image-pro": {
                "description": "Professional 10-step pipeline (RECOMMENDED)",
                "features": [
                    "Auto-crop (supprime bordures parasites)",
                    "Deskew (correction inclinaison)",
                    "Denoising (fastNlMeans)",
                    "Adaptive binarization (Sauvola/Otsu)",
                    "Line removal (lignes parasites MJ)",
                    "Morphological cleanup",
                    "Despeckle (micro-artefacts)",
                    "Line boost (épaississement)",
                    "Border cleanup (cadres noirs)",
                    "Validation (corners, ratio)"
                ],
                "config_options": {
                    "deskew_enabled": "true/false",
                    "line_removal_enabled": "true/false",
                    "line_boost_px": "0-10",
                    "binarize_method": "otsu/adaptive/sauvola"
                }
            }
        },
        "input_methods": {
            "input_base64": "Image encodée en base64 (RECOMMANDÉ)",
            "input_url": "URL directe de l'image (fallback)"
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS SVG TO PNG
# ═══════════════════════════════════════════════════════════════════════════════

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
    return max(72, min(dpi, 1200))


def _normalize_margins_mm(margins_mm):
    try:
        margins_mm = float(margins_mm)
    except Exception:
        margins_mm = 0.0
    return max(0.0, min(margins_mm, 25.0))


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
    Binarisation agressive pour supprimer l'anti-aliasing (gris) issu du rendu SVG.
    0 = noir, 255 = blanc.
    """
    arr = np.array(img_l)
    arr = np.where(arr < threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@app.route('/svg_to_png', methods=['POST'])
def svg_to_png():
    """
    Convertit un SVG en PNG binaire, KDP-ready.
    
    Body JSON attendu:
    {
      "svg_base64": "PHN2ZyB4bWxucz0i...",  // PRIORITAIRE
      "svg_url": "https://...",              // Fallback
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

        format_kdp = data.get("format", "8.5x11")
        dpi = _normalize_dpi(data.get("dpi", 300))
        margins_mm = _normalize_margins_mm(data.get("margins_mm", 0))
        binarize = bool(data.get("binarize", True))
        threshold = int(data.get("threshold", 245))

        if not data.get('svg_base64') and not data.get('svg_url'):
            return jsonify({"error": "svg_base64 ou svg_url requis"}), 400

        # 1) Récupérer le SVG
        svg_bytes = get_svg_from_request(data)

        # 2) render svg -> PIL (RGBA)
        rendered = svg_bytes_to_pil(svg_bytes, dpi=dpi)

        # 3) place on KDP canvas
        page_l = fit_on_kdp_canvas(rendered, format_kdp=format_kdp, dpi=dpi, margins_mm=margins_mm)

        # 4) strict binarize
        if binarize:
            page_l = binarize_strict(page_l, threshold=threshold)

        # 5) return PNG binary
        buf = BytesIO()
        page_l.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=False)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download SVG: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
