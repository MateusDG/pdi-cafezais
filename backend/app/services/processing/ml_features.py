import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import measure
import scipy.stats as stats


class FeatureExtractor:
    """
    Extração de características para classificação de ervas daninhas usando
    algoritmos de Machine Learning clássico.
    
    Implementa características de:
    - Cor (RGB, HSV, ExG, ExR)
    - Textura (GLCM, LBP) 
    - Forma e Geometria (momentos, circularidade, etc.)
    """
    
    def __init__(self):
        # Parâmetros para GLCM
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]
        
        # Parâmetros para LBP
        self.lbp_radius = 3
        self.lbp_n_points = 24
        
    def extract_region_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extrai características de uma região específica da imagem.
        
        Args:
            img: Imagem RGB original
            mask: Máscara binária da região
            
        Returns:
            Dicionário com características extraídas
        """
        features = {}
        
        # 1. Características de cor
        color_features = self.extract_color_features(img, mask)
        features.update(color_features)
        
        # 2. Características de textura
        texture_features = self.extract_texture_features(img, mask)
        features.update(texture_features)
        
        # 3. Características de forma
        shape_features = self.extract_shape_features(mask)
        features.update(shape_features)
        
        return features
    
    def extract_color_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extrai características baseadas em cor.
        
        Inclui:
        - Estatísticas RGB e HSV (média, desvio padrão)
        - Índices de vegetação (ExG, ExR, ExGR)
        - Histogramas de cor
        """
        features = {}
        
        # Aplicar máscara
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        valid_pixels = masked_img[mask > 0]
        
        if len(valid_pixels) == 0:
            return {f'color_{key}': 0.0 for key in self._get_color_feature_names()}
        
        # Estatísticas RGB
        rgb_mean = np.mean(valid_pixels, axis=0)
        rgb_std = np.std(valid_pixels, axis=0)
        
        features.update({
            'color_r_mean': float(rgb_mean[0]),
            'color_g_mean': float(rgb_mean[1]),
            'color_b_mean': float(rgb_mean[2]),
            'color_r_std': float(rgb_std[0]),
            'color_g_std': float(rgb_std[1]),
            'color_b_std': float(rgb_std[2])
        })
        
        # Converter para HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        valid_hsv = masked_hsv[mask > 0]
        
        if len(valid_hsv) > 0:
            hsv_mean = np.mean(valid_hsv, axis=0)
            hsv_std = np.std(valid_hsv, axis=0)
            
            features.update({
                'color_h_mean': float(hsv_mean[0]),
                'color_s_mean': float(hsv_mean[1]),
                'color_v_mean': float(hsv_mean[2]),
                'color_h_std': float(hsv_std[0]),
                'color_s_std': float(hsv_std[1]),
                'color_v_std': float(hsv_std[2])
            })
        
        # Índices de vegetação
        vegetation_indices = self._calculate_vegetation_indices(valid_pixels)
        features.update(vegetation_indices)
        
        # Momentos de cor
        color_moments = self._calculate_color_moments(valid_pixels)
        features.update(color_moments)
        
        return features
    
    def extract_texture_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extrai características de textura usando GLCM e LBP.
        """
        features = {}
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Encontrar região válida para análise
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return {f'texture_{key}': 0.0 for key in self._get_texture_feature_names()}
        
        # Extrair ROI
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        roi_gray = masked_gray[y_min:y_max+1, x_min:x_max+1]
        roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        if roi_gray.size == 0:
            return {f'texture_{key}': 0.0 for key in self._get_texture_feature_names()}
        
        # Características GLCM
        glcm_features = self._extract_glcm_features(roi_gray, roi_mask)
        features.update(glcm_features)
        
        # Características LBP
        lbp_features = self._extract_lbp_features(roi_gray, roi_mask)
        features.update(lbp_features)
        
        return features
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extrai características de forma e geometria.
        """
        features = {}
        
        # Encontrar contorno principal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {f'shape_{key}': 0.0 for key in self._get_shape_feature_names()}
        
        # Usar maior contorno
        main_contour = max(contours, key=cv2.contourArea)
        
        # Características básicas
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        features['shape_area'] = float(area)
        features['shape_perimeter'] = float(perimeter)
        
        if perimeter > 0:
            features['shape_circularity'] = float(4 * np.pi * area / (perimeter ** 2))
        else:
            features['shape_circularity'] = 0.0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        features['shape_width'] = float(w)
        features['shape_height'] = float(h)
        features['shape_aspect_ratio'] = float(w / h if h > 0 else 0)
        features['shape_extent'] = float(area / (w * h) if w * h > 0 else 0)
        
        # Elipse ajustada
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            features['shape_eccentricity'] = float(np.sqrt(1 - (minor_axis/major_axis)**2) 
                                                  if major_axis > 0 else 0)
        else:
            features['shape_eccentricity'] = 0.0
        
        # Convexidade
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        features['shape_solidity'] = float(area / hull_area if hull_area > 0 else 0)
        
        # Momentos de Hu (invariantes geométricos)
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        for i, hu in enumerate(hu_moments):
            # Log transform para estabilizar
            if hu != 0:
                features[f'shape_hu_{i+1}'] = float(-np.sign(hu) * np.log10(abs(hu)))
            else:
                features[f'shape_hu_{i+1}'] = 0.0
        
        return features
    
    def _calculate_vegetation_indices(self, pixels: np.ndarray) -> Dict[str, float]:
        """Calcula índices de vegetação ExG, ExR, ExGR."""
        if len(pixels) == 0:
            return {'color_exg': 0.0, 'color_exr': 0.0, 'color_exgr': 0.0}
        
        # Normalizar canais
        r = pixels[:, 0].astype(np.float32)
        g = pixels[:, 1].astype(np.float32)
        b = pixels[:, 2].astype(np.float32)
        
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        
        # Calcular índices
        exg = 2 * g_norm - r_norm - b_norm
        exr = 1.4 * r_norm - g_norm
        exgr = exg - exr
        
        return {
            'color_exg': float(np.mean(exg)),
            'color_exr': float(np.mean(exr)),
            'color_exgr': float(np.mean(exgr))
        }
    
    def _calculate_color_moments(self, pixels: np.ndarray) -> Dict[str, float]:
        """Calcula momentos de cor de primeira e segunda ordem."""
        if len(pixels) == 0:
            return {f'color_{c}_moment_{m}': 0.0 for c in ['r', 'g', 'b'] for m in [1, 2, 3]}
        
        features = {}
        channels = ['r', 'g', 'b']
        
        for i, channel in enumerate(channels):
            channel_data = pixels[:, i].astype(np.float32)
            
            # Momento 1 (média) - já calculado, mas incluído por completude
            features[f'color_{channel}_moment_1'] = float(np.mean(channel_data))
            
            # Momento 2 (variância)
            features[f'color_{channel}_moment_2'] = float(np.var(channel_data))
            
            # Momento 3 (skewness)
            if np.std(channel_data) > 0:
                features[f'color_{channel}_moment_3'] = float(stats.skew(channel_data))
            else:
                features[f'color_{channel}_moment_3'] = 0.0
        
        return features
    
    def _extract_glcm_features(self, gray_roi: np.ndarray, mask_roi: np.ndarray) -> Dict[str, float]:
        """Extrai características da matriz GLCM."""
        features = {}
        
        # Quantizar para reduzir complexidade computacional
        gray_quantized = (gray_roi // 32).astype(np.uint8)  # 8 níveis
        
        try:
            # Calcular GLCM
            glcm = graycomatrix(gray_quantized, 
                             distances=self.glcm_distances, 
                             angles=np.deg2rad(self.glcm_angles),
                             levels=8, 
                             symmetric=True, 
                             normed=True)
            
            # Extrair propriedades
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                # Média sobre todas as distâncias e ângulos
                features[f'texture_glcm_{prop}'] = float(np.mean(prop_values))
                
        except Exception:
            # Em caso de erro, valores padrão
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features[f'texture_glcm_{prop}'] = 0.0
        
        return features
    
    def _extract_lbp_features(self, gray_roi: np.ndarray, mask_roi: np.ndarray) -> Dict[str, float]:
        """Extrai características do Local Binary Pattern."""
        features = {}
        
        try:
            # Calcular LBP
            lbp = local_binary_pattern(gray_roi, 
                                     self.lbp_n_points, 
                                     self.lbp_radius, 
                                     method='uniform')
            
            # Aplicar máscara
            masked_lbp = lbp[mask_roi > 0]
            
            if len(masked_lbp) > 0:
                # Histograma normalizado
                hist, _ = np.histogram(masked_lbp, bins=self.lbp_n_points + 2, 
                                     range=(0, self.lbp_n_points + 2))
                hist_norm = hist.astype(np.float32) / np.sum(hist)
                
                # Estatísticas do histograma
                features['texture_lbp_uniformity'] = float(np.sum(hist_norm ** 2))
                features['texture_lbp_entropy'] = float(-np.sum(hist_norm[hist_norm > 0] * 
                                                               np.log2(hist_norm[hist_norm > 0])))
                
                # Momentos do LBP
                features['texture_lbp_mean'] = float(np.mean(masked_lbp))
                features['texture_lbp_std'] = float(np.std(masked_lbp))
                features['texture_lbp_skewness'] = float(stats.skew(masked_lbp)) if np.std(masked_lbp) > 0 else 0.0
            else:
                features.update({
                    'texture_lbp_uniformity': 0.0,
                    'texture_lbp_entropy': 0.0,
                    'texture_lbp_mean': 0.0,
                    'texture_lbp_std': 0.0,
                    'texture_lbp_skewness': 0.0
                })
                
        except Exception:
            features.update({
                'texture_lbp_uniformity': 0.0,
                'texture_lbp_entropy': 0.0,
                'texture_lbp_mean': 0.0,
                'texture_lbp_std': 0.0,
                'texture_lbp_skewness': 0.0
            })
        
        return features
    
    def _get_color_feature_names(self) -> List[str]:
        """Retorna nomes das características de cor."""
        names = []
        
        # RGB básico
        for channel in ['r', 'g', 'b']:
            names.extend([f'{channel}_mean', f'{channel}_std'])
        
        # HSV básico  
        for channel in ['h', 's', 'v']:
            names.extend([f'{channel}_mean', f'{channel}_std'])
        
        # Índices de vegetação
        names.extend(['exg', 'exr', 'exgr'])
        
        # Momentos de cor
        for channel in ['r', 'g', 'b']:
            for moment in [1, 2, 3]:
                names.append(f'{channel}_moment_{moment}')
        
        return names
    
    def _get_texture_feature_names(self) -> List[str]:
        """Retorna nomes das características de textura."""
        names = []
        
        # GLCM
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            names.append(f'glcm_{prop}')
        
        # LBP
        names.extend(['lbp_uniformity', 'lbp_entropy', 'lbp_mean', 'lbp_std', 'lbp_skewness'])
        
        return names
    
    def _get_shape_feature_names(self) -> List[str]:
        """Retorna nomes das características de forma."""
        names = ['area', 'perimeter', 'circularity', 'width', 'height', 
                'aspect_ratio', 'extent', 'eccentricity', 'solidity']
        
        # Momentos de Hu
        for i in range(1, 8):
            names.append(f'hu_{i}')
        
        return names
    
    def get_all_feature_names(self) -> List[str]:
        """Retorna todos os nomes de características."""
        names = []
        
        # Prefixos para organização
        color_names = [f'color_{name}' for name in self._get_color_feature_names()]
        texture_names = [f'texture_{name}' for name in self._get_texture_feature_names()]
        shape_names = [f'shape_{name}' for name in self._get_shape_feature_names()]
        
        names.extend(color_names)
        names.extend(texture_names)
        names.extend(shape_names)
        
        return names
    
    def extract_patch_features(self, img: np.ndarray, patch_size: Tuple[int, int] = (32, 32)) -> List[Dict[str, float]]:
        """
        Extrai características de patches da imagem para treinamento.
        
        Args:
            img: Imagem RGB
            patch_size: Tamanho dos patches (height, width)
            
        Returns:
            Lista de características de cada patch
        """
        features_list = []
        h, w = img.shape[:2]
        patch_h, patch_w = patch_size
        
        for y in range(0, h - patch_h + 1, patch_h):
            for x in range(0, w - patch_w + 1, patch_w):
                # Extrair patch
                patch = img[y:y+patch_h, x:x+patch_w]
                
                # Criar máscara completa para o patch
                mask = np.ones((patch_h, patch_w), dtype=np.uint8) * 255
                
                # Extrair características
                patch_features = self.extract_region_features(patch, mask)
                
                # Adicionar coordenadas
                patch_features['patch_x'] = float(x)
                patch_features['patch_y'] = float(y)
                
                features_list.append(patch_features)
        
        return features_list