import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import random
from dataclasses import dataclass
from datetime import datetime

from .ml_classifiers import ClassicalMLWeedDetector
from .weed import detect_weeds_robust


@dataclass
class TrainingConfig:
    """Configuração para treinamento de modelos."""
    patch_size: Tuple[int, int] = (64, 64)
    samples_per_class: int = 500
    test_size: float = 0.2
    random_state: int = 42
    models_to_train: List[str] = None
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['svm', 'random_forest', 'knn', 'naive_bayes']


class MLTrainingPipeline:
    """
    Pipeline para treinamento de modelos de detecção de ervas daninhas.
    
    Suporta:
    - Geração de dados sintéticos
    - Extração de patches para treinamento
    - Treinamento de múltiplos modelos
    - Avaliação e comparação de resultados
    """
    
    def __init__(self, data_dir: str = "data/training", models_dir: str = "models/classical_ml"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.detector = ClassicalMLWeedDetector(models_dir)
        
        # Criar diretórios
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
    def generate_synthetic_training_data(self, base_images: List[np.ndarray], 
                                       config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dados sintéticos de treinamento a partir de imagens base.
        
        Args:
            base_images: Lista de imagens RGB para usar como base
            config: Configuração do treinamento
            
        Returns:
            Tupla (características, labels) para treinamento
        """
        print("Gerando dados sintéticos de treinamento...")
        
        all_samples = []
        
        for i, img in enumerate(base_images):
            print(f"Processando imagem base {i+1}/{len(base_images)}")
            
            # Detectar regiões usando algoritmo tradicional
            detection_result = detect_weeds_robust(img, sensitivity=0.7)
            
            # Extrair patches de diferentes tipos
            weed_samples = self._extract_weed_patches(img, detection_result, config)
            coffee_samples = self._extract_coffee_patches(img, config)
            soil_samples = self._extract_soil_patches(img, config)
            
            all_samples.extend(weed_samples)
            all_samples.extend(coffee_samples) 
            all_samples.extend(soil_samples)
        
        # Balancear classes
        balanced_samples = self._balance_classes(all_samples, config)
        
        # Converter para formato de treinamento
        X, y = self.detector.prepare_training_data(balanced_samples)
        
        return X, y
    
    def _extract_weed_patches(self, img: np.ndarray, detection_result: Dict[str, Any], 
                            config: TrainingConfig) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Extrai patches de ervas daninhas detectadas."""
        weed_samples = []
        contours = detection_result['contours']
        
        patch_h, patch_w = config.patch_size
        height, width = img.shape[:2]
        
        for contour in contours:
            # Criar máscara da região
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Encontrar bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extrair patches da região com overlap
            for patch_y in range(max(0, y-patch_h//2), 
                               min(height-patch_h, y+h), patch_h//2):
                for patch_x in range(max(0, x-patch_w//2), 
                                   min(width-patch_w, x+w), patch_w//2):
                    
                    # Extrair patch
                    patch_img = img[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                    patch_mask = mask[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                    
                    # Verificar se há sobreposição suficiente com a erva
                    overlap_ratio = np.sum(patch_mask > 0) / (patch_h * patch_w)
                    
                    if overlap_ratio > 0.3:  # Pelo menos 30% do patch deve ser erva
                        weed_samples.append((patch_img, patch_mask, 'weed'))
        
        return weed_samples
    
    def _extract_coffee_patches(self, img: np.ndarray, config: TrainingConfig) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Extrai patches de regiões de café (vegetação escura)."""
        coffee_samples = []
        
        # Usar HSV para detectar vegetação mais escura (café)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Parâmetros para café: verde escuro, alta saturação
        lower_coffee = np.array([35, 80, 30])
        upper_coffee = np.array([85, 255, 120])
        
        coffee_mask = cv2.inRange(hsv, lower_coffee, upper_coffee)
        
        # Morfologia para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        coffee_mask = cv2.morphologyEx(coffee_mask, cv2.MORPH_CLOSE, kernel)
        
        # Extrair patches aleatórios de regiões de café
        patch_h, patch_w = config.patch_size
        height, width = img.shape[:2]
        
        # Encontrar pontos válidos para extração
        valid_points = []
        for y in range(0, height-patch_h, patch_h//2):
            for x in range(0, width-patch_w, patch_w//2):
                patch_mask = coffee_mask[y:y+patch_h, x:x+patch_w]
                coffee_ratio = np.sum(patch_mask > 0) / (patch_h * patch_w)
                
                if coffee_ratio > 0.4:  # Pelo menos 40% deve ser café
                    valid_points.append((x, y))
        
        # Amostrar aleatoriamente
        random.shuffle(valid_points)
        for x, y in valid_points[:config.samples_per_class//len(img) if len(img) > 0 else 100]:
            patch_img = img[y:y+patch_h, x:x+patch_w]
            patch_mask = coffee_mask[y:y+patch_h, x:x+patch_w]
            
            coffee_samples.append((patch_img, patch_mask, 'coffee'))
        
        return coffee_samples
    
    def _extract_soil_patches(self, img: np.ndarray, config: TrainingConfig) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Extrai patches de solo."""
        soil_samples = []
        
        # Usar HSV para detectar solo
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Parâmetros para solo: marrom, baixa saturação
        lower_soil1 = np.array([0, 20, 20])
        upper_soil1 = np.array([35, 180, 200])
        
        lower_soil2 = np.array([0, 0, 80])
        upper_soil2 = np.array([30, 60, 255])
        
        soil_mask1 = cv2.inRange(hsv, lower_soil1, upper_soil1)
        soil_mask2 = cv2.inRange(hsv, lower_soil2, upper_soil2)
        soil_mask = cv2.bitwise_or(soil_mask1, soil_mask2)
        
        # Morfologia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
        
        # Extrair patches
        patch_h, patch_w = config.patch_size
        height, width = img.shape[:2]
        
        valid_points = []
        for y in range(0, height-patch_h, patch_h//2):
            for x in range(0, width-patch_w, patch_w//2):
                patch_mask = soil_mask[y:y+patch_h, x:x+patch_w]
                soil_ratio = np.sum(patch_mask > 0) / (patch_h * patch_w)
                
                if soil_ratio > 0.5:  # Pelo menos 50% deve ser solo
                    valid_points.append((x, y))
        
        # Amostrar aleatoriamente
        random.shuffle(valid_points)
        for x, y in valid_points[:config.samples_per_class//len(img) if len(img) > 0 else 100]:
            patch_img = img[y:y+patch_h, x:x+patch_w]
            patch_mask = soil_mask[y:y+patch_h, x:x+patch_w]
            
            soil_samples.append((patch_img, patch_mask, 'soil'))
        
        return soil_samples
    
    def _balance_classes(self, samples: List[Tuple[np.ndarray, np.ndarray, str]], 
                        config: TrainingConfig) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Balanceia as classes removendo ou duplicando amostras."""
        
        # Separar por classe
        class_samples = {}
        for sample in samples:
            class_name = sample[2]
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append(sample)
        
        print(f"Amostras por classe antes do balanceamento:")
        for class_name, class_list in class_samples.items():
            print(f"  {class_name}: {len(class_list)}")
        
        # Balancear para samples_per_class
        balanced_samples = []
        
        for class_name, class_list in class_samples.items():
            target_count = config.samples_per_class
            current_count = len(class_list)
            
            if current_count >= target_count:
                # Subamostrar
                random.shuffle(class_list)
                selected_samples = class_list[:target_count]
            else:
                # Superamostrar (duplicar com variações)
                selected_samples = class_list.copy()
                while len(selected_samples) < target_count:
                    # Duplicar amostra aleatória com pequenas variações
                    original_sample = random.choice(class_list)
                    augmented_sample = self._augment_sample(original_sample)
                    selected_samples.append(augmented_sample)
                
                selected_samples = selected_samples[:target_count]
            
            balanced_samples.extend(selected_samples)
        
        print(f"Amostras balanceadas: {len(balanced_samples)} total")
        
        # Embaralhar
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _augment_sample(self, sample: Tuple[np.ndarray, np.ndarray, str]) -> Tuple[np.ndarray, np.ndarray, str]:
        """Aplica augmentação simples a uma amostra."""
        img, mask, label = sample
        
        # Rotação aleatória pequena
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        augmented_img = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                      flags=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_REFLECT_101)
        augmented_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                                       flags=cv2.INTER_NEAREST)
        
        # Variação de brilho
        brightness_factor = random.uniform(0.8, 1.2)
        augmented_img = np.clip(augmented_img * brightness_factor, 0, 255).astype(np.uint8)
        
        return (augmented_img, augmented_mask, label)
    
    def train_models_from_images(self, image_paths: List[str], 
                               config: TrainingConfig = None) -> Dict[str, Dict[str, Any]]:
        """
        Treina modelos a partir de uma lista de imagens.
        
        Args:
            image_paths: Caminhos para imagens de treinamento
            config: Configuração do treinamento
            
        Returns:
            Resultados do treinamento de todos os modelos
        """
        if config is None:
            config = TrainingConfig()
        
        # Carregar imagens
        print(f"Carregando {len(image_paths)} imagens...")
        base_images = []
        
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    base_images.append(img_rgb)
                    print(f"Carregada: {path}")
                else:
                    print(f"Erro ao carregar: {path}")
            except Exception as e:
                print(f"Erro ao processar {path}: {e}")
        
        if not base_images:
            raise ValueError("Nenhuma imagem válida foi carregada")
        
        # Gerar dados de treinamento
        X, y = self.generate_synthetic_training_data(base_images, config)
        
        # Treinar modelos
        print("Iniciando treinamento dos modelos...")
        results = self.detector.train_all_models(X, y, 
                                                test_size=config.test_size, 
                                                random_state=config.random_state)
        
        # Salvar modelos
        self.detector.save_models()
        
        # Salvar resultados
        self._save_training_results(results, config)
        
        return results
    
    def _save_training_results(self, results: Dict[str, Dict[str, Any]], config: TrainingConfig):
        """Salva resultados do treinamento."""
        
        # Preparar dados para JSON (converter numpy arrays)
        json_results = {}
        
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'accuracy': model_results['accuracy'],
                'cv_mean': model_results['cv_mean'],
                'cv_std': model_results['cv_std'],
                'best_params': model_results.get('best_params', {}),
                'classification_report': model_results['classification_report']
            }
            
            # Feature importance para Random Forest
            if 'feature_importance' in model_results:
                # Pegar top 10 características mais importantes
                importance_dict = model_results['feature_importance']
                sorted_features = sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                json_results[model_name]['top_features'] = dict(sorted_features)
        
        # Adicionar metadados
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'patch_size': config.patch_size,
                'samples_per_class': config.samples_per_class,
                'test_size': config.test_size,
                'random_state': config.random_state,
                'models_trained': config.models_to_train
            },
            'results': json_results
        }
        
        # Salvar arquivo
        results_path = os.path.join(self.models_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados salvos em: {results_path}")
    
    def evaluate_models(self, test_images: List[str]) -> Dict[str, Any]:
        """
        Avalia modelos treinados em imagens de teste.
        
        Args:
            test_images: Lista de caminhos para imagens de teste
            
        Returns:
            Resultados da avaliação
        """
        # Carregar modelos se não estiverem carregados
        self.detector.load_models()
        
        results = {
            'svm': [],
            'random_forest': [], 
            'knn': [],
            'naive_bayes': []
        }
        
        for img_path in test_images:
            try:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                print(f"Avaliando: {img_path}")
                
                # Testar cada modelo
                for model_name in results.keys():
                    if self.detector.models[model_name] is not None:
                        detection_result = self.detector.detect_weeds_ml(img_rgb, model_name)
                        
                        results[model_name].append({
                            'image': img_path,
                            'weed_count': detection_result['weed_count'],
                            'weed_percentage': detection_result['weed_percentage'],
                            'avg_confidence': detection_result['avg_confidence']
                        })
                
            except Exception as e:
                print(f"Erro ao avaliar {img_path}: {e}")
        
        # Calcular estatísticas médias
        summary = {}
        for model_name, model_results in results.items():
            if model_results:
                summary[model_name] = {
                    'avg_weed_count': np.mean([r['weed_count'] for r in model_results]),
                    'avg_weed_percentage': np.mean([r['weed_percentage'] for r in model_results]),
                    'avg_confidence': np.mean([r['avg_confidence'] for r in model_results]),
                    'images_processed': len(model_results)
                }
        
        return {
            'detailed_results': results,
            'summary': summary
        }
    
    def compare_with_traditional(self, test_images: List[str]) -> Dict[str, Any]:
        """
        Compara modelos ML com método tradicional em imagens de teste.
        
        Args:
            test_images: Lista de caminhos para imagens de teste
            
        Returns:
            Comparação dos resultados
        """
        self.detector.load_models()
        
        comparison_results = []
        
        for img_path in test_images:
            try:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                print(f"Comparando métodos em: {img_path}")
                
                # Método tradicional
                traditional_result = detect_weeds_robust(img_rgb)
                
                # Métodos ML
                ml_results = {}
                for model_name in ['svm', 'random_forest', 'knn', 'naive_bayes']:
                    if self.detector.models[model_name] is not None:
                        ml_results[model_name] = self.detector.detect_weeds_ml(img_rgb, model_name)
                
                comparison_results.append({
                    'image': img_path,
                    'traditional': {
                        'weed_count': traditional_result['weed_count'],
                        'weed_percentage': traditional_result['weed_percentage']
                    },
                    'ml_models': {
                        model_name: {
                            'weed_count': result['weed_count'],
                            'weed_percentage': result['weed_percentage'],
                            'avg_confidence': result['avg_confidence']
                        }
                        for model_name, result in ml_results.items()
                    }
                })
                
            except Exception as e:
                print(f"Erro na comparação {img_path}: {e}")
        
        return comparison_results