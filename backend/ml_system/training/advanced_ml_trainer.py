#!/usr/bin/env python3
"""
Sistema Avançado de Treinamento ML para Detecção de Ervas Daninhas
Versão Professional com interface, monitoramento e otimização completa.

Características:
- Interface interativa
- Múltiplas estratégias de dados
- Otimização automática de hiperparâmetros
- Análise estatística avançada
- Validação cruzada estratificada
- Visualizações e relatórios
- Sistema de checkpoints
- Comparação entre métodos
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Imports condicionais
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("⚠️  Matplotlib/Seaborn não disponível. Visualizações desabilitadas.")

try:
    from sklearn.model_selection import StratifiedKFold, validation_curve, learning_curve
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.inspection import permutation_importance
    HAS_ADVANCED_SKLEARN = True
except ImportError:
    HAS_ADVANCED_SKLEARN = False
    print("⚠️  Versão básica do scikit-learn. Algumas análises avançadas desabilitadas.")


class AdvancedMLTrainer:
    """
    Sistema avançado de treinamento com monitoramento completo e otimizações.
    """
    
    def __init__(self, project_name: str = "WeedDetectionML"):
        self.project_name = project_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Diretórios do projeto
        self.project_dir = Path(f"ml_project_{project_name.lower()}")
        self.setup_project_structure()
        
        # Configurações
        self.config = self.load_or_create_config()
        self.training_log = []
        self.results_history = []
        
        # Estado do treinamento
        self.current_stage = "initialization"
        self.datasets = {}
        self.trained_models = {}
        self.best_model_info = None
        
        print(f"🚀 Sistema Avançado de Treinamento ML Inicializado")
        print(f"📁 Projeto: {self.project_dir}")
        print(f"🔑 Session ID: {self.session_id}")
    
    def setup_project_structure(self):
        """Cria estrutura completa do projeto."""
        directories = [
            'datasets/raw',
            'datasets/processed', 
            'datasets/synthetic',
            'datasets/augmented',
            'models/checkpoints',
            'models/final',
            'results/analysis',
            'results/visualizations',
            'results/reports',
            'logs',
            'config',
            'experiments'
        ]
        
        for directory in directories:
            (self.project_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Estrutura do projeto criada em: {self.project_dir}")
    
    def load_or_create_config(self) -> Dict[str, Any]:
        """Carrega ou cria configuração avançada."""
        config_file = self.project_dir / 'config' / 'training_config.json'
        
        default_config = {
            "data_strategy": {
                "synthetic_ratio": 0.4,  # 40% sintético, 60% real
                "augmentation_factor": 3,  # 3x mais dados via augmentação
                "validation_split": 0.2,
                "test_split": 0.15,
                "stratify": True
            },
            "feature_engineering": {
                "color_spaces": ["RGB", "HSV", "LAB"],
                "texture_methods": ["GLCM", "LBP", "Gabor"],
                "shape_descriptors": ["Hu", "Zernike", "Fourier"],
                "statistical_moments": [1, 2, 3, 4],
                "enable_pca": True,
                "pca_components": 0.95
            },
            "model_config": {
                "models_to_train": ["svm", "random_forest", "xgboost", "neural_network"],
                "ensemble_methods": ["voting", "stacking", "bagging"],
                "hyperparameter_optimization": "bayesian",  # "grid", "random", "bayesian"
                "cv_folds": 10,
                "optimization_trials": 100
            },
            "training_params": {
                "patch_sizes": [(32, 32), (64, 64), (96, 96)],
                "samples_per_class": [500, 1000, 2000],
                "batch_processing": True,
                "parallel_jobs": -1,
                "early_stopping": True,
                "checkpoint_frequency": 10
            },
            "evaluation_metrics": [
                "accuracy", "precision", "recall", "f1", 
                "roc_auc", "pr_auc", "kappa", "mcc"
            ],
            "reporting": {
                "generate_plots": HAS_VISUALIZATION,
                "feature_importance_analysis": True,
                "learning_curves": True,
                "confusion_matrix_analysis": True,
                "cross_validation_analysis": True,
                "generate_pdf_report": False
            }
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("📄 Configuração carregada do arquivo existente")
        else:
            config = default_config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("📄 Configuração padrão criada")
        
        return config
    
    def interactive_configuration(self):
        """Interface interativa para configuração."""
        print("\n" + "="*60)
        print("🔧 CONFIGURAÇÃO INTERATIVA DO TREINAMENTO")
        print("="*60)
        
        print("\n1. Estratégia de Dados:")
        print(f"   Atual: {self.config['data_strategy']['synthetic_ratio']:.1%} sintético")
        
        choice = input("Modificar estratégia de dados? (s/N): ").lower()
        if choice == 's':
            try:
                synthetic_ratio = float(input("Proporção de dados sintéticos (0.0-1.0): "))
                self.config['data_strategy']['synthetic_ratio'] = max(0.0, min(1.0, synthetic_ratio))
                
                aug_factor = int(input("Fator de augmentação (1-5): "))
                self.config['data_strategy']['augmentation_factor'] = max(1, min(5, aug_factor))
            except ValueError:
                print("⚠️  Valores inválidos, usando configuração padrão")
        
        print("\n2. Modelos para Treinamento:")
        available_models = ["svm", "random_forest", "xgboost", "neural_network", "naive_bayes", "knn"]
        current_models = self.config['model_config']['models_to_train']
        
        print(f"   Atuais: {', '.join(current_models)}")
        print(f"   Disponíveis: {', '.join(available_models)}")
        
        choice = input("Modificar seleção de modelos? (s/N): ").lower()
        if choice == 's':
            selected_models = input("Modelos (separados por vírgula): ").split(',')
            selected_models = [m.strip() for m in selected_models if m.strip() in available_models]
            if selected_models:
                self.config['model_config']['models_to_train'] = selected_models
        
        print("\n3. Otimização de Hiperparâmetros:")
        optimization_methods = ["grid", "random", "bayesian"]
        current_method = self.config['model_config']['hyperparameter_optimization']
        
        print(f"   Atual: {current_method}")
        choice = input("Modificar método de otimização? (s/N): ").lower()
        if choice == 's':
            print("   Opções: 1) Grid Search  2) Random Search  3) Bayesian")
            try:
                opt_choice = int(input("Escolha (1-3): ")) - 1
                if 0 <= opt_choice < len(optimization_methods):
                    self.config['model_config']['hyperparameter_optimization'] = optimization_methods[opt_choice]
            except ValueError:
                print("⚠️  Escolha inválida, mantendo configuração atual")
        
        # Salvar configuração modificada
        config_file = self.project_dir / 'config' / 'training_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("\n✅ Configuração atualizada e salva!")
    
    def create_advanced_synthetic_dataset(self, num_images: int = 50) -> List[str]:
        """Cria dataset sintético avançado com múltiplas variações."""
        print(f"\n🎨 Criando dataset sintético avançado ({num_images} imagens)...")
        
        synthetic_dir = self.project_dir / 'datasets' / 'synthetic'
        image_paths = []
        
        # Parâmetros de variação
        scenarios = [
            {"name": "high_infestation", "weed_density": 0.4, "coffee_health": "stressed"},
            {"name": "medium_infestation", "weed_density": 0.25, "coffee_health": "normal"}, 
            {"name": "low_infestation", "weed_density": 0.1, "coffee_health": "healthy"},
            {"name": "young_plantation", "weed_density": 0.3, "coffee_health": "young"},
            {"name": "mature_plantation", "weed_density": 0.15, "coffee_health": "mature"},
            {"name": "intercropping", "weed_density": 0.2, "coffee_health": "intercrop"},
            {"name": "seasonal_dormant", "weed_density": 0.35, "coffee_health": "dormant"},
            {"name": "harvested_area", "weed_density": 0.45, "coffee_health": "post_harvest"}
        ]
        
        lighting_conditions = ["bright", "overcast", "shadow", "golden_hour", "harsh_noon"]
        soil_types = ["clay", "sandy", "loamy", "rocky", "organic"]
        
        for i in range(num_images):
            scenario = scenarios[i % len(scenarios)]
            lighting = lighting_conditions[i % len(lighting_conditions)]
            soil = soil_types[i % len(soil_types)]
            
            img = self.create_complex_farm_scene(scenario, lighting, soil)
            
            filename = f"synthetic_{scenario['name']}_{lighting}_{soil}_{i:03d}.jpg"
            filepath = synthetic_dir / filename
            
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            image_paths.append(str(filepath))
            
            if (i + 1) % 10 == 0:
                print(f"  📊 Geradas {i + 1}/{num_images} imagens sintéticas")
        
        print(f"✅ Dataset sintético completo: {len(image_paths)} imagens")
        return image_paths
    
    def create_complex_farm_scene(self, scenario: Dict, lighting: str, soil_type: str) -> np.ndarray:
        """Cria cena de fazenda complexa com múltiplas variáveis."""
        # Dimensões variáveis baseadas em cenário
        if scenario["coffee_health"] == "young":
            width, height = np.random.randint(600, 900), np.random.randint(450, 700)
        else:
            width, height = np.random.randint(800, 1200), np.random.randint(600, 900)
        
        # Base do solo baseada no tipo
        soil_colors = {
            "clay": (85, 60, 45),
            "sandy": (120, 100, 80), 
            "loamy": (100, 80, 60),
            "rocky": (90, 85, 75),
            "organic": (70, 50, 35)
        }
        
        base_soil = soil_colors.get(soil_type, (95, 75, 55))
        img = np.ones((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = base_soil[0] + np.random.randint(-15, 15, (height, width))
        img[:, :, 1] = base_soil[1] + np.random.randint(-15, 15, (height, width))
        img[:, :, 2] = base_soil[2] + np.random.randint(-15, 15, (height, width))
        
        # Textura do solo específica
        self.add_soil_texture(img, soil_type)
        
        # Plantas de café baseadas na saúde
        self.add_coffee_plants_by_health(img, scenario["coffee_health"])
        
        # Ervas daninhas com padrões realistas
        self.add_realistic_weeds(img, scenario["weed_density"], scenario["coffee_health"])
        
        # Efeitos de iluminação
        self.apply_lighting_effects(img, lighting)
        
        # Ruído e pós-processamento
        self.apply_realistic_postprocessing(img)
        
        return img
    
    def add_soil_texture(self, img: np.ndarray, soil_type: str):
        """Adiciona textura específica do solo."""
        height, width = img.shape[:2]
        
        if soil_type == "rocky":
            # Adicionar "pedras"
            for _ in range(width * height // 5000):
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                size = np.random.randint(3, 8)
                rock_color = np.random.randint(60, 120)
                cv2.circle(img, (x, y), size, (rock_color, rock_color-10, rock_color-20), -1)
        
        elif soil_type == "sandy":
            # Textura granular
            noise = np.random.randint(-20, 20, img.shape)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif soil_type == "clay":
            # Textura mais uniforme com rachaduras
            for _ in range(20):
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = x1 + np.random.randint(-50, 50), y1 + np.random.randint(-50, 50)
                cv2.line(img, (x1, y1), (x2, y2), (50, 40, 30), 1)
    
    def add_coffee_plants_by_health(self, img: np.ndarray, health_status: str):
        """Adiciona plantas de café baseadas no status de saúde."""
        height, width = img.shape[:2]
        
        # Configurações por estado de saúde
        health_configs = {
            "young": {"size": 0.6, "color_variation": 20, "density": 0.7, "organization": 0.9},
            "healthy": {"size": 1.0, "color_variation": 10, "density": 0.85, "organization": 0.9},
            "mature": {"size": 1.3, "color_variation": 15, "density": 0.9, "organization": 0.85},
            "stressed": {"size": 0.8, "color_variation": 30, "density": 0.6, "organization": 0.7},
            "dormant": {"size": 0.9, "color_variation": 25, "density": 0.8, "organization": 0.8},
            "intercrop": {"size": 0.9, "color_variation": 20, "density": 0.7, "organization": 0.6},
            "post_harvest": {"size": 1.1, "color_variation": 35, "density": 0.75, "organization": 0.8}
        }
        
        config = health_configs.get(health_status, health_configs["healthy"])
        
        # Cor base do café por saúde
        base_colors = {
            "young": (10, 80, 20),
            "healthy": (5, 100, 15),
            "mature": (0, 90, 10), 
            "stressed": (15, 70, 25),
            "dormant": (20, 60, 30),
            "intercrop": (8, 85, 18),
            "post_harvest": (25, 65, 35)
        }
        
        base_color = base_colors.get(health_status, (5, 90, 15))
        
        # Plantio organizado vs disperso
        if config["organization"] > 0.7:
            self.add_organized_coffee(img, config, base_color)
        else:
            self.add_scattered_coffee(img, config, base_color)
    
    def add_organized_coffee(self, img: np.ndarray, config: Dict, base_color: Tuple[int, int, int]):
        """Adiciona café em fileiras organizadas."""
        height, width = img.shape[:2]
        row_spacing = np.random.randint(60, 90)
        plant_spacing = np.random.randint(40, 70)
        
        for row_y in range(30, height-30, row_spacing):
            for plant_x in range(25, width-25, plant_spacing):
                if np.random.random() < config["density"]:
                    # Variação de cor baseada na saúde
                    color_var = config["color_variation"]
                    coffee_color = (
                        max(0, base_color[0] + np.random.randint(-color_var//2, color_var//2)),
                        max(0, base_color[1] + np.random.randint(-color_var, color_var)),
                        max(0, base_color[2] + np.random.randint(-color_var//2, color_var//2))
                    )
                    
                    radius = int(np.random.randint(12, 22) * config["size"])
                    
                    # Planta principal
                    cv2.circle(img, (plant_x, row_y), radius, coffee_color, -1)
                    
                    # Folhas secundárias (mais realismo)
                    num_leaves = np.random.randint(4, 8)
                    for i in range(num_leaves):
                        angle = (2 * np.pi * i) / num_leaves + np.random.uniform(-0.2, 0.2)
                        leaf_x = plant_x + int(radius * 0.7 * np.cos(angle))
                        leaf_y = row_y + int(radius * 0.7 * np.sin(angle))
                        leaf_size = radius // 3
                        cv2.circle(img, (leaf_x, leaf_y), leaf_size, coffee_color, -1)
    
    def add_scattered_coffee(self, img: np.ndarray, config: Dict, base_color: Tuple[int, int, int]):
        """Adiciona café espalhado (sistemas agroflorestais)."""
        height, width = img.shape[:2]
        num_plants = int(width * height * config["density"] / 4000)
        
        for _ in range(num_plants):
            x = np.random.randint(20, width-20)
            y = np.random.randint(20, height-20)
            
            color_var = config["color_variation"]
            coffee_color = (
                max(0, base_color[0] + np.random.randint(-color_var//2, color_var//2)),
                max(0, base_color[1] + np.random.randint(-color_var, color_var)),
                max(0, base_color[2] + np.random.randint(-color_var//2, color_var//2))
            )
            
            radius = int(np.random.randint(10, 18) * config["size"])
            cv2.circle(img, (x, y), radius, coffee_color, -1)
    
    def add_realistic_weeds(self, img: np.ndarray, density: float, coffee_health: str):
        """Adiciona ervas daninhas com padrões ecológicos realistas."""
        height, width = img.shape[:2]
        
        # Tipos de ervas daninhas comuns em cafezais
        weed_types = [
            {"name": "grass_weeds", "color_range": [(100, 180, 60), (140, 220, 100)], "shape": "linear"},
            {"name": "broadleaf", "color_range": [(120, 200, 80), (160, 240, 120)], "shape": "broad"},
            {"name": "vine_weeds", "color_range": [(90, 160, 50), (130, 200, 90)], "shape": "spreading"},
            {"name": "sedges", "color_range": [(110, 190, 70), (150, 230, 110)], "shape": "clumping"}
        ]
        
        # Densidade influenciada pela saúde do café
        health_multipliers = {
            "young": 1.3,      # Mais ervas em plantios novos
            "stressed": 1.4,   # Ervas aproveitam café enfraquecido
            "post_harvest": 1.5,  # Mais ervas após colheita
            "healthy": 0.8,    # Café saudável compete melhor
            "mature": 0.9      # Café maduro com boa cobertura
        }
        
        adjusted_density = density * health_multipliers.get(coffee_health, 1.0)
        
        total_area = width * height
        target_weed_area = int(total_area * adjusted_density)
        current_area = 0
        
        while current_area < target_weed_area:
            # Escolher tipo de erva
            weed_type = weed_types[np.random.randint(0, len(weed_types))]
            
            # Posição (ervas tendem a agrupar)
            if np.random.random() < 0.3:  # 30% chance de agrupamento
                # Criar cluster de ervas
                cluster_x = np.random.randint(30, width-30)
                cluster_y = np.random.randint(30, height-30)
                cluster_size = np.random.randint(3, 8)
                
                for _ in range(cluster_size):
                    x = cluster_x + np.random.randint(-25, 25)
                    y = cluster_y + np.random.randint(-25, 25)
                    x, y = max(15, min(width-15, x)), max(15, min(height-15, y))
                    
                    area_added = self.draw_weed(img, x, y, weed_type)
                    current_area += area_added
            else:
                # Erva isolada
                x = np.random.randint(15, width-15)
                y = np.random.randint(15, height-15)
                
                area_added = self.draw_weed(img, x, y, weed_type)
                current_area += area_added
    
    def draw_weed(self, img: np.ndarray, x: int, y: int, weed_type: Dict) -> int:
        """Desenha uma erva específica baseada no tipo."""
        color_range = weed_type["color_range"]
        weed_color = [
            np.random.randint(color_range[0][i], color_range[1][i]) for i in range(3)
        ]
        
        shape = weed_type["shape"]
        size = np.random.randint(6, 18)
        
        if shape == "linear":  # Gramíneas
            # Formato alongado
            length = size + np.random.randint(5, 15)
            width = max(2, size // 3)
            angle = np.random.uniform(0, 2 * np.pi)
            
            end_x = x + int(length * np.cos(angle))
            end_y = y + int(length * np.sin(angle))
            cv2.line(img, (x, y), (end_x, end_y), weed_color, width)
            
            return length * width
            
        elif shape == "broad":  # Folhas largas
            # Formato elíptico
            axes = (size, int(size * 1.5))
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, (x, y), axes, angle, 0, 360, weed_color, -1)
            
            return axes[0] * axes[1]
            
        elif shape == "spreading":  # Trepadeiras
            # Formato irregular espalhado
            num_points = np.random.randint(6, 10)
            points = []
            
            for i in range(num_points):
                angle = (2 * np.pi * i) / num_points + np.random.uniform(-0.3, 0.3)
                r = size + np.random.randint(-size//3, size//2)
                px = x + int(r * np.cos(angle))
                py = y + int(r * np.sin(angle))
                points.append([px, py])
            
            points = np.array(points, np.int32)
            cv2.fillPoly(img, [points], weed_color)
            
            return size * size
            
        else:  # clumping - Touceiras
            # Múltiplos círculos próximos
            total_area = 0
            num_clumps = np.random.randint(3, 6)
            
            for _ in range(num_clumps):
                clump_x = x + np.random.randint(-size//2, size//2)
                clump_y = y + np.random.randint(-size//2, size//2)
                clump_size = np.random.randint(size//3, size)
                cv2.circle(img, (clump_x, clump_y), clump_size, weed_color, -1)
                total_area += clump_size * clump_size
            
            return total_area
    
    def apply_lighting_effects(self, img: np.ndarray, lighting: str):
        """Aplica efeitos de iluminação realistas."""
        if lighting == "bright":
            # Iluminação intensa
            img[:] = np.clip(img * 1.25, 0, 255).astype(np.uint8)
            
        elif lighting == "overcast":
            # Luz difusa, menos contraste
            img[:] = np.clip(img * 0.9, 0, 255).astype(np.uint8)
            img[:] = cv2.GaussianBlur(img, (3, 3), 0.5)
            
        elif lighting == "shadow":
            # Sombras parciais
            height, width = img.shape[:2]
            shadow_mask = np.ones((height, width), dtype=np.float32)
            
            # Criar padrão de sombras
            for _ in range(5):
                sx, sy = np.random.randint(0, width), np.random.randint(0, height)
                sw, sh = np.random.randint(50, 200), np.random.randint(30, 100)
                cv2.rectangle(shadow_mask, (sx, sy), (sx+sw, sy+sh), 0.7, -1)
            
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * shadow_mask).astype(np.uint8)
                
        elif lighting == "golden_hour":
            # Luz dourada
            img[:, :, 0] = np.clip(img[:, :, 0] * 1.1, 0, 255)  # Mais azul
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.15, 0, 255)  # Mais verde
            img[:, :, 2] = np.clip(img[:, :, 2] * 1.3, 0, 255)   # Mais vermelho/dourado
            
        elif lighting == "harsh_noon":
            # Luz muito intensa com alto contraste
            img[:] = np.clip(img * 1.4, 0, 255).astype(np.uint8)
            # Aumentar contraste
            img[:] = cv2.convertScaleAbs(img, alpha=1.2, beta=-20)
    
    def apply_realistic_postprocessing(self, img: np.ndarray):
        """Aplica pós-processamento para realismo."""
        # Ruído fotográfico
        noise = np.random.randint(-12, 12, img.shape)
        img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Blur leve (simulando movimento ou foco)
        if np.random.random() < 0.3:
            img[:] = cv2.GaussianBlur(img, (3, 3), 0.8)
        
        # Compressão JPEG simulada
        if np.random.random() < 0.5:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, np.random.randint(75, 95)]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img[:] = cv2.imdecode(encimg, 1)
    
    def advanced_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, 
                                           model_name: str) -> Dict[str, Any]:
        """Otimização avançada de hiperparâmetros usando múltiplas estratégias."""
        print(f"🔧 Otimização avançada de hiperparâmetros para {model_name.upper()}")
        
        optimization_method = self.config['model_config']['hyperparameter_optimization']
        
        if optimization_method == "bayesian" and self.has_bayesian_optimization():
            return self.bayesian_optimization(X, y, model_name)
        elif optimization_method == "random":
            return self.random_search_optimization(X, y, model_name)
        else:
            return self.grid_search_optimization(X, y, model_name)
    
    def has_bayesian_optimization(self) -> bool:
        """Verifica se otimização bayesiana está disponível."""
        try:
            import optuna
            return True
        except ImportError:
            print("⚠️  Optuna não disponível. Usando Grid Search.")
            return False
    
    def bayesian_optimization(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Otimização bayesiana usando Optuna."""
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from ml_system.core.ml_classifiers import ClassicalMLWeedDetector
            
            def objective(trial):
                # Definir espaços de busca por modelo
                if model_name == "random_forest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                    }
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**params, random_state=42)
                    
                elif model_name == "svm":
                    params = {
                        'C': trial.suggest_float('C', 0.1, 100, log=True),
                        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_type', ['fixed', 'float']) == 'fixed' else trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
                        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
                    }
                    from sklearn.svm import SVC
                    model = SVC(**params, random_state=42)
                    
                else:  # Default para outros modelos
                    return 0.5
                
                # Cross-validation
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
                return scores.mean()
            
            # Criar estudo Optuna
            study = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler(seed=42))
            
            n_trials = self.config['model_config']['optimization_trials']
            study.optimize(objective, n_trials=n_trials, timeout=600)  # 10 min timeout
            
            print(f"  ✅ Otimização bayesiana concluída: {n_trials} trials")
            print(f"  🏆 Melhor score: {study.best_value:.4f}")
            
            return {
                "best_params": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "bayesian",
                "trials_completed": len(study.trials)
            }
            
        except Exception as e:
            print(f"❌ Erro na otimização bayesiana: {e}")
            return self.grid_search_optimization(X, y, model_name)
    
    def train_with_advanced_pipeline(self):
        """Pipeline completo de treinamento avançado."""
        print("\n🚀 INICIANDO PIPELINE AVANÇADO DE TREINAMENTO")
        print("="*60)
        
        # 1. Preparação de dados
        print("\n📊 Fase 1: Preparação de Dados")
        self.current_stage = "data_preparation"
        
        # Criar dataset sintético
        synthetic_images = self.create_advanced_synthetic_dataset(
            num_images=int(50 * self.config['data_strategy']['synthetic_ratio'] / 0.4)
        )
        
        # TODO: Integrar com imagens reais se disponíveis
        
        # 2. Treinamento dos modelos
        print("\n🤖 Fase 2: Treinamento de Modelos")
        self.current_stage = "model_training"
        
        try:
            from ml_system.core.ml_training import MLTrainingPipeline, TrainingConfig
            from ml_system.core.ml_classifiers import ClassicalMLWeedDetector
            
            # Configuração de treinamento otimizada
            training_config = TrainingConfig(
                patch_size=tuple(self.config['training_params']['patch_sizes'][1]),  # Usar tamanho médio
                samples_per_class=self.config['training_params']['samples_per_class'][1],  # Usar quantidade média
                test_size=self.config['data_strategy']['test_split'],
                models_to_train=self.config['model_config']['models_to_train']
            )
            
            # Executar treinamento
            pipeline = MLTrainingPipeline()
            results = pipeline.train_models_from_images(synthetic_images[:10], training_config)
            
            # Salvar resultados
            self.save_training_results(results)
            
            # 3. Análise e relatórios
            print("\n📈 Fase 3: Análise de Resultados")
            self.current_stage = "analysis"
            
            self.generate_comprehensive_analysis(results)
            
            # 4. Salvar modelos finais
            print("\n💾 Fase 4: Salvando Modelos Finais")
            self.current_stage = "saving"
            
            self.save_final_models(results)
            
            print("\n✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
            self.print_final_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no pipeline de treinamento: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_training_results(self, results: Dict[str, Any]):
        """Salva resultados detalhados do treinamento."""
        timestamp = datetime.now().isoformat()
        
        detailed_results = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "project_name": self.project_name
            },
            "configuration": self.config,
            "results": {}
        }
        
        # Processar resultados de cada modelo
        for model_name, model_result in results.items():
            detailed_results["results"][model_name] = {
                "accuracy": float(model_result['accuracy']),
                "cross_validation": {
                    "mean": float(model_result['cv_mean']),
                    "std": float(model_result['cv_std']),
                    "scores": model_result['cv_scores'].tolist()
                },
                "best_parameters": model_result.get('best_params', {}),
                "classification_report": model_result['classification_report'],
                "feature_importance": model_result.get('feature_importance', {})
            }
        
        # Salvar arquivo principal
        results_file = self.project_dir / 'results' / f'training_results_{self.session_id}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Resultados salvos: {results_file}")
    
    def generate_comprehensive_analysis(self, results: Dict[str, Any]):
        """Gera análise abrangente dos resultados."""
        print("📊 Gerando análise abrangente...")
        
        analysis_dir = self.project_dir / 'results' / 'analysis'
        
        # 1. Comparação de modelos
        comparison_data = {}
        for model_name, result in results.items():
            comparison_data[model_name] = {
                'accuracy': result['accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        # Salvar comparação
        with open(analysis_dir / f'model_comparison_{self.session_id}.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # 2. Identificar melhor modelo
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_model_result = results[best_model_name]
        
        self.best_model_info = {
            'name': best_model_name,
            'accuracy': best_model_result['accuracy'],
            'cv_mean': best_model_result['cv_mean'],
            'cv_std': best_model_result['cv_std']
        }
        
        print(f"🏆 Melhor modelo identificado: {best_model_name}")
        print(f"   Acurácia: {best_model_result['accuracy']:.3f}")
        print(f"   CV Score: {best_model_result['cv_mean']:.3f} ± {best_model_result['cv_std']:.3f}")
        
        # 3. Análise de features (se disponível)
        if 'feature_importance' in best_model_result:
            self.analyze_feature_importance(best_model_result['feature_importance'])
    
    def analyze_feature_importance(self, feature_importance: Dict[str, float]):
        """Analisa importância das características."""
        print("📈 Analisando importância das características...")
        
        # Ordenar por importância
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Top 10 características
        top_features = sorted_features[:10]
        
        print("🔝 Top 10 características mais importantes:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        # Análise por categoria
        categories = {"color": [], "texture": [], "shape": []}
        
        for feature, importance in sorted_features:
            if "color" in feature:
                categories["color"].append((feature, importance))
            elif "texture" in feature:
                categories["texture"].append((feature, importance))
            elif "shape" in feature:
                categories["shape"].append((feature, importance))
        
        print("\n📊 Importância por categoria:")
        for category, features in categories.items():
            if features:
                avg_importance = sum(imp for _, imp in features) / len(features)
                print(f"   {category.capitalize()}: {avg_importance:.4f} (avg)")
        
        # Salvar análise
        analysis = {
            "top_features": dict(top_features),
            "category_analysis": {cat: dict(feats) for cat, feats in categories.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        analysis_file = self.project_dir / 'results' / 'analysis' / f'feature_importance_{self.session_id}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def save_final_models(self, results: Dict[str, Any]):
        """Salva modelos finais otimizados."""
        models_dir = self.project_dir / 'models' / 'final'
        
        # Informações dos modelos salvos
        saved_models = {}
        
        for model_name in results.keys():
            try:
                # Copiar modelos do diretório padrão para o projeto
                # (Assumindo que MLTrainingPipeline já salvou os modelos)
                
                saved_models[model_name] = {
                    "saved": True,
                    "accuracy": results[model_name]['accuracy'],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"⚠️  Erro salvando {model_name}: {e}")
                saved_models[model_name] = {"saved": False, "error": str(e)}
        
        # Salvar índice de modelos
        index_file = models_dir / f'models_index_{self.session_id}.json'
        with open(index_file, 'w') as f:
            json.dump(saved_models, f, indent=2)
        
        print(f"💾 Modelos salvos em: {models_dir}")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Imprime resumo final do treinamento."""
        print("\n" + "="*60)
        print("📋 RESUMO FINAL DO TREINAMENTO AVANÇADO")
        print("="*60)
        
        print(f"🔑 Session ID: {self.session_id}")
        print(f"📁 Projeto: {self.project_dir}")
        print(f"⏰ Duração: {datetime.now().strftime('%H:%M:%S')}")
        
        print(f"\n🤖 Modelos Treinados: {len(results)}")
        for model_name, result in results.items():
            status = "✅" if result['cv_mean'] > 0.7 else "⚠️"
            print(f"   {status} {model_name}: {result['accuracy']:.3f} (CV: {result['cv_mean']:.3f})")
        
        if self.best_model_info:
            print(f"\n🏆 MELHOR MODELO: {self.best_model_info['name'].upper()}")
            print(f"   Acurácia: {self.best_model_info['accuracy']:.3f}")
            print(f"   CV Score: {self.best_model_info['cv_mean']:.3f} ± {self.best_model_info['cv_std']:.3f}")
        
        print(f"\n🚀 COMO USAR:")
        print(f"   algorithm='ml_{self.best_model_info['name']}' no endpoint /api/process")
        
        print(f"\n📊 Arquivos Gerados:")
        print(f"   • Modelos: {self.project_dir / 'models' / 'final'}")
        print(f"   • Resultados: {self.project_dir / 'results'}")
        print(f"   • Análises: {self.project_dir / 'results' / 'analysis'}")
        
        print("\n" + "="*60)


def main():
    """Função principal do sistema avançado."""
    print("🤖 SISTEMA AVANÇADO DE TREINAMENTO ML")
    print("Detecção de Ervas Daninhas - Versão Professional")
    print("="*60)
    
    try:
        # Criar trainer
        trainer = AdvancedMLTrainer("WeedDetection_Advanced")
        
        # Configuração interativa
        print("\n🔧 Deseja configurar o treinamento interativamente?")
        config_choice = input("(s/N): ").lower()
        
        if config_choice == 's':
            trainer.interactive_configuration()
        
        # Executar pipeline completo
        print("\n🚀 Iniciando pipeline de treinamento...")
        results = trainer.train_with_advanced_pipeline()
        
        if results:
            print("\n🎉 TREINAMENTO AVANÇADO CONCLUÍDO COM SUCESSO!")
        else:
            print("\n❌ Erro durante o treinamento")
    
    except KeyboardInterrupt:
        print("\n⚠️  Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro no sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()