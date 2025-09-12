#!/usr/bin/env python3
"""
Script de teste para a implementação de algoritmos de Machine Learning clássico
para detecção de ervas daninhas.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Adicionar o diretório do backend ao path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.processing.ml_features import FeatureExtractor
from app.services.processing.ml_classifiers import ClassicalMLWeedDetector
from app.services.processing.ml_training import MLTrainingPipeline, TrainingConfig


def create_synthetic_test_image(width=640, height=480):
    """
    Cria uma imagem sintética de teste com diferentes regiões.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Solo (marrom)
    img[300:, :, :] = [139, 69, 19]  # Marrom
    
    # Café (verde escuro)
    img[100:200, 100:250, :] = [34, 139, 34]  # Verde floresta
    img[120:180, 350:500, :] = [0, 100, 0]    # Verde escuro
    
    # Ervas daninhas (verde claro/amarelado)
    img[50:120, 200:300, :] = [124, 252, 0]   # Verde lima
    img[180:250, 180:280, :] = [173, 255, 47] # Verde amarelado
    img[250:320, 400:520, :] = [154, 205, 50] # Verde oliva
    
    # Adicionar ruído
    noise = np.random.randint(-20, 20, (height, width, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def create_test_mask(shape, region_type="weed"):
    """
    Cria uma máscara de teste para uma região específica.
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if region_type == "weed":
        # Região de erva daninha
        mask[50:120, 200:300] = 255
    elif region_type == "coffee":
        # Região de café
        mask[100:200, 100:250] = 255
    elif region_type == "soil":
        # Região de solo
        mask[300:400, 200:400] = 255
    
    return mask


def test_feature_extraction():
    """
    Testa a extração de características.
    """
    print("=== Testando Extração de Características ===")
    
    # Criar extrator
    extractor = FeatureExtractor()
    
    # Criar imagem de teste
    img = create_synthetic_test_image()
    
    # Testar diferentes tipos de região
    for region_type in ["weed", "coffee", "soil"]:
        mask = create_test_mask(img.shape, region_type)
        
        print(f"\nTestando região: {region_type}")
        
        try:
            features = extractor.extract_region_features(img, mask)
            
            print(f"  Características extraídas: {len(features)}")
            print(f"  Características de cor: {sum(1 for k in features.keys() if k.startswith('color_'))}")
            print(f"  Características de textura: {sum(1 for k in features.keys() if k.startswith('texture_'))}")
            print(f"  Características de forma: {sum(1 for k in features.keys() if k.startswith('shape_'))}")
            
            # Mostrar algumas características importantes
            important_features = ['color_exg', 'color_exr', 'color_exgr', 'shape_area', 'shape_circularity']
            for feature in important_features:
                if feature in features:
                    print(f"    {feature}: {features[feature]:.3f}")
                    
        except Exception as e:
            print(f"  ERRO: {e}")
    
    print("✓ Teste de extração de características concluído")


def test_ml_detector_creation():
    """
    Testa a criação do detector ML.
    """
    print("\n=== Testando Criação do Detector ML ===")
    
    try:
        detector = ClassicalMLWeedDetector()
        print(f"✓ Detector criado com sucesso")
        print(f"  Modelos disponíveis: {list(detector.models.keys())}")
        print(f"  Características totais: {len(detector.feature_extractor.get_all_feature_names())}")
        
        # Testar nomes de características
        feature_names = detector.feature_extractor.get_all_feature_names()
        print(f"  Primeiras 10 características: {feature_names[:10]}")
        
    except Exception as e:
        print(f"✗ ERRO ao criar detector: {e}")
        return False
    
    return True


def test_training_pipeline():
    """
    Testa o pipeline de treinamento com dados sintéticos.
    """
    print("\n=== Testando Pipeline de Treinamento ===")
    
    try:
        # Criar dados sintéticos
        synthetic_images = []
        synthetic_labels = []
        
        # Gerar 3 imagens de teste
        for i in range(3):
            img = create_synthetic_test_image(width=320, height=240)  # Menor para teste
            synthetic_images.append(img)
        
        print(f"✓ Geradas {len(synthetic_images)} imagens sintéticas")
        
        # Criar pipeline de treinamento
        pipeline = MLTrainingPipeline("data/test", "models/test")
        
        # Configuração de teste (poucos samples para teste rápido)
        config = TrainingConfig(
            patch_size=(32, 32),
            samples_per_class=50,  # Muito baixo para teste
            test_size=0.3,
            models_to_train=['random_forest']  # Apenas um modelo para teste
        )
        
        print("✓ Configuração de teste criada")
        print(f"  Patch size: {config.patch_size}")
        print(f"  Samples per class: {config.samples_per_class}")
        
        # Gerar dados de treinamento
        X, y = pipeline.generate_synthetic_training_data(synthetic_images, config)
        
        print(f"✓ Dados de treinamento gerados: {X.shape}")
        print(f"  Classes únicas: {np.unique(y)}")
        
        # Verificar se os dados estão balanceados
        unique, counts = np.unique(y, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"    {class_name}: {count} amostras")
        
    except Exception as e:
        print(f"✗ ERRO no pipeline de treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_basic_functionality():
    """
    Testa funcionalidades básicas da implementação.
    """
    print("=== Testando Funcionalidades Básicas ===")
    
    # Teste 1: Extração de características
    success1 = test_feature_extraction()
    
    # Teste 2: Criação do detector
    success2 = test_ml_detector_creation()
    
    # Teste 3: Pipeline de treinamento
    success3 = test_training_pipeline()
    
    print("\n=== Resumo dos Testes ===")
    print(f"Extração de características: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Criação do detector: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Pipeline de treinamento: {'✓ PASS' if success3 else '✗ FAIL'}")
    
    if all([success1, success2, success3]):
        print("\nTodos os testes passaram! Implementacao basica funcionando.")
    else:
        print("\nAlguns testes falharam. Verifique os erros acima.")


def main():
    """
    Função principal de teste.
    """
    print("Iniciando testes da implementacao de ML classico para deteccao de ervas daninhas")
    print("=" * 80)
    
    # Verificar dependências
    try:
        import sklearn
        import skimage
        import scipy
        print(f"✓ Dependências encontradas:")
        print(f"  scikit-learn: {sklearn.__version__}")
        print(f"  scikit-image: {skimage.__version__}")
        print(f"  scipy: {scipy.__version__}")
    except ImportError as e:
        print(f"❌ Erro de dependência: {e}")
        return
    
    # Executar testes
    test_basic_functionality()
    
    print("\n" + "=" * 80)
    print("Testes concluidos")


if __name__ == "__main__":
    main()