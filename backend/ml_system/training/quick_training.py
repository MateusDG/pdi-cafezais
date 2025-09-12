#!/usr/bin/env python3
"""
Quick Training Script - Versao simples sem caracteres especiais
Execute: python quick_training.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Adicionar backend ao path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def create_simple_coffee_image(width=800, height=600):
    """Cria imagem simples de cafezal com ervas."""
    # Base: solo marrom
    img = np.ones((height, width, 3), dtype=np.uint8) * 100
    
    # Adicionar textura do solo
    for _ in range(2000):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        soil_color = np.random.randint(80, 120)
        cv2.circle(img, (x, y), np.random.randint(1, 3), 
                  [soil_color-10, soil_color, soil_color+10], -1)
    
    # Plantas de cafe organizadas
    row_spacing = 70
    plant_spacing = 50
    
    for row_y in range(35, height-35, row_spacing):
        for plant_x in range(25, width-25, plant_spacing):
            if np.random.random() < 0.8:  # 80% de densidade
                # Cor do cafe: verde escuro
                coffee_color = [5, 90, 15]
                radius = 16
                cv2.circle(img, (plant_x, row_y), radius, coffee_color, -1)
    
    # Ervas daninhas espalhadas
    num_weeds = np.random.randint(15, 25)
    for _ in range(num_weeds):
        x = np.random.randint(15, width-15)
        y = np.random.randint(15, height-15)
        
        # Ervas: verde claro
        weed_color = [120, 200, 80]
        size = np.random.randint(8, 15)
        cv2.circle(img, (x, y), size, weed_color, -1)
    
    # Ruido final
    noise = np.random.randint(-10, 10, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def create_training_dataset():
    """Cria dataset de treinamento simples."""
    print("Criando dataset de treinamento...")
    
    dataset_dir = Path("simple_training")
    dataset_dir.mkdir(exist_ok=True)
    
    images_created = []
    
    # Criar 8 imagens variadas
    for i in range(8):
        img = create_simple_coffee_image()
        filename = f"coffee_scene_{i:02d}.jpg"
        filepath = dataset_dir / filename
        
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        images_created.append(str(filepath))
        
        print(f"  Criada: {filename}")
    
    print(f"Dataset pronto: {len(images_created)} imagens")
    return images_created

def train_models(image_paths):
    """Treina modelos ML."""
    print("Iniciando treinamento dos modelos...")
    
    try:
        from app.services.processing.ml_training import MLTrainingPipeline, TrainingConfig
        
        # Configuracao simples
        config = TrainingConfig(
            patch_size=(64, 64),
            samples_per_class=300,  # Reduzido para velocidade
            test_size=0.25,
            models_to_train=['random_forest']  # Apenas o melhor
        )
        
        print(f"Configuracao: {config.samples_per_class} amostras/classe")
        print(f"Usando {len(image_paths)} imagens")
        
        # Executar treinamento
        pipeline = MLTrainingPipeline()
        results = pipeline.train_models_from_images(image_paths, config)
        
        # Mostrar resultados
        print("\nRESULTADOS DO TREINAMENTO:")
        for model_name, result in results.items():
            accuracy = result['accuracy']
            cv_mean = result['cv_mean'] 
            cv_std = result['cv_std']
            
            print(f"  {model_name.upper()}:")
            print(f"    Acuracia: {accuracy:.3f}")
            print(f"    Cross-validation: {cv_mean:.3f} +/- {cv_std:.3f}")
        
        print("\nTREINAMENTO CONCLUIDO!")
        print("\nCOMO USAR:")
        print("  1. Inicie o backend: uvicorn app.main:app --reload")
        print("  2. Use algorithm='ml_random_forest' no endpoint /api/process")
        
        return True
        
    except Exception as e:
        print(f"ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Funcao principal."""
    print("TREINAMENTO RAPIDO DE DETECCAO DE ERVAS")
    print("Sistema simples em 3 minutos!")
    print("=" * 50)
    
    try:
        # 1. Criar dataset
        print("\nPasso 1: Criando dataset...")
        images = create_training_dataset()
        
        if len(images) < 3:
            print("ERRO: dataset insuficiente")
            return
        
        # 2. Treinar modelos
        print("\nPasso 2: Treinando modelos...")
        success = train_models(images)
        
        if success:
            print("\nSISTEMA PRONTO PARA USO!")
        else:
            print("FALHA no treinamento")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario")
    except Exception as e:
        print(f"ERRO geral: {e}")

if __name__ == "__main__":
    main()