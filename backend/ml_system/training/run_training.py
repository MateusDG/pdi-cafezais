#!/usr/bin/env python3
"""
Script ultra-simplificado para treinar modelos ML para detecção de ervas.
Execute apenas: python run_training.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Adicionar backend ao path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def create_instant_training_dataset():
    """Cria dataset de treinamento instantâneo com imagens sintéticas."""
    print("🎨 Criando dataset de treinamento instantâneo...")
    
    dataset_dir = Path("instant_training")
    dataset_dir.mkdir(exist_ok=True)
    
    images_created = []
    
    # Criar 8 imagens sintéticas variadas
    scenarios = [
        {"weeds": "high", "coffee": "organized", "lighting": "normal"},
        {"weeds": "low", "coffee": "organized", "lighting": "bright"},
        {"weeds": "medium", "coffee": "scattered", "lighting": "normal"},
        {"weeds": "high", "coffee": "young", "lighting": "shadow"},
        {"weeds": "low", "coffee": "mature", "lighting": "normal"},
        {"weeds": "medium", "coffee": "organized", "lighting": "overcast"},
        {"weeds": "high", "coffee": "mixed", "lighting": "normal"},
        {"weeds": "scattered", "coffee": "organized", "lighting": "bright"}
    ]
    
    for i, scenario in enumerate(scenarios):
        img = create_realistic_farm_image(scenario)
        filename = f"farm_scene_{i:02d}.jpg"
        filepath = dataset_dir / filename
        
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        images_created.append(str(filepath))
        
        print(f"  ✅ Criada: {filename}")
    
    print(f"🎯 Dataset pronto: {len(images_created)} imagens")
    return images_created

def create_realistic_farm_image(scenario):
    """Cria imagem realista de fazenda baseada no cenário."""
    width, height = 800, 600
    
    # Base: solo marrom variado
    soil_base = np.random.randint(70, 110)
    img = np.ones((height, width, 3), dtype=np.uint8) * soil_base
    
    # Textura do solo
    for _ in range(2000):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        color_var = np.random.randint(-25, 25)
        soil_color = np.clip(soil_base + color_var, 50, 150)
        cv2.circle(img, (x, y), np.random.randint(1, 4), 
                  [soil_color-10, soil_color, soil_color+10], -1)
    
    # Plantas de café baseadas no cenário
    coffee_config = scenario["coffee"]
    if coffee_config == "organized":
        add_organized_coffee_plants(img, density=0.7)
    elif coffee_config == "scattered":
        add_scattered_coffee_plants(img, density=0.5)
    elif coffee_config == "young":
        add_organized_coffee_plants(img, density=0.8, size_modifier=0.6)
    elif coffee_config == "mature":
        add_organized_coffee_plants(img, density=0.9, size_modifier=1.3)
    else:  # mixed
        add_organized_coffee_plants(img, density=0.6)
        add_scattered_coffee_plants(img, density=0.3)
    
    # Ervas daninhas baseadas no cenário
    weed_level = scenario["weeds"]
    if weed_level == "high":
        add_weeds(img, density=0.35)
    elif weed_level == "medium":
        add_weeds(img, density=0.20)
    elif weed_level == "low":
        add_weeds(img, density=0.10)
    else:  # scattered
        add_scattered_weeds(img, density=0.15)
    
    # Efeitos de iluminação
    lighting = scenario["lighting"]
    if lighting == "bright":
        img = np.clip(img * 1.2, 0, 255).astype(np.uint8)
    elif lighting == "shadow":
        img = np.clip(img * 0.8, 0, 255).astype(np.uint8)
    elif lighting == "overcast":
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Ruído final
    noise = np.random.randint(-10, 10, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def add_organized_coffee_plants(img, density=0.8, size_modifier=1.0):
    """Adiciona plantas de café em fileiras organizadas."""
    height, width = img.shape[:2]
    row_spacing = 70
    plant_spacing = 50
    
    for row_y in range(35, height-35, row_spacing):
        for plant_x in range(25, width-25, plant_spacing):
            if np.random.random() < density:
                # Cor do café: verde escuro
                coffee_color = [
                    int(np.random.randint(5, 25)),      # R baixo
                    int(np.random.randint(70, 110)),    # G médio-alto  
                    int(np.random.randint(10, 30))      # B baixo
                ]
                
                radius = int(np.random.randint(12, 18) * size_modifier)
                
                # Planta principal
                cv2.circle(img, (plant_x, row_y), radius, coffee_color, -1)
                
                # Folhas ao redor (mais realista)
                for angle in [0, 60, 120, 180, 240, 300]:
                    leaf_x = plant_x + int(radius * 0.6 * np.cos(np.radians(angle)))
                    leaf_y = row_y + int(radius * 0.6 * np.sin(np.radians(angle)))
                    cv2.circle(img, (leaf_x, leaf_y), radius//3, coffee_color, -1)

def add_scattered_coffee_plants(img, density=0.4):
    """Adiciona plantas de café espalhadas."""
    height, width = img.shape[:2]
    num_plants = int(width * height * density / 5000)
    
    for _ in range(num_plants):
        x = np.random.randint(20, width-20)
        y = np.random.randint(20, height-20)
        
        coffee_color = [
            int(np.random.randint(10, 30)),
            int(np.random.randint(60, 100)), 
            int(np.random.randint(15, 35))
        ]
        
        radius = np.random.randint(10, 16)
        cv2.circle(img, (x, y), radius, coffee_color, -1)

def add_weeds(img, density=0.2):
    """Adiciona ervas daninhas espalhadas."""
    height, width = img.shape[:2]
    total_area = width * height
    target_weed_area = int(total_area * density)
    current_area = 0
    
    while current_area < target_weed_area:
        x = np.random.randint(15, width-15)
        y = np.random.randint(15, height-15)
        
        # Cores de ervas: verdes mais claros e amarelados
        weed_colors = [
            [120, 200, 80],   # Verde-amarelo
            [100, 220, 100],  # Verde claro
            [140, 240, 120],  # Verde lima
            [110, 180, 60],   # Verde oliva claro
        ]
        
        weed_color = weed_colors[np.random.randint(0, len(weed_colors))]
        
        # Formato irregular (característica importante)
        weed_size = np.random.randint(6, 15)
        num_points = np.random.randint(5, 9)
        
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points + np.random.uniform(-0.3, 0.3)
            r = weed_size + np.random.randint(-3, 4)
            px = x + int(r * np.cos(angle))
            py = y + int(r * np.sin(angle))
            points.append([px, py])
        
        points = np.array(points, np.int32)
        cv2.fillPoly(img, [points], weed_color)
        
        current_area += weed_size * weed_size

def add_scattered_weeds(img, density=0.15):
    """Adiciona ervas mais esparsas."""
    height, width = img.shape[:2]
    num_weeds = int(width * height * density / 8000)
    
    for _ in range(num_weeds):
        x = np.random.randint(10, width-10)
        y = np.random.randint(10, height-10)
        
        weed_color = [
            int(np.random.randint(100, 160)),
            int(np.random.randint(180, 255)), 
            int(np.random.randint(80, 140))
        ]
        
        size = np.random.randint(8, 12)
        cv2.circle(img, (x, y), size, weed_color, -1)

def train_models_with_dataset(image_paths):
    """Treina modelos ML com o dataset fornecido."""
    print("🤖 Iniciando treinamento dos modelos ML...")
    
    try:
        from app.services.processing.ml_training import MLTrainingPipeline, TrainingConfig
        
        # Configuração otimizada para teste rápido
        config = TrainingConfig(
            patch_size=(64, 64),
            samples_per_class=400,  # Equilibrio velocidade/qualidade
            test_size=0.25,
            models_to_train=['random_forest']  # Apenas o melhor para velocidade
        )
        
        print(f"📊 Configuração: {config.samples_per_class} amostras/classe")
        print(f"🎯 Usando {len(image_paths)} imagens")
        
        # Executar treinamento
        pipeline = MLTrainingPipeline()
        results = pipeline.train_models_from_images(image_paths, config)
        
        # Mostrar resultados
        print("\n📊 RESULTADOS DO TREINAMENTO:")
        for model_name, result in results.items():
            accuracy = result['accuracy']
            cv_mean = result['cv_mean'] 
            cv_std = result['cv_std']
            
            print(f"  🎯 {model_name.upper()}:")
            print(f"    Acurácia: {accuracy:.3f}")
            print(f"    Cross-validation: {cv_mean:.3f} ± {cv_std:.3f}")
            
            if 'feature_importance' in result:
                print("    Top 3 características:")
                importance = result['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                for feature, score in top_features:
                    print(f"      {feature}: {score:.3f}")
        
        print("\n🎉 TREINAMENTO CONCLUÍDO!")
        print("\n🚀 COMO USAR:")
        print("  1. Inicie o backend: uvicorn app.main:app --reload")
        print("  2. Use algorithm='ml_random_forest' no endpoint /api/process")
        print("  3. Ou teste via API: /docs -> POST /api/process")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal: tudo automático."""
    print("🚀 TREINAMENTO AUTOMÁTICO DE DETECÇÃO DE ERVAS")
    print("Sistema completo em 3 minutos!")
    print("=" * 55)
    
    try:
        # 1. Criar dataset
        print("\n🎨 Passo 1: Criando dataset de treinamento...")
        images = create_instant_training_dataset()
        
        if len(images) < 3:
            print("❌ Erro: dataset insuficiente")
            return
        
        # 2. Treinar modelos
        print("\n🤖 Passo 2: Treinando modelos de Machine Learning...")
        success = train_models_with_dataset(images)
        
        if success:
            print("\n✅ SISTEMA PRONTO PARA USO!")
            print("\n📋 PRÓXIMOS PASSOS:")
            print("  • Inicie o backend")
            print("  • Use algorithm='ml_random_forest'")
            print("  • Upload de imagens e teste!")
        else:
            print("❌ Falha no treinamento")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        print("Tente executar novamente ou verifique as dependências")

if __name__ == "__main__":
    main()