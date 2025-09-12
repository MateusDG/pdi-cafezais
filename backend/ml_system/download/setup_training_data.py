#!/usr/bin/env python3
"""
Script simplificado para configurar dados de treinamento rapidamente.
Combina download automático + geração sintética + treinamento.
"""

import os
import sys
import json
from pathlib import Path
from typing import List

def quick_setup_training_data(num_images: int = 15):
    """
    Setup rápido de dados de treinamento.
    
    Args:
        num_images: Número de imagens por categoria para baixar
    """
    print("🚀 SETUP RÁPIDO DE DADOS DE TREINAMENTO")
    print("=" * 50)
    
    # 1. Tentar download automático
    try:
        print("📥 Tentando download automático...")
        from download_training_images import TrainingImageDownloader
        
        downloader = TrainingImageDownloader("quick_training_data")
        
        # Download mais conservador para teste
        total_downloaded = downloader.download_all_categories(images_per_category=num_images)
        
        if total_downloaded > 0:
            print(f"✅ Download concluído: {total_downloaded} imagens")
        else:
            print("⚠️ Download automático falhou, usando método alternativo...")
            raise Exception("Fallback to synthetic")
            
    except Exception as e:
        print(f"ℹ️ Usando geração sintética como fallback")
        total_downloaded = 0
    
    # 2. Sempre gerar imagens sintéticas (garantia)
    print("\n🎨 Gerando imagens sintéticas...")
    synthetic_dir = Path("quick_training_data/synthetic_generated")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_count = generate_comprehensive_synthetic_dataset(synthetic_dir, count=25)
    
    # 3. Criar lista consolidada
    all_images = collect_all_training_images("quick_training_data")
    
    print(f"\n📊 RESUMO:")
    print(f"  Imagens baixadas: {total_downloaded}")
    print(f"  Imagens sintéticas: {synthetic_count}")
    print(f"  Total disponível: {len(all_images)}")
    
    # 4. Salvar configuração
    save_training_config(all_images)
    
    return all_images

def generate_comprehensive_synthetic_dataset(output_dir: Path, count: int = 25) -> int:
    """
    Gera dataset sintético abrangente com diferentes cenários.
    """
    import cv2
    import numpy as np
    
    scenarios = [
        "heavy_weed_infestation",
        "light_weed_scattered", 
        "young_coffee_plantation",
        "mature_coffee_rows",
        "mixed_vegetation",
        "seasonal_variation",
        "different_lighting"
    ]
    
    generated = 0
    
    for i in range(count):
        scenario = scenarios[i % len(scenarios)]
        
        # Parâmetros baseados no cenário
        if scenario == "heavy_weed_infestation":
            weed_density = 0.4  # 40% de cobertura de ervas
            coffee_organization = 0.6
        elif scenario == "light_weed_scattered":
            weed_density = 0.15  # 15% de ervas
            coffee_organization = 0.9
        elif scenario == "young_coffee_plantation":
            weed_density = 0.25
            coffee_organization = 0.7
            coffee_size_modifier = 0.6  # Plantas menores
        else:
            weed_density = np.random.uniform(0.1, 0.3)
            coffee_organization = np.random.uniform(0.7, 0.9)
            coffee_size_modifier = 1.0
        
        img = create_advanced_synthetic_scene(
            scenario=scenario,
            weed_density=weed_density,
            coffee_organization=coffee_organization
        )
        
        filename = f"synthetic_{scenario}_{i:03d}.jpg"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        generated += 1
        
        if generated % 5 == 0:
            print(f"  ✅ Geradas {generated}/{count} imagens sintéticas")
    
    return generated

def create_advanced_synthetic_scene(scenario: str, weed_density: float = 0.2, 
                                  coffee_organization: float = 0.8) -> np.ndarray:
    """
    Cria cena sintética avançada baseada em cenário específico.
    """
    import cv2
    import numpy as np
    
    # Dimensões variadas
    width = np.random.randint(640, 1200)
    height = np.random.randint(480, 900)
    
    # Base: solo com textura realista
    base_color = np.random.randint(70, 130)
    img = np.ones((height, width, 3), dtype=np.uint8) * base_color
    
    # Textura do solo (crítica para ML aprender)
    for _ in range(width * height // 1000):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        radius = np.random.randint(2, 8)
        soil_variation = base_color + np.random.randint(-40, 40)
        soil_color = np.clip([soil_variation-10, soil_variation-5, soil_variation], 0, 255)
        cv2.circle(img, (x, y), radius, soil_color.tolist(), -1)
    
    # Plantas de café (organizadas)
    if coffee_organization > 0.5:
        row_spacing = np.random.randint(50, 90)
        plant_spacing = np.random.randint(35, 65)
        
        for row_y in range(30, height-30, row_spacing):
            for plant_x in range(20, width-20, plant_spacing):
                if np.random.random() < coffee_organization:
                    # Café: características consistentes
                    coffee_color = (
                        np.random.randint(0, 30),      # Baixo vermelho
                        np.random.randint(60, 120),    # Verde médio-escuro
                        np.random.randint(0, 40)       # Baixo azul
                    )
                    
                    radius = np.random.randint(15, 25)
                    
                    # Formato mais orgânico (não círculo perfeito)
                    for angle in range(0, 360, 30):
                        offset_x = int(radius * 0.7 * np.cos(np.radians(angle)))
                        offset_y = int(radius * 0.7 * np.sin(np.radians(angle)))
                        cv2.circle(img, (plant_x + offset_x, row_y + offset_y), 
                                 radius//3, coffee_color, -1)
                    
                    # Centro da planta
                    cv2.circle(img, (plant_x, row_y), radius, coffee_color, -1)
    
    # Ervas daninhas (crítico: padrão irregular)
    total_area = width * height
    weed_area_target = int(total_area * weed_density)
    current_weed_area = 0
    
    while current_weed_area < weed_area_target:
        # Posição aleatória (sem organização)
        x = np.random.randint(10, width-10)
        y = np.random.randint(10, height-10)
        
        # Ervas: cores mais claras e variadas
        weed_colors = [
            (np.random.randint(100, 180), np.random.randint(180, 255), np.random.randint(50, 150)),  # Verde-amarelo
            (np.random.randint(80, 140), np.random.randint(200, 255), np.random.randint(80, 140)),   # Verde claro
            (np.random.randint(120, 200), np.random.randint(220, 255), np.random.randint(100, 180)), # Verde lima
        ]
        
        weed_color = weed_colors[np.random.randint(0, len(weed_colors))]
        
        # Formato irregular (muito importante para ML distinguir)
        weed_size = np.random.randint(8, 20)
        num_points = np.random.randint(6, 12)
        
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            radius_variation = weed_size + np.random.randint(-5, 8)
            px = x + int(radius_variation * np.cos(angle))
            py = y + int(radius_variation * np.sin(angle))
            points.append([px, py])
        
        points = np.array(points, np.int32)
        cv2.fillPoly(img, [points], weed_color)
        
        current_weed_area += weed_size * weed_size
    
    # Efeitos de iluminação (realismo)
    if scenario == "different_lighting":
        # Gradiente de iluminação
        overlay = np.zeros_like(img)
        center_x, center_y = width//2, height//2
        
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                # Escurecimento radial
                darkness = int(30 * (distance / max_distance))
                overlay[y, x] = [darkness, darkness, darkness]
        
        img = cv2.subtract(img, overlay)
    
    # Ruído final (textura realista)
    noise = np.random.randint(-15, 15, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Blur leve para simular foto real
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img

def collect_all_training_images(base_dir: str) -> List[str]:
    """
    Coleta todas as imagens disponíveis para treinamento.
    """
    base_path = Path(base_dir)
    all_images = []
    
    # Buscar em todos os subdiretórios
    for img_file in base_path.rglob("*.jpg"):
        all_images.append(str(img_file))
    
    for img_file in base_path.rglob("*.jpeg"):
        all_images.append(str(img_file))
        
    for img_file in base_path.rglob("*.png"):
        all_images.append(str(img_file))
    
    return sorted(all_images)

def save_training_config(image_list: List[str]):
    """
    Salva configuração de treinamento para uso fácil.
    """
    config = {
        "training_images": image_list,
        "total_images": len(image_list),
        "recommended_config": {
            "patch_size": 64,
            "samples_per_class": min(500, len(image_list) * 50),
            "test_size": 0.2,
            "models_to_train": ["random_forest", "svm"]
        },
        "usage_instructions": [
            "1. Revisar imagens em quick_training_data/",
            "2. Remover imagens inadequadas",
            "3. Executar: python train_with_config.py",
            "4. Testar com: algorithm='ml_random_forest'"
        ]
    }
    
    config_file = Path("training_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuração salva em: {config_file}")
    
    # Criar script de treinamento simples
    create_training_script(image_list)

def create_training_script(image_list: List[str]):
    """
    Cria script Python simples para executar o treinamento.
    """
    script_content = f'''#!/usr/bin/env python3
"""
Script automático de treinamento gerado.
Execute: python train_with_config.py
"""

import sys
from pathlib import Path

# Adicionar backend ao path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    from app.services.processing.ml_training import MLTrainingPipeline, TrainingConfig
    
    # Lista de imagens (gerada automaticamente)
    image_paths = {image_list[:10]}  # Primeiras 10 para teste rápido
    
    # Configuração otimizada
    config = TrainingConfig(
        patch_size=(64, 64),
        samples_per_class=300,  # Ajuste conforme necessário
        test_size=0.2,
        models_to_train=['random_forest', 'svm']  # Melhores modelos
    )
    
    print("🚀 Iniciando treinamento dos modelos ML...")
    print(f"📊 Usando {{len(image_paths)}} imagens")
    
    # Treinar
    pipeline = MLTrainingPipeline()
    results = pipeline.train_models_from_images(image_paths, config)
    
    # Mostrar resultados
    print("\\n📊 RESULTADOS:")
    for model_name, result in results.items():
        accuracy = result['accuracy']
        cv_mean = result['cv_mean']
        print(f"  {{model_name}}: Acurácia = {{accuracy:.3f}} (CV = {{cv_mean:.3f}})")
    
    # Melhor modelo
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    print(f"\\n🏆 MELHOR MODELO: {{best_model}} ({{best_accuracy:.3f}})")
    print(f"\\n🚀 Para usar: algorithm='ml_{{best_model}}'")
    
    print("\\n✅ Treinamento concluído!")

if __name__ == "__main__":
    main()
'''
    
    script_file = Path("train_with_config.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"📜 Script de treinamento criado: {script_file}")

def main():
    """
    Função principal para setup rápido.
    """
    print("⚡ SETUP RÁPIDO DE DADOS DE TREINAMENTO")
    print("Detectar Ervas Daninhas em Cafezais")
    print("=" * 50)
    
    try:
        # Setup automático
        images = quick_setup_training_data(num_images=10)  # Conservador para teste
        
        if len(images) > 5:
            print("\\n🎉 SETUP CONCLUÍDO COM SUCESSO!")
            print("\\n📋 PRÓXIMOS PASSOS:")
            print("1. Executar: python train_with_config.py")
            print("2. Aguardar treinamento (pode demorar alguns minutos)")
            print("3. Testar com algorithm='ml_random_forest'")
            print("\\n📁 Imagens em: quick_training_data/")
        else:
            print("❌ Não foi possível obter imagens suficientes")
            print("Tente executar novamente ou verifique a conexão")
    
    except Exception as e:
        print(f"❌ Erro no setup: {e}")
        print("\\nTentando método alternativo...")
        
        # Fallback: só sintéticas
        from pathlib import Path
        synthetic_dir = Path("fallback_training")
        synthetic_dir.mkdir(exist_ok=True)
        
        count = generate_comprehensive_synthetic_dataset(synthetic_dir, 15)
        if count > 0:
            print(f"✅ Criadas {count} imagens sintéticas como fallback")

if __name__ == "__main__":
    main()