#!/usr/bin/env python3
"""
Script para treinar usando as imagens já baixadas
"""

import os
import sys
from pathlib import Path
import glob

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def find_downloaded_images():
    """Encontra imagens já baixadas."""
    print("Procurando imagens baixadas...")
    
    # Locais onde podem estar as imagens
    search_paths = [
        "training_dataset/**/*.jpg",
        "training_images/**/*.jpg", 
        "quick_training_data/**/*.jpg",
        "data/**/*.jpg",
        "simple_training/*.jpg"
    ]
    
    all_images = []
    
    for pattern in search_paths:
        images = glob.glob(pattern, recursive=True)
        all_images.extend(images)
    
    # Remover duplicatas e manter apenas arquivos existentes
    unique_images = []
    seen = set()
    
    for img_path in all_images:
        abs_path = os.path.abspath(img_path)
        if abs_path not in seen and os.path.exists(abs_path):
            unique_images.append(abs_path)
            seen.add(abs_path)
    
    print(f"Encontradas {len(unique_images)} imagens:")
    for img in unique_images[:10]:  # Mostrar primeiras 10
        print(f"  - {Path(img).name}")
    if len(unique_images) > 10:
        print(f"  ... e mais {len(unique_images) - 10} imagens")
    
    return unique_images

def train_with_existing_images():
    """Treina usando imagens existentes."""
    
    # Encontrar imagens
    images = find_downloaded_images()
    
    if len(images) < 3:
        print("ERRO: Necessario pelo menos 3 imagens")
        print("Execute primeiro: python download_training_images.py")
        return False
    
    try:
        print(f"\nIniciando treinamento com {len(images)} imagens...")
        
        from ml_system.core.ml_training import MLTrainingPipeline, TrainingConfig
        
        # Usar apenas primeiras 8 imagens para velocidade
        training_images = images[:8]
        
        # Configuracao otimizada
        config = TrainingConfig(
            patch_size=(64, 64),
            samples_per_class=200,  # Reduzido para teste
            test_size=0.3,
            models_to_train=['random_forest']  # Apenas melhor modelo
        )
        
        print(f"Configuracao: {config.samples_per_class} amostras por classe")
        print("Modelos: Random Forest")
        
        # Executar treinamento
        pipeline = MLTrainingPipeline()
        results = pipeline.train_models_from_images(training_images, config)
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("RESULTADOS DO TREINAMENTO:")
        print("="*50)
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            
            print(f"\n{model_name.upper()}:")
            print(f"  Acuracia: {accuracy:.3f}")
            print(f"  Cross-validation: {cv_mean:.3f} +/- {cv_std:.3f}")
            
            if accuracy > 0.8:
                print(f"  Status: EXCELENTE!")
            elif accuracy > 0.7:
                print(f"  Status: BOM")
            else:
                print(f"  Status: Pode melhorar")
        
        print(f"\n" + "="*50)
        print("TREINAMENTO CONCLUIDO COM SUCESSO!")
        print("="*50)
        print("\nCOMO USAR:")
        print("1. Inicie o backend:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("\n2. Use no endpoint /api/process:")
        print("   algorithm='ml_random_forest'")
        print("\n3. Ou teste via browser:")
        print("   http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("TREINAMENTO COM IMAGENS BAIXADAS")
    print("Usa as imagens ja obtidas pelo download_training_images.py")
    print("="*60)
    
    success = train_with_existing_images()
    
    if not success:
        print("\nFALHA no treinamento")
        print("\nPossibles solucoes:")
        print("1. Execute: python download_training_images.py")
        print("2. Ou use: python quick_training.py (sinteticas)")
        print("3. Verifique se scikit-learn esta instalado")

if __name__ == "__main__":
    main()