#!/usr/bin/env python3
"""
Treina todos os algoritmos ML automaticamente (sem interação)
"""

import os
import sys
from pathlib import Path

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def train_all_algorithms():
    """Treina todos os 4 algoritmos automaticamente."""
    print("🤖 TREINAMENTO AUTOMÁTICO DE TODOS OS ALGORITMOS")
    print("=" * 60)
    
    try:
        from ml_system.core.ml_classifiers import ClassicalMLWeedDetector
        from ml_system.core.ml_features import FeatureExtractor
        import numpy as np
        import cv2
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        print("📊 Criando dataset sintético...")
        
        # Criar dataset sintético balanceado
        extractor = FeatureExtractor()
        training_data = []
        
        # 100 amostras por classe
        samples_per_class = 100
        
        # Ervas daninhas
        for i in range(samples_per_class):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.circle(img, (50, 50), 20, [120, 200, 80], -1)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(mask, (50, 50), 25, 255, -1)
            training_data.append((img, mask, 'weed'))
        
        # Café
        for i in range(samples_per_class):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.circle(img, (50, 50), 18, [5, 90, 15], -1)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(mask, (50, 50), 22, 255, -1)
            training_data.append((img, mask, 'coffee'))
        
        # Solo
        for i in range(samples_per_class):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 80
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(mask, (50, 50), 30, 255, -1)
            training_data.append((img, mask, 'soil'))
        
        print(f"✅ Dataset criado: {len(training_data)} amostras")
        
        # Extrair características
        print("🔍 Extraindo características...")
        X = []
        y = []
        
        for i, (img, mask, label) in enumerate(training_data):
            features = extractor.extract_region_features(img, mask)
            if features:
                feature_vector = [features[key] for key in sorted(features.keys())]
                X.append(feature_vector)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Características extraídas: {X.shape}")
        print(f"✅ Classes: {np.unique(y)}")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📈 Treino: {X_train.shape[0]} amostras")
        print(f"📊 Teste: {X_test.shape[0]} amostras")
        
        # Treinar todos os algoritmos
        detector = ClassicalMLWeedDetector()
        algorithms = ['svm', 'random_forest', 'knn', 'naive_bayes']
        
        results = {}
        
        for algorithm in algorithms:
            print(f"\n🔄 Treinando {algorithm.upper()}...")
            
            try:
                # Treinar modelo
                detector.train_model(X_train, y_train, algorithm)
                
                # Testar
                y_pred = detector.predict_batch(X_test, algorithm)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[algorithm] = accuracy
                
                print(f"✅ {algorithm.upper()}: {accuracy:.3f} ({accuracy*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ Erro no {algorithm}: {e}")
                results[algorithm] = 0.0
        
        # Salvar modelos
        print(f"\n💾 Salvando modelos...")
        detector.save_models("models/classical_ml")
        
        print(f"\n🎯 RESULTADOS FINAIS:")
        print("=" * 40)
        for algorithm, accuracy in results.items():
            status = "✅" if accuracy > 0.8 else "⚠️" if accuracy > 0.5 else "❌"
            print(f"{status} {algorithm.upper():<15}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\n✅ TODOS OS MODELOS TREINADOS!")
        print(f"📁 Modelos salvos em: models/classical_ml/")
        print(f"\n🚀 Algoritmos disponíveis:")
        print(f"   • ml_svm")
        print(f"   • ml_random_forest") 
        print(f"   • ml_knn")
        print(f"   • ml_naive_bayes")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO no treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤖 INICIANDO TREINAMENTO AUTOMÁTICO")
    success = train_all_algorithms()
    
    if success:
        print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
    else:
        print("💥 TREINAMENTO FALHOU!")
        sys.exit(1)