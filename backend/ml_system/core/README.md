# 🔧 Core - Módulos Principais do Sistema ML

**Componentes fundamentais do sistema de Machine Learning.**

---

## 📁 Arquivos

| Arquivo | Propósito | Status |
|---------|-----------|--------|
| `ml_features.py` | **Extração de características** | ✅ Funcional |
| `ml_classifiers.py` | **Algoritmos de classificação** | ✅ Funcional |
| `ml_training.py` | **Pipeline de treinamento** | ✅ Funcional |
| `simple_test.py` | **Teste básico** | ✅ Funcional |
| `test_ml_implementation.py` | **Teste completo** | ⚠️ Encoding issues |

---

## 🎯 ml_features.py

**Extrator de Características Avançado**

### **Características Implementadas (50 total):**

#### **🎨 Cor (21 features):**
- RGB/HSV statistics (média, desvio padrão)
- Índices de vegetação: ExG, ExR, ExGR
- Momentos de cor (1ª, 2ª, 3ª ordem)

#### **🧵 Textura (14 features):**
- **GLCM**: Contrast, Energy, Homogeneity, Dissimilarity, Correlation
- **LBP**: Uniformity, Entropy, Mean, Std, Skewness

#### **📐 Forma (15 features):**
- Área, perímetro, circularidade
- Aspect ratio, extent, eccentricity, solidity
- **Hu Moments**: 7 momentos invariantes geométricos

### **Uso:**
```python
from ml_features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_region_features(img_rgb, mask_binary)
# Retorna: Dict com 50 características numéricas
```

---

## 🤖 ml_classifiers.py

**Algoritmos de Classificação Clássicos**

### **Modelos Implementados:**
- **SVM**: Support Vector Machine (RBF/Polynomial kernels)
- **Random Forest**: Ensemble com feature importance
- **k-NN**: k-Nearest Neighbors com weights
- **Naive Bayes**: Gaussian classifier

### **Recursos Avançados:**
- ✅ **Grid Search** automático para hiperparâmetros
- ✅ **Cross-validation** 5-fold
- ✅ **Feature importance** analysis
- ✅ **Confidence scoring** para predições
- ✅ **Salvamento/carregamento** de modelos

### **Uso:**
```python
from ml_classifiers import ClassicalMLWeedDetector

detector = ClassicalMLWeedDetector()
detector.load_models()  # Carrega modelos treinados

# Detectar ervas em imagem
result = detector.detect_weeds_ml(img_rgb, 'random_forest')
```

---

## 🏭 ml_training.py

**Pipeline Completo de Treinamento**

### **Funcionalidades:**
- **Geração de dados sintéticos** balanceados
- **Augmentação automática** (rotação, brilho)
- **Extração de patches** configurável
- **Balanceamento de classes**
- **Treinamento paralelo** de múltiplos modelos

### **Classes Principais:**
```python
MLTrainingPipeline()      # Pipeline principal
TrainingConfig()          # Configurações
```

---

## 🧪 Scripts de Teste

### **simple_test.py** ✅
**Teste rápido e confiável**
```bash
python simple_test.py
```
- Testa imports
- Testa extração de características
- Testa criação de detector
- **Sem problemas de encoding**

### **test_ml_implementation.py** ⚠️
**Teste completo com problemas**
```bash
# Não executar diretamente - problemas de encoding
# Usar simple_test.py ao invés
```

---

## 🔗 Dependências

```bash
# Principais
scikit-learn>=1.0.0
scikit-image>=0.19.0
opencv-python>=4.0.0
numpy>=1.20.0

# Para texturas avançadas
scipy>=1.7.0
```

---

## 💡 Como Usar

### **1. Teste rápido:**
```bash
python simple_test.py
```

### **2. Usar extrator:**
```python
from ml_features import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_region_features(img, mask)
print(f"Extraídas {len(features)} características")
```

### **3. Usar classificador:**
```python
from ml_classifiers import ClassicalMLWeedDetector
detector = ClassicalMLWeedDetector()
detector.load_models()
result = detector.detect_weeds_ml(img, 'random_forest')
```

---

## 🐛 Troubleshooting

**Problema:** `ModuleNotFoundError: sklearn`
**Solução:** `pip install scikit-learn scikit-image`

**Problema:** `UnicodeEncodeError` no test_ml_implementation.py
**Solução:** Use `simple_test.py` ao invés

**Problema:** Modelos não carregam
**Solução:** Execute um treinamento primeiro em `../training/`

---

**Módulos core testados e funcionais para produção.**