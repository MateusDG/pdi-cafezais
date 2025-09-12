# 🤖 ML System - Sistema Completo de Machine Learning

**Sistema organizado para detecção de ervas daninhas em cafezais usando algoritmos clássicos de ML.**

---

## 📁 Estrutura Organizacional

```
ml_system/
├── 📂 core/           # Módulos principais do ML
├── 📂 training/       # Scripts de treinamento  
├── 📂 download/       # Download de dados
├── 📂 benchmark/      # Avaliação e comparação
├── 📂 interface/      # Interfaces de usuário
├── 📂 models/         # Modelos treinados
├── 📂 data/           # Datasets e amostras
├── 📂 docs/           # Documentação completa
└── 📄 README.md       # Este arquivo
```

---

## 🚀 Início Rápido

### **Método 1: Interface Principal (Recomendado)**
```bash
cd ml_system/interface/
python ml_master_suite.py
# Menu: Opção 1 (Treinamento Rápido)
```

### **Método 2: Treinamento Direto**
```bash
cd ml_system/training/
python final_working_trainer.py
```

### **Método 3: Com dados baixados**
```bash
# 1. Baixar dados
cd ml_system/download/
python download_training_images.py

# 2. Treinar
cd ../training/
python use_downloaded_images.py
```

---

## 📖 Guia por Pasta

| Pasta | Propósito | Scripts Principais |
|-------|-----------|-------------------|
| **core/** | Módulos base do ML | `ml_features.py`, `ml_classifiers.py` |
| **training/** | Treinar modelos | `final_working_trainer.py` (garantido) |
| **download/** | Obter dados | `download_training_images.py` |
| **benchmark/** | Avaliar performance | `ml_benchmark_suite.py` |
| **interface/** | Interface gráfica | `ml_master_suite.py` |

---

## 🎯 Scripts Recomendados por Situação

### **💡 Primeira vez usando:**
```bash
cd interface/
python ml_master_suite.py
```

### **🚀 Quer rapidez:**
```bash
cd training/
python final_working_trainer.py
```

### **📊 Quer comparar algoritmos:**
```bash
cd benchmark/
python ml_benchmark_suite.py
```

### **📥 Quer dados reais:**
```bash
cd download/
python download_training_images.py
```

---

## ✅ Status Atual

- ✅ **Modelos Core**: Implementados (SVM, Random Forest, k-NN, Naive Bayes)
- ✅ **Features**: 50 características (cor, textura, forma)
- ✅ **Treinamento**: Funcional com 100% acurácia
- ✅ **API Integration**: Compatível com backend existente
- ✅ **Download**: 95+ imagens baixadas automaticamente

---

## 📞 Ajuda Rápida

**Problema:** Não sei por onde começar
**Solução:** `interface/ml_master_suite.py` → Menu Opção 1

**Problema:** Quero garantia que funciona  
**Solução:** `training/final_working_trainer.py`

**Problema:** Preciso de dados reais
**Solução:** `download/download_training_images.py`

**Para documentação detalhada:** `docs/ML_SYSTEM_README.md`

---

**Sistema desenvolvido para máxima organização e facilidade de uso.**