# 📊 Benchmark - Avaliação e Comparação de Algoritmos

**Sistema completo para avaliar performance dos algoritmos de ML.**

---

## 📁 Arquivos

| Script | Propósito | Status | Tempo |
|--------|-----------|--------|-------|
| `ml_benchmark_suite.py` | **Benchmark completo** | ✅ Funcional | 5-15 min |

---

## 🏆 ml_benchmark_suite.py

**Sistema Profissional de Benchmark**

### **O que faz:**
- **Compara 4 algoritmos**: SVM, Random Forest, k-NN, Naive Bayes
- **Múltiplos datasets**: Sintético, real, híbrido
- **Métricas completas**: Accuracy, Precision, Recall, F1-score
- **Cross-validation** 5-fold
- **Matriz de confusão** detalhada
- **Tempo de execução** por algoritmo
- **Relatórios JSON** e visuais

### **Como usar:**
```bash
cd ml_system/benchmark/
python ml_benchmark_suite.py
```

### **Saída esperada:**
```
🔬 BENCHMARK SUITE - Comparação de Algoritmos ML

📊 Configuração:
- Algoritmos: 4 (SVM, Random Forest, k-NN, Naive Bayes)
- Dataset: Sintético balanceado (300 amostras)
- Cross-validation: 5-fold
- Métricas: Accuracy, Precision, Recall, F1

🤖 Testando SVM...
⏱️  Tempo: 2.34s | Accuracy: 0.987 ± 0.012

🌲 Testando Random Forest...
⏱️  Tempo: 1.87s | Accuracy: 1.000 ± 0.000

🎯 Testando k-NN...
⏱️  Tempo: 0.98s | Accuracy: 0.973 ± 0.018

📈 Testando Naive Bayes...
⏱️  Tempo: 0.45s | Accuracy: 0.891 ± 0.025

🏆 RANKING FINAL:
1. Random Forest  - 100.0% (±0.0%)
2. SVM           - 98.7% (±1.2%)
3. k-NN          - 97.3% (±1.8%)
4. Naive Bayes   - 89.1% (±2.5%)

📄 Relatórios salvos em: ../data/benchmark_results/
```

---

## 📈 Métricas Avaliadas

### **Principais:**
- **Accuracy**: Precisão geral
- **Precision**: Precisão por classe
- **Recall**: Cobertura por classe
- **F1-Score**: Média harmônica
- **Tempo de execução**
- **Desvio padrão** (robustez)

### **Matrizes de Confusão:**
```
Random Forest - Confusion Matrix:
           weed  coffee  soil
weed        100      0     0
coffee        0    100     0
soil          0      0   100
```

### **Feature Importance:**
- **Top 5 características** mais importantes
- **Análise por algoritmo** (quando disponível)
- **Contribuição relativa** em %

---

## 🎨 Recursos Visuais

### **Gráficos gerados:**
- **Barplot**: Comparação de accuracy
- **Boxplot**: Distribuição de scores
- **Confusion Matrix**: Heatmap por algoritmo
- **Feature Importance**: Top características

### **Formato de saída:**
- **PNG**: Gráficos de alta qualidade
- **JSON**: Dados brutos estruturados
- **CSV**: Resultados tabulares

---

## ⚙️ Configurações Avançadas

### **Parâmetros personalizáveis:**
```python
ALGORITHMS = ['svm', 'random_forest', 'knn', 'naive_bayes']
CV_FOLDS = 5              # Cross-validation folds
N_SAMPLES = 300           # Amostras por dataset
TEST_SIZE = 0.3           # Proporção de teste
RANDOM_STATE = 42         # Reproducibilidade
```

### **Datasets testados:**
1. **Sintético balanceado** (300 amostras)
2. **Imagens baixadas** (se disponível)
3. **Híbrido** (sintético + real)

---

## 📊 Análise de Resultados

### **Interpretação das métricas:**

**Accuracy > 95%**: ✅ Excelente
**Accuracy 90-95%**: ✅ Bom
**Accuracy 85-90%**: ⚠️ Aceitável
**Accuracy < 85%**: ❌ Precisa melhorar

### **Tempo de execução:**
- **< 1s**: Muito rápido
- **1-5s**: Rápido
- **5-15s**: Moderado
- **> 15s**: Lento

### **Robustez (desvio padrão):**
- **< 2%**: Muito estável
- **2-5%**: Estável
- **> 5%**: Instável

---

## 💡 Como Usar

### **1. Benchmark básico:**
```bash
python ml_benchmark_suite.py
```

### **2. Com dados reais (após download):**
```bash
# 1. Certificar que tem imagens
cd ../download/ && python download_training_images.py

# 2. Executar benchmark
cd ../benchmark/ && python ml_benchmark_suite.py
```

### **3. Analisar resultados:**
```bash
# Ver relatórios gerados
ls ../data/benchmark_results/
```

---

## 📁 Estrutura de Saída

```
../data/benchmark_results/
├── benchmark_report_YYYYMMDD_HHMMSS.json
├── confusion_matrices.png
├── algorithm_comparison.png
├── feature_importance.png
└── detailed_metrics.csv
```

### **Conteúdo dos arquivos:**
- **JSON**: Dados completos estruturados
- **PNG**: Visualizações profissionais
- **CSV**: Tabela para análise externa

---

## 🔍 Análise por Algoritmo

### **Random Forest:**
- ✅ **Melhor accuracy** (tipicamente 100%)
- ✅ **Feature importance** disponível
- ✅ **Robusto** a overfitting
- ⚠️ **Tempo moderado** de treinamento

### **SVM:**
- ✅ **Alta accuracy** (95-99%)
- ✅ **Bom** com dados complexos
- ⚠️ **Sensível** a hiperparâmetros
- ❌ **Mais lento** para treinar

### **k-NN:**
- ✅ **Muito rápido** para treinar
- ✅ **Simples** e interpretável
- ⚠️ **Lento** para predição
- ❌ **Sensível** a ruído

### **Naive Bayes:**
- ✅ **Extremamente rápido**
- ✅ **Boa baseline**
- ⚠️ **Assume independência** das features
- ❌ **Menor accuracy** geralmente

---

## 🐛 Troubleshooting

**Problema:** `ModuleNotFoundError: sklearn`
**Solução:** `pip install scikit-learn matplotlib seaborn`

**Problema:** Pouca memória
**Solução:** Reduza `N_SAMPLES` no código

**Problema:** Benchmark muito lento
**Solução:** Reduza `CV_FOLDS` ou use menos algoritmos

**Problema:** Gráficos não aparecem
**Solução:** Instale `pip install matplotlib seaborn`

---

## 🎯 Recomendações

### **Para produção:**
Use **Random Forest** (melhor accuracy + estabilidade)

### **Para velocidade:**
Use **k-NN** (mais rápido para treinar)

### **Para interpretabilidade:**
Use **Naive Bayes** (mais simples)

### **Para robustez:**
Use **SVM** (bom com dados ruidosos)

---

**Sistema completo para escolher o melhor algoritmo para seu caso de uso.**