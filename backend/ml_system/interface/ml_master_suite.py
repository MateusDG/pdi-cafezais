#!/usr/bin/env python3
"""
ML Master Suite - Sistema Completo de Machine Learning
Integra treinamento avançado, benchmark e deployment.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


class MLMasterSuite:
    """
    Sistema master que coordena todas as funcionalidades ML.
    """
    
    def __init__(self):
        self.suite_dir = Path("ml_master_suite")
        self.suite_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_stage = "initialization"
        
        print("🎯 ML MASTER SUITE - SISTEMA COMPLETO")
        print("="*50)
        print(f"📁 Diretório: {self.suite_dir}")
        print(f"🔑 Session: {self.session_id}")
    
    def show_main_menu(self):
        """Mostra menu principal interativo."""
        while True:
            print("\n" + "="*50)
            print("🎯 ML MASTER SUITE - MENU PRINCIPAL")
            print("="*50)
            print("1. 🚀 Treinamento Rápido (3 min)")
            print("2. 🔧 Treinamento Avançado (configurável)")
            print("3. 🏁 Benchmark Completo")
            print("4. 📊 Análise de Resultados")
            print("5. 🔍 Status dos Modelos")
            print("6. 🛠️  Configurações")
            print("7. 📚 Ajuda e Documentação")
            print("8. ❌ Sair")
            print("="*50)
            
            choice = input("Escolha uma opção (1-8): ").strip()
            
            if choice == "1":
                self.quick_training()
            elif choice == "2":
                self.advanced_training()
            elif choice == "3":
                self.run_benchmark()
            elif choice == "4":
                self.analyze_results()
            elif choice == "5":
                self.check_model_status()
            elif choice == "6":
                self.configure_suite()
            elif choice == "7":
                self.show_help()
            elif choice == "8":
                print("👋 Encerrando ML Master Suite...")
                break
            else:
                print("❌ Opção inválida. Tente novamente.")
    
    def quick_training(self):
        """Executa treinamento rápido usando o script simples."""
        print("\n🚀 TREINAMENTO RÁPIDO")
        print("-" * 30)
        print("Este modo criará um dataset sintético e treinará um modelo Random Forest.")
        print("Tempo estimado: ~3 minutos")
        
        confirm = input("\nContinuar? (S/n): ").lower()
        if confirm in ['n', 'no', 'não']:
            return
        
        try:
            # Importar e executar script rápido garantido
            import sys
            import subprocess
            from pathlib import Path
            
            # Caminho para o script de treinamento
            training_script = Path(__file__).parent.parent / "training" / "final_working_trainer.py"
            
            print("\n⏳ Executando treinamento rápido...")
            print("📦 Usando final_working_trainer.py (garantido)")
            
            # Executar o script
            result = subprocess.run([sys.executable, str(training_script)], 
                                  capture_output=True, text=True, cwd=str(training_script.parent))
            
            if result.returncode == 0:
                print("\n✅ Treinamento rápido concluído!")
                print(result.stdout)
                self.check_model_status()
            else:
                print(f"\n❌ Erro durante execução:")
                print(result.stderr)
            
        except Exception as e:
            print(f"❌ Erro no treinamento rápido: {e}")
    
    def advanced_training(self):
        """Executa treinamento avançado com configurações."""
        print("\n🔧 TREINAMENTO AVANÇADO")
        print("-" * 30)
        print("Sistema completo com otimização de hiperparâmetros e análise detalhada.")
        
        try:
            print("\n⏳ Iniciando treinamento avançado...")
            print("📦 Executando advanced_ml_trainer.py")
            
            training_script = Path(__file__).parent.parent / "training" / "advanced_ml_trainer.py"
            result = subprocess.run([sys.executable, str(training_script)], 
                                  cwd=str(training_script.parent))
            
            if result.returncode == 0:
                print("\n✅ Treinamento avançado concluído!")
            else:
                print("\n❌ Erro durante o treinamento avançado")
                
        except Exception as e:
            print(f"❌ Erro no treinamento avançado: {e}")
            import traceback
            traceback.print_exc()
    
    def run_benchmark(self):
        """Executa suite de benchmark completa."""
        print("\n🏁 BENCHMARK COMPLETO")
        print("-" * 30)
        print("Compara todos os algoritmos disponíveis (tradicionais + ML).")
        
        try:
            print("\n⏳ Iniciando benchmark completo...")
            print("📦 Executando ml_benchmark_suite.py")
            
            benchmark_script = Path(__file__).parent.parent / "benchmark" / "ml_benchmark_suite.py"
            result = subprocess.run([sys.executable, str(benchmark_script)], 
                                  cwd=str(benchmark_script.parent))
            
            if result.returncode == 0:
                print("\n✅ Benchmark concluído!")
            else:
                print("\n❌ Erro durante o benchmark")
                
        except Exception as e:
            print(f"❌ Erro no benchmark: {e}")
    
    def analyze_results(self):
        """Analisa resultados de treinamentos e benchmarks anteriores."""
        print("\n📊 ANÁLISE DE RESULTADOS")
        print("-" * 30)
        
        # Buscar resultados disponíveis
        result_files = self.find_result_files()
        
        if not result_files:
            print("📭 Nenhum resultado encontrado.")
            print("Execute um treinamento ou benchmark primeiro.")
            return
        
        print("📁 Resultados encontrados:")
        for i, (file_type, filepath, timestamp) in enumerate(result_files, 1):
            print(f"   {i}. {file_type} - {timestamp}")
        
        try:
            choice = int(input("Escolha um resultado para analisar (número): ")) - 1
            if 0 <= choice < len(result_files):
                self.detailed_analysis(result_files[choice][1])
            else:
                print("❌ Escolha inválida.")
        except ValueError:
            print("❌ Por favor, digite um número válido.")
    
    def find_result_files(self) -> List[tuple]:
        """Encontra arquivos de resultados disponíveis."""
        result_files = []
        
        # Buscar resultados de treinamento
        for pattern in ["ml_project_*/results/training_results_*.json", 
                       "training_config.json",
                       "benchmark_*/complete_benchmark_*.json"]:
            for filepath in Path(".").glob(pattern):
                try:
                    stat = filepath.stat()
                    timestamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    
                    if "training_results" in str(filepath):
                        file_type = "Treinamento"
                    elif "benchmark" in str(filepath):
                        file_type = "Benchmark"
                    else:
                        file_type = "Configuração"
                    
                    result_files.append((file_type, filepath, timestamp))
                except:
                    continue
        
        return sorted(result_files, key=lambda x: x[2], reverse=True)
    
    def detailed_analysis(self, filepath: Path):
        """Análise detalhada de um arquivo de resultado."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n📊 ANÁLISE DETALHADA: {filepath.name}")
            print("=" * 50)
            
            if "training_results" in str(filepath):
                self.analyze_training_results(data)
            elif "benchmark" in str(filepath):
                self.analyze_benchmark_results(data)
            else:
                print("📄 Arquivo de configuração:")
                print(json.dumps(data, indent=2)[:500] + "...")
                
        except Exception as e:
            print(f"❌ Erro analisando arquivo: {e}")
    
    def analyze_training_results(self, data: Dict[str, Any]):
        """Analisa resultados de treinamento."""
        if "results" in data:
            results = data["results"]
            
            print("🤖 RESULTADOS DE TREINAMENTO:")
            print(f"   Modelos treinados: {len(results)}")
            
            # Ranking por acurácia
            model_scores = [(name, res["accuracy"]) for name, res in results.items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            print("\n🏆 Ranking por Acurácia:")
            for i, (model, accuracy) in enumerate(model_scores, 1):
                print(f"   {i}. {model}: {accuracy:.3f}")
            
            # Melhor modelo
            best_model = model_scores[0]
            print(f"\n⭐ Melhor modelo: {best_model[0]} ({best_model[1]:.3f})")
            
            # Cross-validation info
            if best_model[0] in results:
                cv_info = results[best_model[0]].get("cross_validation", {})
                if cv_info:
                    print(f"   CV Score: {cv_info.get('mean', 0):.3f} ± {cv_info.get('std', 0):.3f}")
    
    def analyze_benchmark_results(self, data: Dict[str, Any]):
        """Analisa resultados de benchmark."""
        if "comparison_report" in data:
            report = data["comparison_report"]
            
            print("🏁 RESULTADOS DE BENCHMARK:")
            print(f"   Métodos testados: {report['benchmark_info']['total_methods_tested']}")
            
            # Melhor precisão
            best_acc = report['best_methods']['highest_accuracy']
            print(f"\n🎯 Maior Precisão: {best_acc['method']}")
            print(f"   Precisão: {best_acc['accuracy']:.3f}")
            
            # Mais rápido
            best_speed = report['best_methods']['fastest']
            print(f"\n⚡ Mais Rápido: {best_speed['method']}")
            print(f"   Tempo: {best_speed['time']:.3f}s")
            
            # Top 3 por precisão
            print(f"\n📊 Top 3 Precisão:")
            for i, (method, metrics) in enumerate(report['accuracy_ranking'][:3], 1):
                print(f"   {i}. {method}: {metrics['avg_accuracy']:.3f}")
    
    def check_model_status(self):
        """Verifica status dos modelos disponíveis."""
        print("\n🔍 STATUS DOS MODELOS")
        print("-" * 30)
        
        try:
            from ml_system.core.ml_classifiers import ClassicalMLWeedDetector
            
            detector = ClassicalMLWeedDetector()
            detector.load_models()
            
            print("🤖 Modelos ML Clássico:")
            available_count = 0
            
            for model_name, model in detector.models.items():
                if model is not None:
                    print(f"   ✅ {model_name.upper()}: Disponível ({type(model).__name__})")
                    available_count += 1
                else:
                    print(f"   ❌ {model_name.upper()}: Não treinado")
            
            print(f"\n📊 Total disponível: {available_count}/4 modelos")
            
            if available_count > 0:
                print(f"\n🚀 Para usar: algorithm='ml_random_forest' (ou outro modelo)")
            else:
                print(f"\n💡 Execute um treinamento primeiro!")
                
            # Features disponíveis
            feature_count = len(detector.feature_extractor.get_all_feature_names())
            print(f"🔧 Características extraídas: {feature_count}")
            
        except Exception as e:
            print(f"❌ Erro verificando modelos: {e}")
    
    def configure_suite(self):
        """Configurações gerais da suite."""
        print("\n🛠️  CONFIGURAÇÕES")
        print("-" * 30)
        
        # Verificar dependências
        print("📦 Verificando dependências:")
        
        dependencies = [
            ("scikit-learn", "sklearn"),
            ("scikit-image", "skimage"), 
            ("OpenCV", "cv2"),
            ("NumPy", "numpy"),
            ("Matplotlib", "matplotlib"),
            ("Seaborn", "seaborn")
        ]
        
        missing = []
        
        for name, module in dependencies:
            try:
                __import__(module)
                print(f"   ✅ {name}")
            except ImportError:
                print(f"   ❌ {name} - FALTANDO")
                missing.append(name)
        
        if missing:
            print(f"\n⚠️  Dependências faltando: {', '.join(missing)}")
            install = input("Instalar automaticamente? (S/n): ").lower()
            
            if install not in ['n', 'no', 'não']:
                self.install_dependencies(missing)
        else:
            print(f"\n✅ Todas as dependências estão instaladas!")
        
        # Configurações de diretório
        print(f"\n📁 Diretórios:")
        print(f"   Suite: {self.suite_dir}")
        print(f"   Backend: {backend_dir}")
        
        # Espaço em disco
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            print(f"\n💾 Espaço em disco:")
            print(f"   Livre: {free // (1024**3)} GB")
            print(f"   Total: {total // (1024**3)} GB")
        except:
            pass
    
    def install_dependencies(self, missing: List[str]):
        """Instala dependências faltantes."""
        import subprocess
        
        package_map = {
            "scikit-learn": "scikit-learn",
            "scikit-image": "scikit-image",
            "Matplotlib": "matplotlib",
            "Seaborn": "seaborn"
        }
        
        for dep in missing:
            if dep in package_map:
                try:
                    print(f"📦 Instalando {dep}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_map[dep]])
                    print(f"   ✅ {dep} instalado")
                except subprocess.CalledProcessError:
                    print(f"   ❌ Erro instalando {dep}")
    
    def show_help(self):
        """Mostra ajuda e documentação."""
        print("\n📚 AJUDA E DOCUMENTAÇÃO")
        print("=" * 50)
        
        help_text = """
🎯 ML MASTER SUITE - GUIA RÁPIDO

1. TREINAMENTO RÁPIDO (Opção 1)
   • Cria dataset sintético automaticamente
   • Treina Random Forest em ~3 minutos
   • Ideal para testes e primeiras experiências

2. TREINAMENTO AVANÇADO (Opção 2) 
   • Configuração interativa completa
   • Múltiplos algoritmos (SVM, RF, k-NN, etc.)
   • Otimização de hiperparâmetros
   • Análise detalhada de características
   • Relatórios completos

3. BENCHMARK (Opção 3)
   • Compara métodos tradicionais vs ML
   • Métricas de precisão e velocidade
   • Dataset padronizado para comparação justa
   • Relatórios de performance

4. ANÁLISE DE RESULTADOS (Opção 4)
   • Visualiza resultados anteriores  
   • Ranking de modelos por performance
   • Estatísticas detalhadas

COMO USAR OS MODELOS TREINADOS:
• Inicie o backend: uvicorn app.main:app --reload
• Use algorithm='ml_random_forest' (ou outro)
• Endpoint: POST /api/process

DICAS:
• Comece com Treinamento Rápido
• Use Benchmark para comparar métodos
• Treinamento Avançado para produção
• Verifique Status dos Modelos regularmente

ARQUIVOS IMPORTANTES:
• final_working_trainer.py - Script rápido garantido
• advanced_ml_trainer.py - Sistema avançado
• ml_benchmark_suite.py - Benchmark completo
"""
        
        print(help_text)
        
        input("\nPressione Enter para continuar...")
    
    def save_training_summary(self, trainer, results):
        """Salva resumo do treinamento na suite."""
        summary = {
            "session_id": self.session_id,
            "trainer_project": trainer.project_name,
            "timestamp": datetime.now().isoformat(),
            "models_trained": list(results.keys()),
            "best_model": trainer.best_model_info,
            "status": "completed"
        }
        
        summary_file = self.suite_dir / f"training_summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Resumo salvo: {summary_file}")
    
    def save_benchmark_summary(self, results):
        """Salva resumo do benchmark na suite."""
        if 'comparison_report' in results:
            report = results['comparison_report']
            
            summary = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "best_accuracy_method": report['best_methods']['highest_accuracy'],
                "fastest_method": report['best_methods']['fastest'],
                "total_methods": report['benchmark_info']['total_methods_tested'],
                "status": "completed"
            }
            
            summary_file = self.suite_dir / f"benchmark_summary_{self.session_id}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📄 Resumo salvo: {summary_file}")


def main():
    """Função principal da Master Suite."""
    try:
        suite = MLMasterSuite()
        suite.show_main_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 ML Master Suite encerrada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro na Master Suite: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()