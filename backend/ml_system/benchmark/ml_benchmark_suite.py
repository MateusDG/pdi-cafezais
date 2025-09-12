#!/usr/bin/env python3
"""
Suite de Benchmark e Comparação de Algoritmos ML
Compara performance entre métodos tradicionais e ML clássico.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


class MLBenchmarkSuite:
    """
    Suite completa para benchmark de algoritmos de detecção de ervas.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resultados
        self.results = {
            "session_info": {
                "benchmark_id": self.benchmark_id,
                "timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_info()
            },
            "algorithms": {},
            "comparison": {},
            "performance": {}
        }
        
        print(f"🏁 Suite de Benchmark Inicializada")
        print(f"📁 Resultados em: {self.output_dir}")
        print(f"🔑 Benchmark ID: {self.benchmark_id}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Coleta informações do sistema para contexto."""
        import platform
        import psutil
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version
        }
    
    def create_benchmark_dataset(self, num_images: int = 20) -> List[str]:
        """Cria dataset padronizado para benchmark."""
        print(f"🎨 Criando dataset de benchmark ({num_images} imagens)...")
        
        dataset_dir = self.output_dir / "benchmark_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Cenários padronizados para comparação justa
        benchmark_scenarios = [
            {"name": "light_infestation", "weed_ratio": 0.1, "complexity": "low"},
            {"name": "medium_infestation", "weed_ratio": 0.25, "complexity": "medium"},
            {"name": "heavy_infestation", "weed_ratio": 0.4, "complexity": "high"},
            {"name": "scattered_weeds", "weed_ratio": 0.15, "complexity": "medium"},
            {"name": "clustered_weeds", "weed_ratio": 0.3, "complexity": "high"},
        ]
        
        images_created = []
        
        for i in range(num_images):
            scenario = benchmark_scenarios[i % len(benchmark_scenarios)]
            
            img = self.create_benchmark_image(scenario, image_id=i)
            
            filename = f"benchmark_{scenario['name']}_{i:03d}.jpg"
            filepath = dataset_dir / filename
            
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            images_created.append(str(filepath))
            
            # Salvar ground truth (para métricas precisas)
            self.save_ground_truth(filepath, scenario)
        
        print(f"✅ Dataset de benchmark criado: {len(images_created)} imagens")
        return images_created
    
    def create_benchmark_image(self, scenario: Dict[str, Any], image_id: int) -> np.ndarray:
        """Cria imagem padronizada para benchmark."""
        # Dimensões fixas para comparação justa
        width, height = 800, 600
        
        # Solo base padronizado
        img = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Textura do solo consistente
        for _ in range(1000):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            cv2.circle(img, (x, y), np.random.randint(1, 3), 
                      (np.random.randint(80, 120), np.random.randint(75, 115), 
                       np.random.randint(70, 110)), -1)
        
        # Plantas de café padronizadas (sempre mesma distribuição)
        np.random.seed(42 + image_id)  # Seed controlada para reproduzibilidade
        
        row_spacing = 70
        plant_spacing = 50
        
        for row_y in range(35, height-35, row_spacing):
            for plant_x in range(25, width-25, plant_spacing):
                if np.random.random() < 0.8:  # 80% de densidade de café
                    coffee_color = (5, 90, 15)
                    radius = 16
                    cv2.circle(img, (plant_x, row_y), radius, coffee_color, -1)
        
        # Ervas daninhas baseadas no cenário
        self.add_benchmark_weeds(img, scenario)
        
        # Reset seed para variabilidade em outros aspectos
        np.random.seed(None)
        
        # Ruído controlado
        noise = np.random.randint(-8, 8, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def add_benchmark_weeds(self, img: np.ndarray, scenario: Dict[str, Any]):
        """Adiciona ervas com padrões controlados para benchmark."""
        height, width = img.shape[:2]
        total_area = width * height
        target_weed_area = int(total_area * scenario["weed_ratio"])
        
        weed_color = (120, 200, 80)  # Verde claro padrão
        
        if scenario["name"] == "scattered_weeds":
            # Ervas espalhadas uniformemente
            num_weeds = target_weed_area // 400  # Tamanho médio 20x20
            for _ in range(num_weeds):
                x = np.random.randint(15, width-15)
                y = np.random.randint(15, height-15)
                size = np.random.randint(8, 15)
                cv2.circle(img, (x, y), size, weed_color, -1)
        
        elif scenario["name"] == "clustered_weeds":
            # Ervas em grupos
            num_clusters = 5
            for _ in range(num_clusters):
                cluster_x = np.random.randint(50, width-50)
                cluster_y = np.random.randint(50, height-50)
                cluster_weeds = target_weed_area // (num_clusters * 300)
                
                for _ in range(cluster_weeds):
                    x = cluster_x + np.random.randint(-30, 30)
                    y = cluster_y + np.random.randint(-30, 30)
                    x, y = max(10, min(width-10, x)), max(10, min(height-10, y))
                    size = np.random.randint(6, 12)
                    cv2.circle(img, (x, y), size, weed_color, -1)
        
        else:
            # Ervas padrão (distribuição mista)
            current_area = 0
            while current_area < target_weed_area:
                x = np.random.randint(15, width-15)
                y = np.random.randint(15, height-15)
                size = np.random.randint(7, 14)
                cv2.circle(img, (x, y), size, weed_color, -1)
                current_area += size * size
    
    def save_ground_truth(self, image_path: Path, scenario: Dict[str, Any]):
        """Salva ground truth para métricas precisas."""
        gt_file = self.output_dir / "ground_truth" / f"{image_path.stem}.json"
        gt_file.parent.mkdir(exist_ok=True)
        
        ground_truth = {
            "image_file": str(image_path),
            "scenario": scenario,
            "expected_weed_ratio": scenario["weed_ratio"],
            "complexity_level": scenario["complexity"],
            "benchmark_id": self.benchmark_id
        }
        
        with open(gt_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
    
    def benchmark_traditional_methods(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark dos métodos tradicionais."""
        print("\n🔧 Benchmark dos Métodos Tradicionais")
        print("-" * 40)
        
        from app.services.processing import weed, robust_detection
        
        traditional_methods = {
            "HSV_Segmentation": lambda img: weed.detect_weeds_hsv(img, sensitivity=0.5),
            "Robust_ExGR_v1": lambda img: weed.detect_weeds_robust_v1(img, sensitivity=0.5),
            "Robust_ExGR_v2": lambda img: weed.detect_weeds_robust_v2(img, sensitivity=0.5),
            "Vegetation_Indices": lambda img: robust_detection.detect_weeds_robust_pipeline(img, algorithm="vegetation_indices")
        }
        
        results = {}
        
        for method_name, method_func in traditional_methods.items():
            print(f"  🔄 Testando {method_name}...")
            method_results = self.run_method_benchmark(method_func, test_images, method_name)
            results[method_name] = method_results
            
            avg_time = method_results["performance"]["avg_processing_time"]
            avg_accuracy = method_results["metrics"]["avg_accuracy_vs_ground_truth"]
            print(f"     ⏱️  Tempo médio: {avg_time:.3f}s")
            print(f"     🎯 Precisão média: {avg_accuracy:.3f}")
        
        return results
    
    def benchmark_ml_methods(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark dos métodos de ML clássico."""
        print("\n🤖 Benchmark dos Métodos de ML Clássico")
        print("-" * 40)
        
        try:
            from ml_system.core.ml_classifiers import ClassicalMLWeedDetector
            
            # Inicializar detector
            ml_detector = ClassicalMLWeedDetector()
            ml_detector.load_models()
            
            ml_methods = {}
            
            # Verificar quais modelos estão disponíveis
            for model_name, model in ml_detector.models.items():
                if model is not None:
                    ml_methods[f"ML_{model_name.upper()}"] = lambda img, mn=model_name: ml_detector.detect_weeds_ml(img, mn, confidence_threshold=0.6)
            
            if not ml_methods:
                print("  ⚠️  Nenhum modelo ML treinado encontrado")
                return {}
            
            results = {}
            
            for method_name, method_func in ml_methods.items():
                print(f"  🔄 Testando {method_name}...")
                method_results = self.run_method_benchmark(method_func, test_images, method_name)
                results[method_name] = method_results
                
                avg_time = method_results["performance"]["avg_processing_time"]
                avg_accuracy = method_results["metrics"]["avg_accuracy_vs_ground_truth"]
                print(f"     ⏱️  Tempo médio: {avg_time:.3f}s")
                print(f"     🎯 Precisão média: {avg_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Erro no benchmark ML: {e}")
            return {}
    
    def run_method_benchmark(self, method_func, test_images: List[str], method_name: str) -> Dict[str, Any]:
        """Executa benchmark de um método específico."""
        results = {
            "method_name": method_name,
            "performance": {
                "processing_times": [],
                "memory_usage": [],
                "avg_processing_time": 0.0,
                "std_processing_time": 0.0
            },
            "metrics": {
                "weed_counts": [],
                "weed_percentages": [],
                "accuracies_vs_ground_truth": [],
                "avg_accuracy_vs_ground_truth": 0.0
            },
            "detailed_results": []
        }
        
        for img_path in test_images:
            try:
                # Carregar imagem
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Medir tempo
                start_time = time.time()
                
                # Executar método
                detection_result = method_func(img_rgb)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Coletar métricas
                weed_count = detection_result.get('weed_count', 0)
                weed_percentage = detection_result.get('weed_percentage', 0.0)
                
                # Calcular precisão vs ground truth
                gt_accuracy = self.calculate_accuracy_vs_ground_truth(img_path, detection_result)
                
                # Salvar resultados
                results["performance"]["processing_times"].append(processing_time)
                results["metrics"]["weed_counts"].append(weed_count)
                results["metrics"]["weed_percentages"].append(weed_percentage)
                results["metrics"]["accuracies_vs_ground_truth"].append(gt_accuracy)
                
                results["detailed_results"].append({
                    "image": Path(img_path).name,
                    "processing_time": processing_time,
                    "weed_count": weed_count,
                    "weed_percentage": weed_percentage,
                    "accuracy_vs_ground_truth": gt_accuracy
                })
                
            except Exception as e:
                print(f"    ❌ Erro processando {Path(img_path).name}: {e}")
                results["detailed_results"].append({
                    "image": Path(img_path).name,
                    "error": str(e)
                })
        
        # Calcular estatísticas
        if results["performance"]["processing_times"]:
            results["performance"]["avg_processing_time"] = np.mean(results["performance"]["processing_times"])
            results["performance"]["std_processing_time"] = np.std(results["performance"]["processing_times"])
        
        if results["metrics"]["accuracies_vs_ground_truth"]:
            results["metrics"]["avg_accuracy_vs_ground_truth"] = np.mean(results["metrics"]["accuracies_vs_ground_truth"])
        
        return results
    
    def calculate_accuracy_vs_ground_truth(self, img_path: str, detection_result: Dict[str, Any]) -> float:
        """Calcula precisão em relação ao ground truth."""
        try:
            # Carregar ground truth
            gt_file = self.output_dir / "ground_truth" / f"{Path(img_path).stem}.json"
            
            if not gt_file.exists():
                return 0.5  # Valor neutro se não houver ground truth
            
            with open(gt_file, 'r') as f:
                ground_truth = json.load(f)
            
            expected_ratio = ground_truth["expected_weed_ratio"]
            detected_ratio = detection_result.get('weed_percentage', 0.0) / 100.0
            
            # Calcular precisão baseada na diferença relativa
            if expected_ratio == 0:
                accuracy = 1.0 if detected_ratio < 0.05 else 0.0
            else:
                relative_error = abs(detected_ratio - expected_ratio) / expected_ratio
                accuracy = max(0.0, 1.0 - relative_error)
            
            return accuracy
            
        except Exception as e:
            print(f"    ⚠️  Erro calculando precisão: {e}")
            return 0.5
    
    def generate_comparison_report(self, traditional_results: Dict[str, Any], 
                                 ml_results: Dict[str, Any]):
        """Gera relatório comparativo detalhado."""
        print("\n📊 Gerando Relatório Comparativo...")
        
        all_results = {**traditional_results, **ml_results}
        
        # Comparação de performance
        performance_comparison = {}
        accuracy_comparison = {}
        
        for method_name, method_results in all_results.items():
            performance_comparison[method_name] = {
                "avg_time": method_results["performance"]["avg_processing_time"],
                "std_time": method_results["performance"]["std_processing_time"]
            }
            
            accuracy_comparison[method_name] = {
                "avg_accuracy": method_results["metrics"]["avg_accuracy_vs_ground_truth"]
            }
        
        # Encontrar melhor método
        best_accuracy_method = max(accuracy_comparison.keys(), 
                                 key=lambda k: accuracy_comparison[k]["avg_accuracy"])
        best_speed_method = min(performance_comparison.keys(),
                              key=lambda k: performance_comparison[k]["avg_time"])
        
        # Compilar relatório
        comparison_report = {
            "benchmark_info": {
                "benchmark_id": self.benchmark_id,
                "timestamp": datetime.now().isoformat(),
                "total_methods_tested": len(all_results),
                "traditional_methods": len(traditional_results),
                "ml_methods": len(ml_results)
            },
            "performance_ranking": sorted(performance_comparison.items(), 
                                        key=lambda x: x[1]["avg_time"]),
            "accuracy_ranking": sorted(accuracy_comparison.items(),
                                     key=lambda x: x[1]["avg_accuracy"], reverse=True),
            "best_methods": {
                "highest_accuracy": {
                    "method": best_accuracy_method,
                    "accuracy": accuracy_comparison[best_accuracy_method]["avg_accuracy"]
                },
                "fastest": {
                    "method": best_speed_method,
                    "time": performance_comparison[best_speed_method]["avg_time"]
                }
            },
            "detailed_comparison": {
                "performance": performance_comparison,
                "accuracy": accuracy_comparison
            }
        }
        
        # Salvar relatório
        report_file = self.output_dir / f"comparison_report_{self.benchmark_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        self.print_comparison_summary(comparison_report)
        
        return comparison_report
    
    def print_comparison_summary(self, report: Dict[str, Any]):
        """Imprime resumo da comparação."""
        print("\n" + "="*60)
        print("📈 RELATÓRIO COMPARATIVO FINAL")
        print("="*60)
        
        print(f"\n🔍 Métodos testados: {report['benchmark_info']['total_methods_tested']}")
        print(f"   • Tradicionais: {report['benchmark_info']['traditional_methods']}")
        print(f"   • ML Clássico: {report['benchmark_info']['ml_methods']}")
        
        print(f"\n🏆 MELHORES MÉTODOS:")
        best_acc = report['best_methods']['highest_accuracy']
        best_speed = report['best_methods']['fastest']
        
        print(f"   🎯 Maior Precisão: {best_acc['method']}")
        print(f"      Precisão: {best_acc['accuracy']:.3f}")
        
        print(f"   ⚡ Mais Rápido: {best_speed['method']}")
        print(f"      Tempo: {best_speed['time']:.3f}s")
        
        print(f"\n📊 RANKING POR PRECISÃO:")
        for i, (method, metrics) in enumerate(report['accuracy_ranking'][:5], 1):
            print(f"   {i}. {method}: {metrics['avg_accuracy']:.3f}")
        
        print(f"\n⚡ RANKING POR VELOCIDADE:")
        for i, (method, metrics) in enumerate(report['performance_ranking'][:5], 1):
            print(f"   {i}. {method}: {metrics['avg_time']:.3f}s")
        
        print(f"\n📁 Relatório completo: comparison_report_{self.benchmark_id}.json")
        print("="*60)
    
    def run_full_benchmark(self, num_test_images: int = 20) -> Dict[str, Any]:
        """Executa benchmark completo."""
        print("🚀 INICIANDO BENCHMARK COMPLETO")
        print("="*50)
        
        # 1. Criar dataset de teste
        test_images = self.create_benchmark_dataset(num_test_images)
        
        # 2. Benchmark métodos tradicionais
        traditional_results = self.benchmark_traditional_methods(test_images)
        
        # 3. Benchmark métodos ML
        ml_results = self.benchmark_ml_methods(test_images)
        
        # 4. Gerar comparação
        if traditional_results or ml_results:
            comparison_report = self.generate_comparison_report(traditional_results, ml_results)
        else:
            print("❌ Nenhum método pôde ser testado")
            return {}
        
        # 5. Salvar resultados completos
        complete_results = {
            "traditional_methods": traditional_results,
            "ml_methods": ml_results,
            "comparison_report": comparison_report
        }
        
        complete_file = self.output_dir / f"complete_benchmark_{self.benchmark_id}.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ BENCHMARK COMPLETO FINALIZADO")
        print(f"📁 Resultados em: {self.output_dir}")
        
        return complete_results


def main():
    """Função principal do benchmark."""
    print("🏁 SUITE DE BENCHMARK ML - DETECÇÃO DE ERVAS DANINHAS")
    print("="*60)
    
    try:
        # Criar suite de benchmark
        benchmark = MLBenchmarkSuite()
        
        # Configuração
        print("\n⚙️  Configuração do Benchmark:")
        num_images = input("Número de imagens de teste (padrão: 20): ").strip()
        num_images = int(num_images) if num_images.isdigit() else 20
        
        # Executar benchmark completo
        results = benchmark.run_full_benchmark(num_images)
        
        if results:
            print("\n🎉 Benchmark executado com sucesso!")
            
            # Sugestões baseadas nos resultados
            if 'comparison_report' in results:
                best_method = results['comparison_report']['best_methods']['highest_accuracy']['method']
                print(f"\n💡 RECOMENDAÇÃO: Use '{best_method}' para melhor precisão")
        else:
            print("\n❌ Falha na execução do benchmark")
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro no benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()