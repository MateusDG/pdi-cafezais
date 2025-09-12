#!/usr/bin/env python3
"""
Script para download automático de imagens de treinamento para detecção de ervas daninhas em cafezais.
Baixa imagens de múltiplas fontes com filtros de qualidade.
"""

import os
import sys
import requests
import time
import json
from pathlib import Path
from urllib.parse import urlparse, urljoin
import hashlib
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Tuple

# Instalar dependências necessárias
REQUIRED_PACKAGES = [
    "requests",
    "pillow", 
    "opencv-python",
    "beautifulsoup4",
    "bing-image-downloader"
]

def install_requirements():
    """Instala pacotes necessários se não estiverem disponíveis."""
    import subprocess
    import importlib
    
    for package in REQUIRED_PACKAGES:
        try:
            if package == "opencv-python":
                importlib.import_module("cv2")
            elif package == "pillow":
                importlib.import_module("PIL")
            elif package == "beautifulsoup4":
                importlib.import_module("bs4")
            elif package == "bing-image-downloader":
                importlib.import_module("bing_image_downloader")
            else:
                importlib.import_module(package)
        except ImportError:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class TrainingImageDownloader:
    """
    Classe para download eficiente de imagens de treinamento para detecção de ervas daninhas.
    """
    
    def __init__(self, output_dir: str = "training_images"):
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.downloaded_urls = set()
        self.stats = {
            'total_searched': 0,
            'downloaded': 0,
            'filtered_out': 0,
            'errors': 0
        }
        
    def setup_directories(self):
        """Cria estrutura de diretórios para organizar as imagens."""
        directories = [
            'coffee_with_weeds',      # Imagens de cafezal com ervas (PRIORITÁRIO)
            'aerial_drone_views',     # Vistas aéreas/drone
            'coffee_plantations',     # Plantações gerais de café
            'weed_management',        # Manejo de ervas
            'synthetic_generated',    # Imagens sintéticas geradas
            'filtered_quality',       # Imagens que passaram no filtro de qualidade
            'metadata'               # Metadados e logs
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Estrutura de diretórios criada em: {self.output_dir}")
    
    def get_search_terms(self) -> Dict[str, List[str]]:
        """Retorna termos de busca organizados por categoria."""
        return {
            'coffee_with_weeds': [
                "coffee plantation with weeds aerial view",
                "coffee farm weed infestation drone",
                "weedy coffee field management",
                "coffee plantation weed control before",
                "fazenda cafe ervas daninhas vista aerea",
                "coffee intercropping weeds birds eye",
                "coffee plantation weed problem aerial"
            ],
            
            'aerial_drone_views': [
                "coffee plantation drone photography Brazil",
                "aerial coffee farm birds eye view",
                "coffee plantation overhead view",
                "fazenda cafe vista drone top",
                "coffee estate aerial photography",
                "coffee plantation satellite view",
                "drone coffee farm inspection"
            ],
            
            'coffee_plantations': [
                "coffee plantation rows aerial",
                "coffee farm landscape drone",
                "coffee plantation sustainable farming",
                "coffee field different growth stages",
                "young coffee plants aerial view",
                "mature coffee plantation overhead",
                "coffee plantation various conditions"
            ],
            
            'weed_management': [
                "coffee weed management techniques",
                "coffee plantation before herbicide",
                "coffee farm weed identification",
                "agricultural weed detection coffee",
                "coffee plantation integrated pest management",
                "precision agriculture coffee weeds"
            ]
        }
    
    def download_with_bing(self, search_term: str, category: str, limit: int = 15) -> int:
        """Download usando Bing Image Downloader."""
        try:
            from bing_image_downloader import downloader
            
            print(f"🔍 Buscando no Bing: '{search_term}' (limite: {limit})")
            
            # Download para diretório temporário
            temp_dir = self.output_dir / "temp_bing"
            temp_dir.mkdir(exist_ok=True)
            
            downloader.download(
                search_term, 
                limit=limit,
                output_dir=str(temp_dir),
                adult_filter_off=True,
                force_replace=False,
                timeout=30
            )
            
            # Mover e filtrar imagens
            search_dir = temp_dir / search_term
            if search_dir.exists():
                return self._process_downloaded_images(search_dir, category)
            
        except Exception as e:
            print(f"❌ Erro no Bing download para '{search_term}': {e}")
            self.stats['errors'] += 1
            
        return 0
    
    def download_with_unsplash(self, search_term: str, category: str, limit: int = 10) -> int:
        """Download usando Unsplash API (requer chave, mas tem versão gratuita)."""
        # Unsplash tem API gratuita limitada
        base_url = "https://source.unsplash.com/800x600/?"
        
        downloaded = 0
        for i in range(limit):
            try:
                url = f"{base_url}{search_term.replace(' ', ',')}"
                
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    
                    # Verificar se não é duplicata
                    content_hash = hashlib.md5(response.content).hexdigest()
                    if content_hash not in self.downloaded_urls:
                        
                        filename = f"unsplash_{search_term.replace(' ', '_')}_{i}.jpg"
                        filepath = self.output_dir / category / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        if self._is_valid_image(filepath):
                            self.downloaded_urls.add(content_hash)
                            downloaded += 1
                            print(f"  ✅ Baixada: {filename}")
                        else:
                            filepath.unlink()
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Erro Unsplash: {e}")
                self.stats['errors'] += 1
        
        return downloaded
    
    def _process_downloaded_images(self, source_dir: Path, category: str) -> int:
        """Processa imagens baixadas, aplicando filtros de qualidade."""
        processed = 0
        
        for img_file in source_dir.rglob("*.jpg"):
            try:
                if self._is_valid_image(img_file):
                    # Gerar nome único
                    new_name = f"{category}_{int(time.time())}_{processed}.jpg"
                    dest_path = self.output_dir / category / new_name
                    
                    # Mover e redimensionar se necessário
                    self._process_and_save_image(img_file, dest_path)
                    processed += 1
                    
                    if processed % 5 == 0:
                        print(f"  📊 Processadas: {processed} imagens")
                        
                else:
                    self.stats['filtered_out'] += 1
                    
            except Exception as e:
                print(f"  ❌ Erro processando {img_file.name}: {e}")
                self.stats['errors'] += 1
        
        # Limpar diretório temporário
        import shutil
        if source_dir.parent.name == "temp_bing":
            shutil.rmtree(source_dir.parent, ignore_errors=True)
        
        return processed
    
    def _is_valid_image(self, filepath: Path) -> bool:
        """Verifica se a imagem atende aos critérios de qualidade."""
        try:
            # Verificar se é arquivo válido
            if not filepath.exists() or filepath.stat().st_size < 10000:  # Mínimo 10KB
                return False
            
            # Abrir com PIL para validar
            with Image.open(filepath) as img:
                # Verificar dimensões mínimas
                if img.width < 300 or img.height < 300:
                    return False
                
                # Verificar formato
                if img.format not in ['JPEG', 'JPG', 'PNG']:
                    return False
                
                # Verificar se não é muito escura ou clara (pode ser inválida)
                img_array = np.array(img)
                if len(img_array.shape) != 3:  # Deve ser colorida
                    return False
                
                mean_brightness = np.mean(img_array)
                if mean_brightness < 20 or mean_brightness > 240:  # Muito escura ou clara
                    return False
                
                return True
                
        except Exception:
            return False
    
    def _process_and_save_image(self, source_path: Path, dest_path: Path):
        """Processa e salva imagem com otimizações."""
        try:
            # Abrir imagem
            img = cv2.imread(str(source_path))
            
            if img is None:
                return False
            
            # Redimensionar se muito grande (otimização)
            height, width = img.shape[:2]
            if width > 1024 or height > 1024:
                scale_factor = 1024 / max(width, height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Salvar com compressão otimizada
            cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Registrar metadados
            self._save_metadata(dest_path, source_path, img.shape)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro processando {source_path}: {e}")
            return False
    
    def _save_metadata(self, dest_path: Path, source_path: Path, shape: Tuple[int, int, int]):
        """Salva metadados da imagem."""
        metadata = {
            'filename': dest_path.name,
            'source': str(source_path),
            'dimensions': f"{shape[1]}x{shape[0]}",
            'channels': shape[2],
            'download_time': time.time(),
            'file_size': dest_path.stat().st_size
        }
        
        metadata_file = self.output_dir / "metadata" / f"{dest_path.stem}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_synthetic_images(self, count: int = 10):
        """Gera imagens sintéticas para complementar o dataset."""
        print(f"🎨 Gerando {count} imagens sintéticas...")
        
        for i in range(count):
            img = self._create_synthetic_coffee_scene()
            filename = f"synthetic_coffee_scene_{i:03d}.jpg"
            filepath = self.output_dir / "synthetic_generated" / filename
            
            cv2.imwrite(str(filepath), img)
            
            # Metadados para sintéticas
            metadata = {
                'filename': filename,
                'type': 'synthetic',
                'generation_time': time.time(),
                'parameters': 'realistic_coffee_scene_with_weeds'
            }
            
            metadata_file = self.output_dir / "metadata" / f"synthetic_{i:03d}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ {count} imagens sintéticas geradas")
    
    def _create_synthetic_coffee_scene(self) -> np.ndarray:
        """Cria cena sintética realista de cafezal com ervas."""
        width = np.random.randint(640, 1024)
        height = np.random.randint(480, 768)
        
        # Base da imagem (solo)
        img = np.ones((height, width, 3), dtype=np.uint8) * np.random.randint(80, 120)
        
        # Adicionar textura do solo
        noise = np.random.randint(-30, 30, (height, width, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Fileiras de café (organizadas)
        row_spacing = np.random.randint(60, 100)
        plant_spacing = np.random.randint(40, 70)
        
        for row in range(0, height, row_spacing):
            for x in range(0, width, plant_spacing):
                if np.random.random() > 0.1:  # 90% de chance de ter planta
                    # Café: verde escuro, circular
                    coffee_color = (0, np.random.randint(60, 100), np.random.randint(10, 40))
                    radius = np.random.randint(12, 20)
                    cv2.circle(img, (x + plant_spacing//2, row + 30), radius, coffee_color, -1)
        
        # Ervas daninhas (espalhadas irregularmente)
        num_weeds = np.random.randint(10, 25)
        for _ in range(num_weeds):
            x = np.random.randint(20, width-20)
            y = np.random.randint(20, height-20)
            
            # Ervas: verde mais claro, formato irregular
            weed_color = (np.random.randint(80, 150), 
                         np.random.randint(150, 255), 
                         np.random.randint(30, 100))
            
            # Formato irregular
            size = np.random.randint(8, 18)
            pts = []
            for angle in range(0, 360, 45):
                r = size + np.random.randint(-5, 5)
                px = x + int(r * np.cos(np.radians(angle)))
                py = y + int(r * np.sin(np.radians(angle)))
                pts.append([px, py])
            
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], weed_color)
        
        # Adicionar sombras e variações de iluminação
        overlay = np.zeros_like(img)
        cv2.ellipse(overlay, (width//2, height//2), (width//3, height//4), 0, 0, 360, (20, 20, 20), -1)
        img = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)
        
        return img
    
    def download_all_categories(self, images_per_category: int = 15):
        """Download de imagens de todas as categorias."""
        search_terms = self.get_search_terms()
        total_downloaded = 0
        
        print("🚀 Iniciando download massivo de imagens de treinamento...")
        print(f"📊 Meta: {images_per_category} imagens por categoria")
        print("=" * 60)
        
        for category, terms in search_terms.items():
            print(f"\n🔍 CATEGORIA: {category.upper()}")
            category_total = 0
            
            for term in terms:
                self.stats['total_searched'] += 1
                
                # Download com Bing
                downloaded = self.download_with_bing(term, category, limit=images_per_category//len(terms) + 2)
                category_total += downloaded
                
                # Complementar com Unsplash se necessário
                if downloaded < 3:
                    downloaded += self.download_with_unsplash(term, category, limit=5)
                    category_total += downloaded
                
                time.sleep(2)  # Rate limiting
                
                if category_total >= images_per_category:
                    break
            
            total_downloaded += category_total
            print(f"✅ {category}: {category_total} imagens baixadas")
        
        self.stats['downloaded'] = total_downloaded
        
        # Gerar imagens sintéticas como complemento
        self.generate_synthetic_images(count=20)
        
        return total_downloaded
    
    def create_training_list(self) -> List[str]:
        """Cria lista de imagens prontas para treinamento."""
        training_images = []
        
        # Coletar de todas as categorias
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir() and category_dir.name != 'metadata':
                for img_file in category_dir.glob("*.jpg"):
                    training_images.append(str(img_file))
        
        # Salvar lista
        list_file = self.output_dir / "training_image_list.txt"
        with open(list_file, 'w') as f:
            for img_path in training_images:
                f.write(f"{img_path}\n")
        
        print(f"📝 Lista de treinamento criada: {len(training_images)} imagens")
        print(f"📁 Arquivo: {list_file}")
        
        return training_images
    
    def print_summary(self):
        """Imprime resumo do download."""
        print("\n" + "=" * 60)
        print("📊 RESUMO DO DOWNLOAD")
        print("=" * 60)
        print(f"🔍 Buscas realizadas: {self.stats['total_searched']}")
        print(f"✅ Imagens baixadas: {self.stats['downloaded']}")
        print(f"🚫 Filtradas (baixa qualidade): {self.stats['filtered_out']}")
        print(f"❌ Erros: {self.stats['errors']}")
        
        # Contar por categoria
        print("\n📁 Por categoria:")
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir() and category_dir.name != 'metadata':
                count = len(list(category_dir.glob("*.jpg")))
                print(f"  {category_dir.name}: {count} imagens")
        
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*.jpg"))
        print(f"\n💾 Espaço total usado: {total_size / (1024*1024):.1f} MB")
        print(f"📂 Diretório: {self.output_dir}")


def main():
    """Função principal para download de imagens de treinamento."""
    print("🤖 DOWNLOAD AUTOMÁTICO DE IMAGENS PARA TREINAMENTO ML")
    print("Detecção de Ervas Daninhas em Cafezais")
    print("=" * 60)
    
    # Instalar dependências
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Erro instalando dependências: {e}")
        return
    
    # Criar downloader
    downloader = TrainingImageDownloader("training_dataset")
    
    # Configuração
    images_per_category = 20  # Ajuste conforme necessário
    
    try:
        # Download massivo
        total = downloader.download_all_categories(images_per_category)
        
        # Criar lista de treinamento
        training_list = downloader.create_training_list()
        
        # Resumo
        downloader.print_summary()
        
        if total > 0:
            print(f"\n🎉 SUCESSO! {total} imagens prontas para treinamento")
            print("\n🚀 Próximos passos:")
            print("1. Revisar imagens baixadas")
            print("2. Remover imagens inadequadas manualmente")
            print("3. Executar treinamento ML:")
            print(f"   python -c \"from app.services.processing.ml_training import *; MLTrainingPipeline().train_models_from_images({training_list[:10]})\"")
        else:
            print("❌ Nenhuma imagem foi baixada. Verifique conexão e termos de busca.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Download interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante o download: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()