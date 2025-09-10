#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de estrutura — compatível com o monorepo `pdi-cafezais-skeleton`.

- Não sobrescreve arquivos existentes
- Cria pastas do Coffee PDI System mapeadas para `backend/app/...`
- Pode rodar com: `python scripts/project_generator.py --base-path .`
"""
import argparse, os
from pathlib import Path

MAP = {
    # app principal (mapeado para backend/app)
    "app/core": "backend/app/core_extra",
    "app/models": "backend/app/models_extra",
    "app/modules/acquisition": "backend/app/modules/acquisition",
    "app/modules/preprocessing": "backend/app/modules/preprocessing",
    "app/modules/analysis": "backend/app/modules/analysis",
    "app/modules/disease_detection": "backend/app/modules/disease_detection",
    "app/modules/reporting/templates": "backend/app/modules/reporting/templates",
    "app/modules/visualization": "backend/app/modules/visualization",
    "app/services": "backend/app/services",
    "app/api/v1/endpoints": "backend/app/api/v1/endpoints",
    "app/utils": "backend/app/utils",
    # tests (mapeado para backend/tests)
    "tests/unit": "backend/tests/unit",
    "tests/integration": "backend/tests/integration",
    "tests/performance": "backend/tests/performance",
    "tests/fixtures/images": "backend/tests/fixtures/images",
    "tests/fixtures/data": "backend/tests/fixtures/data",
    # data, notebooks, docs, ml_models
    "ml_models/weights": "ml_models/weights",
    "ml_models/configs": "ml_models/configs",
    "data/raw": "data/raw",
    "data/processed": "data/processed",
    "data/cache": "data/cache",
    "data/uploads": "data/uploads",
    "notebooks": "notebooks",
    "docs/source": "docs/source",
    "docs/images": "docs/images",
}

STUB_FILES = {
    "backend/app/api/v1/endpoints/__init__.py": "",
    "backend/app/services/__init__.py": "",
    "backend/app/utils/__init__.py": "",
    "backend/tests/unit/__init__.py": "",
    "backend/tests/integration/__init__.py": "",
    "backend/tests/performance/__init__.py": "",
}

def safe_touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.touch()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=".", help="Raiz do repositório")
    args = ap.parse_args()
    base = Path(args.base_path).resolve()
    print(f"📦 Gerando estrutura no monorepo: {base}")

    # Criar diretórios
    for src, dst in MAP.items():
        target = base / dst
        target.mkdir(parents=True, exist_ok=True)
        # .gitkeep para pastas vazias relevantes
        if any(seg in ["raw", "processed", "cache", "uploads", "weights"] for seg in target.parts):
            safe_touch(target / ".gitkeep")
        print(f"📁 {dst}")

    # Criar arquivos stub
    for fp, content in STUB_FILES.items():
        safe_touch(base / fp)
        if content:
            (base / fp).write_text(content, encoding="utf-8")

    # Exemplo de template extra (README para modules)
    readme_modules = base / "backend/app/modules/README.md"
    if not readme_modules.exists():
        readme_modules.write_text(
            "# Modules\n\n- acquisition\n- preprocessing\n- analysis\n- disease_detection\n- reporting\n- visualization\n",
            encoding="utf-8",
        )

    print("✅ Estrutura pronta!")

if __name__ == "__main__":
    main()
