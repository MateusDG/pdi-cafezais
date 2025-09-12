import { useState } from 'react'
import Upload from './components/Upload'
import MapView from './components/MapView'
import AnalysisResult from './components/AnalysisResult'
import { ProcessResult } from './api'

export default function App(){
  const [result, setResult] = useState<ProcessResult | null>(null)
  const [error, setError] = useState<string>('')

  const handleResult = (newResult: ProcessResult) => {
    setResult(newResult)
    setError('')
  }

  const handleError = (errorMessage: string) => {
    setError(errorMessage)
    setResult(null)
  }

  const clearResults = () => {
    setResult(null)
    setError('')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>☕ Café Mapper</h1>
        <p className="app-description">
          Sistema de análise de cafezais por imagem. Faça upload de uma foto aérea (RGB) 
          para detectar ervas daninhas usando visão computacional.
        </p>
      </header>

      <main className="app-main">
        <section className="upload-section-wrapper">
          <Upload onResult={handleResult} onError={handleError} />
          
          {error && (
            <div className="error-message">
              <h3>❌ Erro no Processamento</h3>
              <p>{error}</p>
              <button className="btn btn-secondary" onClick={clearResults}>
                Tentar Novamente
              </button>
            </div>
          )}
        </section>

        {result && (
          <section className="results-section">
            <AnalysisResult result={result} />
            
            <div className="action-buttons">
              <button className="btn btn-secondary" onClick={clearResults}>
                🔄 Nova Análise
              </button>
            </div>
          </section>
        )}

        <section className="map-section">
          <div className="section-header">
            <h3>🗺️ Mapa de Referência</h3>
            <p className="section-description">
              Visualização geográfica básica. Em versões futuras, os resultados 
              da análise serão plotados diretamente no mapa.
            </p>
          </div>
          <MapView />
        </section>
      </main>

      <footer className="app-footer">
        <p>
          💡 <strong>Dica:</strong> Para melhores resultados, use imagens com boa iluminação, 
          capturadas entre 10h-14h, altitude de 30-50m, e evite sombras excessivas.
        </p>
      </footer>
    </div>
  )
}
