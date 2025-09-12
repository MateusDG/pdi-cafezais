import { ProcessResult } from '../api'

interface Props {
  result: ProcessResult
}

export default function AnalysisResult({ result }: Props) {
  const { summary, result_image_url, analysis_notes, processing_parameters } = result

  return (
    <div className="analysis-result">
      <div className="result-header">
        <h3>📊 Resultado da Análise</h3>
        <div className="processing-info">
          <span className="algorithm-badge">
            {processing_parameters.algorithm.replace('_', ' ')} v{processing_parameters.version}
          </span>
          <span className="time-badge">
            ⏱️ {summary.processing_time_seconds}s
          </span>
        </div>
      </div>

      <div className="result-content">
        <div className="result-image-section">
          <h4>🖼️ Imagem Processada</h4>
          <div className="image-container">
            <img 
              src={result_image_url} 
              alt="Resultado da análise" 
              className="result-image"
              loading="lazy"
            />
          </div>
        </div>

        <div className="analysis-summary">
          <h4>🌿 Resumo da Detecção</h4>
          
          <div className="summary-cards">
            <div className="summary-card">
              <div className="card-icon">🚨</div>
              <div className="card-content">
                <div className="card-title">Áreas Detectadas</div>
                <div className="card-value">{summary.weed_detection.areas_detected}</div>
              </div>
            </div>

            <div className="summary-card">
              <div className="card-icon">📏</div>
              <div className="card-content">
                <div className="card-title">Cobertura de Ervas</div>
                <div className="card-value">
                  {summary.weed_detection.weed_coverage_percentage.toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="summary-card">
              <div className="card-icon">🎯</div>
              <div className="card-content">
                <div className="card-title">Sensibilidade Usada</div>
                <div className="card-value">
                  {(summary.weed_detection.detection_sensitivity * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            <div className="summary-card">
              <div className="card-icon">🖼️</div>
              <div className="card-content">
                <div className="card-title">Dimensões</div>
                <div className="card-value">
                  {summary.image_stats.width} × {summary.image_stats.height}
                </div>
              </div>
            </div>
          </div>

          <div className="detailed-stats">
            <h5>📋 Estatísticas Detalhadas</h5>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Área total analisada:</span>
                <span className="stat-value">
                  {summary.image_stats.total_pixels.toLocaleString()} pixels
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Área com ervas daninhas:</span>
                <span className="stat-value">
                  {summary.weed_detection.total_weed_area_pixels.toLocaleString()} pixels
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Tamanho do arquivo:</span>
                <span className="stat-value">
                  {summary.image_stats.file_size_mb.toFixed(1)} MB
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Status da análise:</span>
                <span className="stat-value status-completed">
                  {summary.analysis_status === 'completed' ? '✅ Concluída' : summary.analysis_status}
                </span>
              </div>
              {summary.scale_factor && summary.scale_factor < 1 && (
                <div className="stat-item">
                  <span className="stat-label">Imagem redimensionada:</span>
                  <span className="stat-value">
                    {(summary.scale_factor * 100).toFixed(0)}% do tamanho original
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="analysis-notes">
            <h5>📝 Observações</h5>
            <p>{analysis_notes}</p>
          </div>

          {summary.detected_issues.length > 0 && (
            <div className="issues-section">
              <h5>⚠️ Problemas Detectados</h5>
              <ul className="issues-list">
                {summary.detected_issues.map((issue, index) => (
                  <li key={index} className="issue-item">{issue}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="result-actions">
        <button className="btn btn-secondary" onClick={() => window.open(result_image_url, '_blank')}>
          🔍 Ver Imagem em Tela Cheia
        </button>
      </div>
    </div>
  )
}