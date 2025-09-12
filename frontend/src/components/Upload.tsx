import { useRef, useState } from 'react'
import { uploadImage, ProcessResult } from '../api'

type Props = { 
  onResult: (result: ProcessResult) => void 
  onError: (error: string) => void
}

export default function Upload({ onResult, onError }: Props){
  const inputRef = useRef<HTMLInputElement|null>(null)
  const [busy, setBusy] = useState(false)
  const [sensitivity, setSensitivity] = useState(0.5)

  async function onSubmit(e: React.FormEvent){
    e.preventDefault()
    const file = inputRef.current?.files?.[0]
    if(!file) return

    setBusy(true)
    try{
      const result = await uploadImage(file, sensitivity)
      onResult(result)
    }catch(err: any){
      const errorMessage = err?.message ?? 'Erro desconhecido durante o processamento'
      console.error('Upload error:', err)
      onError(errorMessage)
    }finally{
      setBusy(false)
    }
  }

  return (
    <div className="upload-section">
      <form onSubmit={onSubmit} className="upload-form">
        <div className="file-input-group">
          <label htmlFor="file-input">📸 Selecione uma imagem do cafezal:</label>
          <input 
            id="file-input"
            ref={inputRef} 
            type="file" 
            accept=".jpg,.jpeg,.png,.bmp,.tiff" 
            required 
          />
        </div>
        
        <div className="sensitivity-group">
          <label htmlFor="sensitivity-input">🎚️ Sensibilidade de detecção:</label>
          <input
            id="sensitivity-input"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={sensitivity}
            onChange={(e) => setSensitivity(parseFloat(e.target.value))}
            disabled={busy}
          />
          <span className="sensitivity-value">
            {(sensitivity * 100).toFixed(0)}% 
            {sensitivity <= 0.3 && " (Menos sensível)"}
            {sensitivity >= 0.7 && " (Mais sensível)"}
          </span>
        </div>

        <button 
          className="btn btn-primary" 
          type="submit" 
          disabled={busy}
        >
          {busy ? '🔄 Analisando...' : '🔍 Analisar Cafezal'}
        </button>
      </form>

      {busy && (
        <div className="processing-info">
          <p>⏳ Processando imagem... Isso pode levar alguns segundos.</p>
        </div>
      )}
    </div>
  )
}
