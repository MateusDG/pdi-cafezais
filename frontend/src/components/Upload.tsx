import { useRef, useState, useCallback } from 'react'
import { uploadImage, ProcessResult } from '../api'

type Props = { 
  onResult: (result: ProcessResult) => void 
  onError: (error: string) => void
}

export default function Upload({ onResult, onError }: Props){
  const inputRef = useRef<HTMLInputElement|null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [sensitivity, setSensitivity] = useState(0.5)

  const validateFile = (file: File): string | null => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']
    const maxSize = 50 * 1024 * 1024 // 50MB
    
    if (!validTypes.includes(file.type)) {
      return 'Formato não suportado. Use JPG, PNG, BMP ou TIFF.'
    }
    
    if (file.size > maxSize) {
      return 'Arquivo muito grande. Máximo 50MB.'
    }
    
    return null
  }

  const processFile = async (file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      onError(validationError)
      return
    }

    setBusy(true)
    setProgress(0)
    setSelectedFile(file)

    // Simulate progress during upload
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev
        return prev + Math.random() * 10
      })
    }, 200)

    try {
      const result = await uploadImage(file, sensitivity)
      setProgress(100)
      setTimeout(() => {
        onResult(result)
        setProgress(0)
      }, 300)
    } catch(err: any) {
      const errorMessage = err?.message ?? 'Erro desconhecido durante o processamento'
      console.error('Upload error:', err)
      onError(errorMessage)
      setProgress(0)
    } finally {
      clearInterval(progressInterval)
      setBusy(false)
    }
  }

  async function onSubmit(e: React.FormEvent){
    e.preventDefault()
    const file = selectedFile || inputRef.current?.files?.[0]
    if(!file) return
    await processFile(file)
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    const imageFile = files.find(file => file.type.startsWith('image/'))
    
    if (imageFile) {
      setSelectedFile(imageFile)
      if (inputRef.current) {
        const dataTransfer = new DataTransfer()
        dataTransfer.items.add(imageFile)
        inputRef.current.files = dataTransfer.files
      }
      await processFile(imageFile)
    } else {
      onError('Por favor, arraste apenas arquivos de imagem.')
    }
  }, [onError])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  return (
    <div className="upload-section">
      <form onSubmit={onSubmit} className="upload-form">
        {/* Drag & Drop Area */}
        <div 
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          {selectedFile ? (
            <div className="file-preview">
              <div className="file-icon">📸</div>
              <div className="file-info">
                <div className="file-name">{selectedFile.name}</div>
                <div className="file-size">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
              {!busy && (
                <button 
                  type="button" 
                  className="remove-file"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedFile(null)
                    if (inputRef.current) inputRef.current.value = ''
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          ) : (
            <div className="drop-zone-content">
              <div className="drop-zone-icon">📤</div>
              <div className="drop-zone-text">
                <strong>Clique aqui ou arraste uma imagem</strong>
                <div className="drop-zone-hint">
                  Formatos: JPG, PNG, BMP, TIFF • Máximo: 50MB
                </div>
              </div>
            </div>
          )}
          
          <input 
            ref={inputRef} 
            type="file" 
            accept=".jpg,.jpeg,.png,.bmp,.tiff,image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        {/* Progress Bar */}
        {busy && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="progress-text">
              {progress < 100 ? `Enviando... ${progress.toFixed(0)}%` : 'Processando...'}
            </div>
          </div>
        )}
        
        {/* Sensitivity Control */}
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

        {/* Submit Button */}
        <button 
          className="btn btn-primary" 
          type="submit" 
          disabled={busy || !selectedFile}
        >
          {busy ? '🔄 Analisando...' : '🔍 Analisar Cafezal'}
        </button>
      </form>
    </div>
  )
}
