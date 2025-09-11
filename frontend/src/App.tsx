import { useState } from 'react'
import Upload from './components/Upload'
import MapView from './components/MapView'

export default function App(){
  const [resultUrl, setResultUrl] = useState<string>('')
  const [notes, setNotes] = useState<string>('')

  return (
    <div className="card">
      <h1>☕ Café Mapper — Frontend</h1>
      <p className="muted">Envie uma foto aérea (RGB). O backend fará uma análise demo e retorna uma imagem anotada.</p>
      <Upload onResult={(url, n)=>{ setResultUrl(url); setNotes(n || '') }} />

      {resultUrl && (
        <div style={{marginTop:'1rem'}}>
          <h3>Resultado</h3>
          <img className="result" src={resultUrl} alt="resultado" />
          {notes && <p className="muted">{notes}</p>}
        </div>
      )}

      <div style={{marginTop:'1rem'}}>
        <h3>Mapa (base)</h3>
        <MapView />
      </div>
    </div>
  )
}
