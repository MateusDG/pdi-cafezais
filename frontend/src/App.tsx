import { useEffect } from 'react'
import 'leaflet/dist/leaflet.css'

function App() {
  useEffect(() => {
    // Lazy import Leaflet to avoid SSR issues
    (async () => {
      const L = await import('leaflet')
      const map = L.map('map').setView([-19.9, -43.9], 5)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap'
      }).addTo(map)
      L.marker([-19.9, -43.9]).addTo(map).bindPopup('PDI Cafezais')
    })()
  }, [])

  return (
    <div style={{height: '100vh', width: '100vw'}}>
      <header style={{padding: 12, fontWeight: 600}}>PDI Cafezais</header>
      <div id="map" style={{height: 'calc(100vh - 48px)'}} />
    </div>
  )
}
export default App
