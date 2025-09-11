import { MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

export default function MapView(){
  return (
    <MapContainer id="map" center={[-20.0, -40.0]} zoom={13} scrollWheelZoom={true} style={{height: '480px', width:'100%'}}>
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
    </MapContainer>
  )
}
