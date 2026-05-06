import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import { useEffect } from 'react'

const zoneColors = {
  bajo: '#15803d',
  medio: '#ca8a04',
  alto: '#dc2626',
}

const clusterColors = ['#ef4444', '#f59e0b', '#2563eb', '#7c3aed', '#0891b2', '#db2777']

function FitRoute({ route }) {
  const map = useMap()

  useEffect(() => {
    if (route.length < 2) return
    map.fitBounds(route, { padding: [36, 36] })
  }, [map, route])

  return null
}

function MapCenterUpdater({ center }) {
  const map = useMap()

  useEffect(() => {
    if (!center) return
    map.setView(center, map.getZoom(), { animate: true })
  }, [map, center])

  return null
}

function MapClickPicker({ selectionMode, onPick }) {
  useMapEvents({
    click(event) {
      if (!selectionMode) return
      onPick(selectionMode, event.latlng)
    },
  })

  return null
}

function MapView({
  mapCenter,
  selectionMode,
  onPick,
  origin,
  destination,
  riskZones,
  crimePoints,
  safeRoutePositions,
  traditionalRoutePositions,
}) {
  return (
    <div className="map-wrapper" aria-label="Mapa de rutas seguras">
      <MapContainer center={mapCenter} zoom={13} className="map">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapCenterUpdater center={mapCenter} />

        <MapClickPicker selectionMode={selectionMode} onPick={onPick} />

        {origin && (
          <CircleMarker center={origin} radius={9} pathOptions={{ color: '#0f766e' }}>
            <Popup>Origen</Popup>
          </CircleMarker>
        )}
        {destination && (
          <CircleMarker center={destination} radius={9} pathOptions={{ color: '#2563eb' }}>
            <Popup>Destino</Popup>
          </CircleMarker>
        )}

        {riskZones.map((zone) => (
          <CircleMarker
            key={zone.cluster}
            center={[zone.center.lat, zone.center.lng]}
            radius={Math.max(12, Math.min(36, zone.radius_m / 80))}
            pathOptions={{
              color: zoneColors[zone.risk_level],
              fillColor: zoneColors[zone.risk_level],
              fillOpacity: 0.22,
            }}
          >
            <Popup>
              Cluster {zone.cluster} · riesgo {zone.risk_level}
              <br />
              Delitos: {zone.total_crimes}
            </Popup>
          </CircleMarker>
        ))}

        {crimePoints.map((point) => (
          <CircleMarker
            key={point.id}
            center={[point.location.lat, point.location.lng]}
            radius={3}
            pathOptions={{
              color: clusterColors[point.cluster % clusterColors.length],
              fillColor: clusterColors[point.cluster % clusterColors.length],
              fillOpacity: 0.72,
              opacity: 0.9,
              weight: 1,
            }}
          >
            <Popup>
              Delito #{point.id}
              <br />
              Cluster {point.cluster}
              <br />
              {point.distrito}
              <br />
              {point.tipo} - {point.subtipo}
              <br />
              Turno: {point.turno}
            </Popup>
          </CircleMarker>
        ))}

        {traditionalRoutePositions.length > 1 && (
          <Polyline
            positions={traditionalRoutePositions}
            pathOptions={{ color: '#4b5563', weight: 4, opacity: 0.7 }}
          />
        )}

        {safeRoutePositions.length > 1 && (
          <>
            <Polyline
              positions={safeRoutePositions}
              pathOptions={{ color: '#22c55e', weight: 6, opacity: 0.9 }}
            />
            <FitRoute route={safeRoutePositions} />
          </>
        )}
      </MapContainer>
      <div className="map-search">
        <input type="text" placeholder="Buscar lugar..." />
      </div>
      <div className="map-legend">
        <p>Mapa de riesgo (predicción)</p>
        <div className="legend-bar"></div>
        <span>Bajo</span>
        <span>Alto</span>
      </div>
    </div>
  )
}

export default MapView
