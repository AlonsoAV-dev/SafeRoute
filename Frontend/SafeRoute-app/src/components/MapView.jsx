import {
  Circle,
  MapContainer,
  Marker,
  Polyline,
  TileLayer,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet.heat'
import { useEffect, useMemo, useState } from 'react'
import { Moon, Sun, X } from 'lucide-react'

const zoneColors = {
  bajo: '#22c55e',
  medio: '#f59e0b',
  alto: '#ef4444',
}

const darkTiles = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
const lightTiles = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'

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

function MapSizeFixer() {
  const map = useMap()

  useEffect(() => {
    const refresh = () => {
      map.invalidateSize()
    }

    refresh()
    const timer = window.setTimeout(refresh, 240)
    window.addEventListener('resize', refresh)

    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('resize', refresh)
    }
  }, [map])

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

function HeatLayer({ points }) {
  const map = useMap()

  useEffect(() => {
    if (!points?.length) return undefined
    if (map.getSize().y === 0 || map.getSize().x === 0) return undefined
    const layer = L.heatLayer(points, {
      radius: 22,
      blur: 18,
      maxZoom: 18,
      minOpacity: 0.35,
      gradient: {
        0.0: '#22c55e',
        0.3: '#84cc16',
        0.5: '#f59e0b',
        0.7: '#f97316',
        1.0: '#ef4444',
      },
    }).addTo(map)

    return () => {
      map.removeLayer(layer)
    }
  }, [map, points])

  return null
}

function MapView({
  mapCenter,
  selectionMode,
  onPick,
  origin,
  destination,
  riskZones,
  heatmapPoints,
  safeRoutePositions,
  traditionalRoutePositions,
}) {
  const [searchValue, setSearchValue] = useState('')
  const [isDark, setIsDark] = useState(false)

  const originIcon = useMemo(
    () =>
      L.divIcon({
        className: 'pin-icon pin-icon--green',
        html: '<div class="pin-inner"></div>',
        iconSize: [24, 24],
        iconAnchor: [12, 24],
      }),
    [],
  )

  const destinationIcon = useMemo(
    () =>
      L.divIcon({
        className: 'pin-icon pin-icon--red',
        html: '<div class="pin-inner"></div>',
        iconSize: [24, 24],
        iconAnchor: [12, 24],
      }),
    [],
  )

  return (
    <div className="map-wrapper" aria-label="Mapa de rutas seguras">
      <MapContainer center={mapCenter} zoom={13} className="map" zoomControl={false}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url={isDark ? darkTiles : lightTiles}
        />

        <MapCenterUpdater center={mapCenter} />

        <MapSizeFixer />

        <MapClickPicker selectionMode={selectionMode} onPick={onPick} />

        <HeatLayer points={heatmapPoints} />

        {origin && <Marker position={origin} icon={originIcon} />}
        {destination && <Marker position={destination} icon={destinationIcon} />}

        {riskZones.map((zone, index) => (
          <Circle
            key={`${zone.center[0]}-${zone.center[1]}-${index}`}
            center={zone.center}
            radius={Math.max(35, Math.min(140, zone.radius * 0.9))}
            pathOptions={{
              color: zoneColors[zone.risk_level],
              fillColor: zoneColors[zone.risk_level],
              fillOpacity: 0.22,
            }}
          />
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
              pathOptions={{ color: '#22c55e', weight: 5, opacity: 0.9 }}
            />
            <FitRoute route={safeRoutePositions} />
          </>
        )}
      </MapContainer>
      <div className="map-search">
        <span>Buscar lugar...</span>
        <div className="map-search-field">
          <input
            type="text"
            placeholder="Buscar lugar..."
            value={searchValue}
            onChange={(event) => setSearchValue(event.target.value)}
          />
          <button type="button" onClick={() => setSearchValue('')}>
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="map-legend">
        <p>Mapa de riesgo (predicción)</p>
        <div className="legend-bar"></div>
        <div className="legend-labels">
          <span>Bajo</span>
          <span>Alto</span>
        </div>
        <small>Predicción: Random Forest</small>
      </div>
      <button
        type="button"
        className="map-toggle"
        onClick={() => setIsDark((current) => !current)}
        aria-label="Alternar modo del mapa"
      >
        {isDark ? <Sun size={16} /> : <Moon size={16} />}
      </button>
    </div>
  )
}

export default MapView
