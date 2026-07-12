import {
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
import {
  BrainCircuit,
  Crosshair,
  Filter,
  History,
  Layers3,
  Moon,
  RotateCcw,
  Sun,
  X,
} from 'lucide-react'

const darkTiles = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
const lightTiles = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const XGBOOST_HEAT_MAX_ZOOM = 13

function FitRoute({ route }) {
  const map = useMap()
  useEffect(() => {
    if (route.length > 1) map.fitBounds(route, { padding: [42, 42] })
  }, [map, route])
  return null
}

function MapCenterUpdater({ center }) {
  const map = useMap()
  useEffect(() => {
    if (center) map.setView(center, map.getZoom(), { animate: true })
  }, [map, center])
  return null
}

function MapSizeFixer() {
  const map = useMap()
  useEffect(() => {
    const refresh = () => map.invalidateSize()
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
      if (selectionMode) onPick(selectionMode, event.latlng)
    },
  })
  return null
}

function HeatLayer({ points, variant = 'historical' }) {
  const map = useMap()
  useEffect(() => {
    if (!points?.length || map.getSize().y === 0 || map.getSize().x === 0) return undefined
    const gradient =
      variant === 'predicted'
        ? {
            0.08: '#fef3c7',
            0.35: '#facc15',
            0.65: '#f97316',
            1: '#b91c1c',
          }
        : {
            0.05: '#22c55e',
            0.25: '#84cc16',
            0.4: '#facc15',
            0.55: '#f97316',
            0.68: '#ef4444',
            0.82: '#dc2626',
            1: '#7f1d1d',
          }
    const layer = L.heatLayer(points, {
      radius: variant === 'predicted' ? 20 : 17,
      blur: variant === 'predicted' ? 16 : 8,
      maxZoom: variant === 'predicted' ? XGBOOST_HEAT_MAX_ZOOM : 17,
      minOpacity: 0.12,
      max: variant === 'predicted' ? 1.2 : 1,
      gradient,
    }).addTo(map)
    return () => map.removeLayer(layer)
  }, [map, points, variant])
  return null
}

function CrimeLayer({ points }) {
  const map = useMap()
  useEffect(() => {
    if (!points?.length) return undefined
    const renderer = L.canvas({ padding: 0.4 })
    const group = L.layerGroup().addTo(map)
    points.forEach((point) => {
      const severe = point.peso_delito >= 4
      L.circleMarker([point.lat, point.lng], {
        renderer,
        radius: severe ? 4 : 2.6,
        color: severe ? '#581c87' : '#7e22ce',
        fillColor: severe ? '#a855f7' : '#c084fc',
        weight: severe ? 1.2 : 0.5,
        fillOpacity: severe ? 0.82 : 0.52,
      })
        .bindTooltip(
          `<strong>${point.modalidad}</strong><br>Peso: ${point.peso_delito}/5<br>${point.distrito}`,
          { direction: 'top' },
        )
        .addTo(group)
    })
    return () => map.removeLayer(group)
  }, [map, points])
  return null
}

function PredictionLayer({ points }) {
  const map = useMap()
  useEffect(() => {
    if (!points?.length) return undefined
    const renderer = L.canvas({ padding: 0.4 })
    const group = L.layerGroup().addTo(map)
    points.forEach((point) => {
      const score = Number(point.risk_score)
      const color = score >= 0.66 ? '#dc2626' : score >= 0.34 ? '#f59e0b' : '#16a34a'
      L.circleMarker([point.lat, point.lng], {
        renderer,
        radius: score >= 0.66 ? 4.2 : 3,
        color: '#ffffff',
        fillColor: color,
        weight: 0.8,
        fillOpacity: 0.78,
      })
        .bindTooltip(
          `<strong>Predicci&oacute;n XGBoost</strong><br>Riesgo: ${(score * 100).toFixed(1)}% (${point.risk_level})<br>Tramo: ${point.tramo_id}`,
          { direction: 'top' },
        )
        .addTo(group)
    })
    return () => map.removeLayer(group)
  }, [map, points])
  return null
}

function PredictionVisualization({ heatmapPoints, predictionPoints, enabled }) {
  const map = useMap()
  const [zoom, setZoom] = useState(map.getZoom())
  const [viewport, setViewport] = useState(() => {
    const bounds = map.getBounds()
    return {
      south: bounds.getSouth(),
      west: bounds.getWest(),
      north: bounds.getNorth(),
      east: bounds.getEast(),
    }
  })
  const updateViewport = () => {
    const bounds = map.getBounds()
    setViewport({
      south: bounds.getSouth(),
      west: bounds.getWest(),
      north: bounds.getNorth(),
      east: bounds.getEast(),
    })
  }
  useMapEvents({
    zoomend() {
      setZoom(map.getZoom())
      updateViewport()
    },
    moveend() {
      updateViewport()
    },
  })
  const visiblePredictionPoints = useMemo(
    () =>
      predictionPoints.filter(
        (point) =>
          point.lat >= viewport.south &&
          point.lat <= viewport.north &&
          point.lng >= viewport.west &&
          point.lng <= viewport.east,
      ),
    [predictionPoints, viewport],
  )

  if (!enabled) return null
  return zoom <= XGBOOST_HEAT_MAX_ZOOM ? (
    <HeatLayer points={heatmapPoints} variant="predicted" />
  ) : (
    <PredictionLayer points={visiblePredictionPoints} />
  )
}

function FilterSelect({ label, name, value, options, onChange }) {
  return (
    <label className="map-filter-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(name, event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option === 'todos' ? 'Todos' : option}
          </option>
        ))}
      </select>
    </label>
  )
}

function MapView({
  mapCenter,
  selectionMode,
  onPick,
  origin,
  destination,
  heatmapPoints,
  crimePoints,
  predictionHeatmapPoints,
  predictionPoints,
  crimeTotal,
  predictionTotal,
  crimeFilters,
  filterOptions,
  mapLayers,
  mapDataLoading,
  onFilterChange,
  onLayerChange,
  onResetFilters,
  safeRoutePositions,
  traditionalRoutePositions,
}) {
  const [isDark, setIsDark] = useState(false)
  const [filtersOpen, setFiltersOpen] = useState(false)
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
    <div className={`map-wrapper ${selectionMode ? 'is-selecting' : ''}`}>
      <MapContainer
        center={mapCenter}
        zoom={13}
        className="map"
        zoomControl={false}
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url={isDark ? darkTiles : lightTiles}
        />
        <MapCenterUpdater center={mapCenter} />
        <MapSizeFixer />
        <MapClickPicker selectionMode={selectionMode} onPick={onPick} />
        <HeatLayer points={heatmapPoints} variant="historical" />
        <CrimeLayer points={crimePoints} />
        <PredictionVisualization
          heatmapPoints={predictionHeatmapPoints}
          predictionPoints={predictionPoints}
          enabled={mapLayers.predictionHeatmap}
        />

        {origin && <Marker position={origin} icon={originIcon} />}
        {destination && <Marker position={destination} icon={destinationIcon} />}
        {traditionalRoutePositions.length > 1 && (
          <>
            <Polyline
              positions={traditionalRoutePositions}
              pathOptions={{ color: '#ffffff', weight: 7, opacity: 0.85 }}
            />
            <Polyline
              positions={traditionalRoutePositions}
              pathOptions={{
                color: '#334155',
                weight: 4,
                opacity: 0.95,
                dashArray: '9 7',
              }}
            />
          </>
        )}
        {safeRoutePositions.length > 1 && (
          <>
            <Polyline
              positions={safeRoutePositions}
              pathOptions={{ color: '#ffffff', weight: 8, opacity: 0.92 }}
            />
            <Polyline
              positions={safeRoutePositions}
              pathOptions={{ color: '#0284c7', weight: 5, opacity: 1 }}
            />
            <FitRoute route={safeRoutePositions} />
          </>
        )}
      </MapContainer>

      {selectionMode && (
        <div className="map-selection-banner">
          <Crosshair size={18} />
          Selecciona el {selectionMode === 'origin' ? 'punto de partida' : 'punto de llegada'}
        </div>
      )}

      <div className="map-controls">
        <button
          type="button"
          className={filtersOpen ? 'map-control-button is-active' : 'map-control-button'}
          onClick={() => setFiltersOpen((current) => !current)}
          aria-label="Mostrar filtros del mapa"
        >
          <Filter size={17} />
        </button>
        <button
          type="button"
          className="map-control-button"
          onClick={() => setIsDark((current) => !current)}
          aria-label="Alternar modo del mapa"
        >
          {isDark ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>

      {filtersOpen && (
        <aside className="map-filter-panel">
          <div className="map-filter-header">
            <div>
              <span>Capas de riesgo</span>
              <small>
                {mapDataLoading
                  ? 'Cargando...'
                  : !mapLayers.heatmap && !mapLayers.predictionHeatmap && !mapLayers.crimes
                    ? 'Todas las capas están desactivadas'
                    : [
                        mapLayers.heatmap
                          ? `Histórico: ${heatmapPoints.length.toLocaleString('es-PE')} zonas`
                          : null,
                        mapLayers.predictionHeatmap
                          ? `XGBoost: ${predictionHeatmapPoints.length.toLocaleString('es-PE')} zonas · ${predictionTotal.toLocaleString('es-PE')} tramos`
                          : null,
                        mapLayers.crimes
                          ? `${crimeTotal.toLocaleString('es-PE')} delitos`
                          : null,
                      ]
                        .filter(Boolean)
                        .join(' · ')}
              </small>
            </div>
            <button type="button" onClick={() => setFiltersOpen(false)} aria-label="Cerrar filtros">
              <X size={16} />
            </button>
          </div>

          <div className="map-layer-options">
            <label className="map-layer-toggle">
              <input
                type="checkbox"
                checked={mapLayers.crimes}
                onChange={(event) => onLayerChange('crimes', event.target.checked)}
              />
              <Layers3 size={14} /> Delitos históricos
            </label>
            <label className="map-layer-toggle map-layer-toggle--historical">
              <input
                type="checkbox"
                checked={mapLayers.heatmap}
                onChange={(event) => onLayerChange('heatmap', event.target.checked)}
              />
              <History size={14} /> Mapa de calor histórico
            </label>
            <label className="map-layer-toggle map-layer-toggle--predicted">
              <input
                type="checkbox"
                checked={mapLayers.predictionHeatmap}
                onChange={(event) =>
                  onLayerChange('predictionHeatmap', event.target.checked)
                }
              />
              <BrainCircuit size={14} /> Mapa predictivo XGBoost
            </label>
          </div>

          <FilterSelect
            label="Día de la semana"
            name="dia_semana"
            value={crimeFilters.dia_semana}
            options={filterOptions.dias_semana}
            onChange={onFilterChange}
          />
          <FilterSelect
            label="Momento del día"
            name="turno"
            value={crimeFilters.turno}
            options={filterOptions.turnos}
            onChange={onFilterChange}
          />
          <FilterSelect
            label="Tipo de delito"
            name="tipo"
            value={crimeFilters.tipo}
            options={filterOptions.tipos}
            onChange={onFilterChange}
          />
          <FilterSelect
            label="Modalidad"
            name="modalidad"
            value={crimeFilters.modalidad}
            options={filterOptions.modalidades}
            onChange={onFilterChange}
          />
          <button type="button" className="reset-filter-button" onClick={onResetFilters}>
            <RotateCcw size={14} /> Restablecer filtros
          </button>
          <p className="filter-note">
            El calor histórico responde a los filtros. XGBoost muestra concentraciones de riesgo
            alto en la vista general y los tramos prioritarios al acercar el mapa.
          </p>
        </aside>
      )}

      {(mapLayers.heatmap || mapLayers.predictionHeatmap) && (
        <div className="heatmap-legends">
          {mapLayers.heatmap && (
            <div className="heatmap-legend">
              <strong>Concentración delictiva histórica</strong>
              <div className="heatmap-gradient heatmap-gradient--historical" />
              <div>
                <span>Menor</span>
                <span>Media</span>
                <span>Mayor</span>
              </div>
            </div>
          )}
          {mapLayers.predictionHeatmap && (
            <div className="heatmap-legend">
              <strong>Riesgo alto predicho por XGBoost</strong>
              <div className="heatmap-gradient heatmap-gradient--predicted" />
              <div>
                <span>Menor concentración</span>
                <span>Mayor concentración</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default MapView
