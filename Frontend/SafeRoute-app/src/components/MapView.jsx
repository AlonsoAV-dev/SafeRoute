import {
  MapContainer,
  Marker,
  Pane,
  Polyline,
  TileLayer,
  Tooltip,
  ZoomControl,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet.heat'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import { ALTERNATIVE_COLOR, RISK_LEVELS, ROUTE_COLOR, escapeHtml } from '../lib/presentation'
import RiskSurfaceLayer from './RiskSurfaceLayer'

const baseTiles = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
// Ajusta el encuadre cuando se solicita enfocar la ruta, preservando después el zoom elegido por el usuario.
function FitRoute({ route, resultsOpen, focusRequest }) {
  const map = useMap()
  const lastFocusRequest = useRef(null)
  useEffect(() => {
    if (route.length < 2 || lastFocusRequest.current === focusRequest) return
    lastFocusRequest.current = focusRequest
    const card = map.getContainer().closest('.map-stage')?.querySelector('.results-area')
    const floatingCard = card && window.matchMedia('(min-width: 1025px)').matches
    map.fitBounds(route, {
      paddingTopLeft: [48, 76],
      paddingBottomRight: [floatingCard ? card.offsetWidth + 36 : 42, 48],
    })
  }, [map, route, resultsOpen, focusRequest])
  return null
}

// Centra el mapa al cambiar una ubicación seleccionada desde el formulario.
function MapCenterUpdater({ center }) {
  const map = useMap()
  useEffect(() => {
    if (center) map.setView(center, map.getZoom(), { animate: true })
  }, [map, center])
  return null
}

// Actualiza el tamaño de Leaflet cuando cambia la distribución responsive del aplicativo.
function MapSizeFixer() {
  const map = useMap()
  useEffect(() => {
    const refresh = () => map.invalidateSize({ pan: false, debounceMoveend: true })
    const observer = new ResizeObserver(refresh)
    observer.observe(map.getContainer())
    refresh()
    const timer = window.setTimeout(refresh, 240)
    window.addEventListener('resize', refresh)
    return () => {
      window.clearTimeout(timer)
      observer.disconnect()
      window.removeEventListener('resize', refresh)
    }
  }, [map])
  return null
}

// Convierte el clic del usuario en un punto de inicio o destino según el modo de selección.
function MapClickPicker({ selectionMode, onPick }) {
  useMapEvents({
    click(event) {
      if (selectionMode) onPick(selectionMode, event.latlng)
    },
  })
  return null
}

// Dibuja la capa de calor histórica y elimina sus recursos al desactivarla.
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
      maxZoom: variant === 'predicted' ? 13 : 17,
      minOpacity: 0.12,
      max: variant === 'predicted' ? 1.2 : 1,
      gradient,
    }).addTo(map)
    return () => map.removeLayer(layer)
  }, [map, points, variant])
  return null
}

// Representa los delitos históricos con una capa canvas para reducir el costo de dibujar numerosos puntos.
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
          `<strong>${escapeHtml(point.modalidad)}</strong><br>Peso: ${point.peso_delito}/5<br>${escapeHtml(point.distrito)}`,
          { direction: 'top' },
        )
        .addTo(group)
    })
    return () => map.removeLayer(group)
  }, [map, points])
  return null
}

// Comunica el área visible y el zoom para consultar únicamente el riesgo correspondiente.
function MapViewportReporter({ onBoundsChange }) {
  const map = useMap()
  const updateViewport = useCallback(() => {
    const bounds = map.getBounds()
    onBoundsChange({
      bounds: [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()],
      zoom: Math.round(map.getZoom()),
    })
  }, [map, onBoundsChange])
  useEffect(() => { updateViewport() }, [updateViewport])
  useMapEvents({
    zoomend() {
      updateViewport()
    },
    moveend() {
      updateViewport()
    },
  })
  return null
}

// Dibuja la geometría de las rutas con sus estilos y la información de riesgo de cada tramo.
function RouteLines({ positions, segments, color, title, dashed, comparing, modelName }) {
  if (positions.length < 2) return null
  return <>
    <Polyline positions={positions} pathOptions={{ color: '#ffffff', weight: dashed ? 7 : 10, opacity: .92, lineCap: 'round', lineJoin: 'round' }} interactive={false} />
    <Polyline positions={positions} pathOptions={{ color, weight: dashed ? 4 : 7,
      dashArray: dashed ? '10 8' : undefined, opacity: dashed ? .9 : 1, lineCap: 'round', lineJoin: 'round' }}
      interactive={!segments.length}>{!segments.length && <Tooltip sticky>{title}</Tooltip>}</Polyline>
    {segments.length ? segments.map((segment) => <Polyline key={`${segment.id_segmento}-${segment.orden}`}
      positions={segment.positions} pathOptions={{ color, weight: 10, opacity: 0 }}>
      <Tooltip sticky><strong>{title}</strong><br />
        Mapa · máximo {(segment.riesgo_mapa_maximo * 100).toFixed(1)} / 100 · promedio {(segment.riesgo_segmento_normalizado * 100).toFixed(1)} / 100<br />
        Rojo: {(segment.metros_rojos ?? 0).toFixed(0)} m · Naranja: {(segment.metros_naranjas ?? 0).toFixed(0)} m<br />
        {modelName} · Mensual{segment.compartido && comparing ? ' · Tramo compartido' : ''}
      </Tooltip>
    </Polyline>) : null}
  </>
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

// Integra el mapa base, las capas de riesgo, las rutas y los controles de visualización.
function MapView({
  mapCenter,
  selectionMode,
  onPick,
  origin,
  destination,
  originLabel,
  destinationLabel,
  heatmapPoints,
  crimePoints,
  predictionSurface,
  crimeTotal,
  predictionAvailable,
  predictionCounts,
  onPredictionBoundsChange,
  modelName,
  crimeFilters,
  filterOptions,
  mapLayers,
  mapDataLoading,
  mapError,
  onFilterChange,
  onLayerChange,
  onResetFilters,
  safeRoutePositions,
  traditionalRoutePositions,
  routePreference,
  safeSegments,
  traditionalSegments,
  routeView,
  routeFocusRequest,
  sameRoute,
  accessConnectors,
  resultsOpen,
}) {
  const [isDark, setIsDark] = useState(false)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const recommended = safeRoutePositions
  const alternative = traditionalRoutePositions
  const safeDetails = useMemo(() => (safeSegments ?? []).filter((segment) => segment.coordenadas?.length > 1)
    .map((segment) => ({ ...segment, positions: segment.coordenadas.map((point) => [point.lat, point.lng]) })), [safeSegments])
  const fastDetails = useMemo(() => (traditionalSegments ?? []).filter((segment) => segment.coordenadas?.length > 1)
    .map((segment) => ({ ...segment, positions: segment.coordenadas.map((point) => [point.lat, point.lng]) })), [traditionalSegments])
  const recommendedDetails = safeDetails
  const alternativeDetails = fastDetails
  const effectiveView = sameRoute ? 'both' : routeView
  const showRecommended = effectiveView === 'both' || effectiveView === 'safe'
  const showAlternative = !sameRoute && (effectiveView === 'both' || effectiveView === 'fast')
  const fitPositions = useMemo(() => {
    const positions = sameRoute ? safeRoutePositions : routeView === 'safe' ? safeRoutePositions
      : routeView === 'fast' ? traditionalRoutePositions : [...safeRoutePositions, ...traditionalRoutePositions]
    return positions.length > 1 ? [...(origin ? [origin] : []), ...positions, ...(destination ? [destination] : [])] : []
  }, [safeRoutePositions, traditionalRoutePositions, routeView, sameRoute, origin?.[0], origin?.[1], destination?.[0], destination?.[1]])
  const originIcon = useMemo(
    () =>
      L.divIcon({
        className: 'pin-icon pin-icon--green',
        html: '<svg viewBox="0 0 32 42" aria-hidden="true"><path d="M16 40S2 24 2 16a14 14 0 0 1 28 0c0 8-14 24-14 24Z" fill="#22c55e" stroke="white" stroke-width="3"/><circle cx="16" cy="16" r="5" fill="white"/></svg>',
        iconSize: [32, 42],
        iconAnchor: [16, 40],
      }),
    [],
  )
  const destinationIcon = useMemo(
    () =>
      L.divIcon({
        className: 'pin-icon pin-icon--red',
        html: '<svg viewBox="0 0 32 42" aria-hidden="true"><path d="M16 40S2 24 2 16a14 14 0 0 1 28 0c0 8-14 24-14 24Z" fill="#ef4444" stroke="white" stroke-width="3"/><circle cx="16" cy="16" r="5" fill="white"/></svg>',
        iconSize: [32, 42],
        iconAnchor: [16, 40],
      }),
    [],
  )

  return (
    <div className={`map-wrapper ${selectionMode ? 'is-selecting' : ''} ${isDark ? 'map-wrapper--dark' : ''}`}>
      <MapContainer
        center={mapCenter}
        zoom={13}
        maxZoom={18}
        className="map"
        zoomControl={false}
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url={baseTiles}
          maxZoom={19}
        />
        <MapCenterUpdater center={mapCenter} />
        <ZoomControl position="topleft" />
        <MapSizeFixer />
        <MapClickPicker selectionMode={selectionMode} onPick={onPick} />
        <HeatLayer points={heatmapPoints} variant="historical" />
        <CrimeLayer points={crimePoints} />
        <MapViewportReporter onBoundsChange={onPredictionBoundsChange} />
        <Pane name="risk-heatmap" style={{ zIndex: 410, pointerEvents: 'none' }}>
          <RiskSurfaceLayer surface={predictionSurface} enabled={mapLayers.predictionHeatmap} />
        </Pane>

        {origin && <Marker position={origin} icon={originIcon}><Tooltip permanent direction="right" offset={[13, -22]} className="endpoint-label"><strong>Inicio</strong><span>{originLabel || 'Punto de partida'}</span></Tooltip></Marker>}
        {destination && <Marker position={destination} icon={destinationIcon}><Tooltip permanent direction="right" offset={[13, -22]} className="endpoint-label"><strong>Destino</strong><span>{destinationLabel || 'Punto de llegada'}</span></Tooltip></Marker>}
        <Pane name="alternative-route" style={{ zIndex: 440 }}>
          {showAlternative && <RouteLines positions={alternative} segments={alternativeDetails} color={ALTERNATIVE_COLOR}
            title="Ruta alternativa · Más corta" dashed comparing={showRecommended} modelName={modelName} />}
        </Pane>
        <Pane name="recommended-route" style={{ zIndex: 450 }}>
          {showRecommended && <RouteLines positions={recommended} segments={recommendedDetails} color={ROUTE_COLOR}
            title="Ruta recomendada · Menor riesgo" dashed={false} comparing={showAlternative} modelName={modelName} />}
          {(accessConnectors ?? []).map((connection, index) => <Polyline key={`access-${index}`}
            positions={connection.map((point) => [point.lat, point.lng])}
            pathOptions={{ color: '#64748b', weight: 3, dashArray: '3 6' }}>
            <Tooltip sticky>Acceso a la red vial · Riesgo no evaluado</Tooltip>
          </Polyline>)}
        </Pane>
        <FitRoute route={fitPositions} resultsOpen={resultsOpen} focusRequest={routeFocusRequest} />
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
          aria-expanded={filtersOpen}
          aria-controls="map-filters"
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

      {!filtersOpen && (mapError || (mapDataLoading && mapLayers.predictionHeatmap && !predictionSurface)) && (
        <div className="map-data-message" role="status">
          {mapError || `Cargando riesgo mensual de ${modelName}…`}
        </div>
      )}

      {filtersOpen && (
        <aside className="map-filter-panel" id="map-filters" aria-label="Capas y filtros del mapa">
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
                          ? `${modelName}: ${predictionAvailable.toLocaleString('es-PE')} tramos en esta vista`
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
              <BrainCircuit size={14} /> Mapa de calor predictivo · {modelName}
            </label>
          </div>

          <details className="historical-filters">
            <summary>Filtros del histórico</summary>
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
          </details>
          <p className="filter-note">
            El ruteo utiliza el riesgo mensual del modelo seleccionado. Los filtros del histórico solo modifican esas capas.
          </p>
          {mapError && <p className="filter-note" role="status">{mapError}</p>}
          {mapLayers.predictionHeatmap && <div className="prediction-counts">
            <strong>Tramos en esta vista</strong>
            {Object.entries(RISK_LEVELS).map(([key, level]) => <span key={key}>
              <i style={{ background: level.color }} />{level.label}: {predictionCounts[key].toLocaleString('es-PE')}
            </span>)}
            <small>El calor agrupa focos cercanos y destaca los scores más elevados. El ruteo evalúa todos los tramos, incluidos los que no destacan en esta capa.</small>
          </div>}
        </aside>
      )}

      {!filtersOpen && (mapLayers.heatmap || mapLayers.predictionHeatmap || recommended.length > 1) && (
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
              <strong>Nivel de riesgo en la zona</strong>
              <div className="street-risk-gradient" />
              <div className="street-risk-labels">
                <span>Muy bajo</span><span>Bajo</span><span>Medio</span><span>Alto</span><span>Muy alto</span>
              </div>
              <small className="surface-note">{modelName} · Focos de mayor riesgo. Sin color no implica riesgo nulo.</small>
            </div>
          )}
          {recommended.length > 1 && <div className="heatmap-legend route-legend">
            {showRecommended && <span><i className="legend-route-line" />Ruta recomendada · Menor riesgo</span>}
            {showAlternative && <span><i className="legend-route-line alternative" />Ruta alternativa · Más corta</span>}
          </div>}
        </div>
      )}
    </div>
  )
}

export default MapView
