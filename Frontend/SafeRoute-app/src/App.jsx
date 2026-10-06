import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, MapPin, Route } from 'lucide-react'
import 'leaflet/dist/leaflet.css'
import './App.css'
import RoutePanel from './components/RoutePanel'
import MapView from './components/MapView'
import InfoPanel from './components/InfoPanel'
import { MODEL_OPTIONS, RISK_LEVELS } from './lib/presentation'

// Define la conexión con la API y los valores iniciales de la consulta y los filtros.
const API_URL = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000/api`
const LIMA_METRO_CENTER = [-12.0464, -77.0428]
const EMPTY_ROUTE_POSITIONS = Object.freeze([])
const SHARED_RISK_FIELD_VERSION = 'shared-risk-field-v1'
const DEFAULT_FORM = {
  originLat: '',
  originLng: '',
  destinationLat: '',
  destinationLng: '',
  safetyWeight: 0.7,
}
const DEFAULT_FILTERS = {
  turno: 'todos',
  dia_semana: 'todos',
  tipo: 'todos',
  modalidad: 'todos',
}
const DEFAULT_FILTER_OPTIONS = {
  turnos: ['todos', 'manana', 'tarde', 'noche', 'madrugada'],
  dias_semana: ['todos'],
  tipos: ['todos'],
  modalidades: ['todos'],
}

// Convierte los campos del formulario a coordenadas únicamente cuando ambos valores son válidos.
const parseCoordinatePair = (latValue, lngValue) => {
  if (latValue === '' || lngValue === '') return null
  const lat = Number(latValue)
  const lng = Number(lngValue)
  return Number.isFinite(lat) && Number.isFinite(lng) ? [lat, lng] : null
}

// Coordina el formulario, las capas del mapa, las consultas y los resultados de las rutas.
function App() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [originQuery, setOriginQuery] = useState('')
  const [destinationQuery, setDestinationQuery] = useState('')
  const [routeData, setRouteData] = useState({ safe: null, traditional: null })
  const [routeMeta, setRouteMeta] = useState(null)
  const [heatmapPoints, setHeatmapPoints] = useState([])
  const [crimePoints, setCrimePoints] = useState([])
  const [predictionSurface, setPredictionSurface] = useState(null)
  const [crimeTotal, setCrimeTotal] = useState(0)
  const [predictionAvailable, setPredictionAvailable] = useState(0)
  const [predictionBounds, setPredictionBounds] = useState(null)
  const [predictionCounts, setPredictionCounts] = useState({ bajo: 0, medio: 0, alto: 0 })
  const [filterOptions, setFilterOptions] = useState(DEFAULT_FILTER_OPTIONS)
  const [crimeFilters, setCrimeFilters] = useState(DEFAULT_FILTERS)
  const [mapLayers, setMapLayers] = useState({
    crimes: false,
    heatmap: false,
    predictionHeatmap: true,
  })
  const [historicalDataLoading, setHistoricalDataLoading] = useState(false)
  const [predictionDataLoading, setPredictionDataLoading] = useState(false)
  const [mapErrors, setMapErrors] = useState({ historical: '', prediction: '' })
  const mapDataLoading = historicalDataLoading || predictionDataLoading
  const mapError = [
    (mapLayers.heatmap || mapLayers.crimes) && mapErrors.historical,
    mapLayers.predictionHeatmap && mapErrors.prediction,
  ].filter(Boolean).join(' ')
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [selectionMode, setSelectionMode] = useState(null)
  const [geoStatus, setGeoStatus] = useState({ origin: '', destination: '' })
  const [geoLoading, setGeoLoading] = useState({ origin: false, destination: false })
  const [mapCenter, setMapCenter] = useState(LIMA_METRO_CENTER)
  const [routePreference, setRoutePreference] = useState('safe')
  const riskMode = 'predicted'
  const [riskModel, setRiskModel] = useState('random_forest')
  const [models, setModels] = useState(MODEL_OPTIONS.map((model) => ({ ...model, available: false })))
  const [modelsLoading, setModelsLoading] = useState(true)
  const [resultsOpen, setResultsOpen] = useState(true)
  const [routeView, setRouteView] = useState('both')
  const [routeFocusRequest, setRouteFocusRequest] = useState(0)
  const routeRequest = useRef(null)
  const activeModel = models.find((model) => model.key === riskModel)
  const handlePredictionBoundsChange = useCallback((viewport) => {
    setPredictionBounds((current) => current && current.zoom === viewport.zoom
      && viewport.bounds.every((value, index) => Math.abs(value - current.bounds[index]) < 0.000001)
      ? current : viewport)
  }, [])
  const handleLayerChange = useCallback((name, value) => {
    setMapLayers((current) => ({ ...current, [name]: value,
      ...(value && name === 'heatmap' ? { predictionHeatmap: false } : {}),
      ...(value && name === 'predictionHeatmap' ? { heatmap: false } : {}),
    }))
  }, [])

  const compatibleRoute = routeMeta?.risk_field?.version === SHARED_RISK_FIELD_VERSION
  const safeRoute = compatibleRoute ? routeData.safe : null
  const traditionalRoute = compatibleRoute ? routeData.traditional : null
  const origin = parseCoordinatePair(form.originLat, form.originLng)
  const destination = parseCoordinatePair(form.destinationLat, form.destinationLng)
  const safeRoutePositions = useMemo(
    () => safeRoute?.route.map((point) => [point.lat, point.lng]) ?? [],
    [safeRoute],
  )
  const traditionalRoutePositions = useMemo(
    () => traditionalRoute?.route.map((point) => [point.lat, point.lng]) ?? [],
    [traditionalRoute],
  )

  useEffect(() => {
    const controller = new AbortController()
    // Consulta los modelos disponibles y selecciona uno que disponga de predicciones exportadas.
    async function loadModels() {
      try {
        const response = await fetch(`${API_URL}/models`, { signal: controller.signal })
        if (!response.ok) throw new Error('No se pudieron consultar los modelos disponibles.')
        const data = await response.json()
        if (controller.signal.aborted) return
        setModels(MODEL_OPTIONS.map((option) => ({
          ...option, ...data.models.find((model) => model.key === option.key),
        })))
        setRiskModel((current) => data.models.some((model) => model.key === current && model.available)
          ? current : data.default_model)
      } catch (requestError) {
        if (requestError.name !== 'AbortError') setError('No se pudo conectar con el servicio de predicción.')
      } finally {
        if (!controller.signal.aborted) setModelsLoading(false)
      }
    }
    loadModels()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    // Descarta la ruta anterior cuando cambian el modelo o los extremos de la consulta.
    routeRequest.current?.abort()
    setRouteData({ safe: null, traditional: null })
    setRouteMeta(null)
    setStatus('idle')
    setResultsOpen(true)
    setRouteView('both')
  }, [riskModel, form.originLat, form.originLng, form.destinationLat, form.destinationLng])

  useEffect(() => () => routeRequest.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    // Carga las opciones de los filtros históricos conservando sus valores iniciales si falla la consulta.
    async function loadFilterOptions() {
      try {
        const response = await fetch(`${API_URL}/crime-filters`, { signal: controller.signal })
        if (response.ok) setFilterOptions(await response.json())
      } catch {
        // Los valores por defecto mantienen el mapa utilizable.
      }
    }
    loadFilterOptions()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const query = new URLSearchParams()
    Object.entries(crimeFilters).forEach(([key, value]) => {
      if (value && value !== 'todos') query.set(key, value)
    })
    const suffix = query.toString() ? `?${query.toString()}` : ''

    // Solicita las capas históricas activas y cancela las respuestas de consultas anteriores.
    async function loadMapData() {
      setHistoricalDataLoading(true)
      setMapErrors((current) => ({ ...current, historical: '' }))
      try {
        const requests = [
          mapLayers.heatmap ? fetch(`${API_URL}/heatmap${suffix}`, { signal: controller.signal }) : null,
          mapLayers.crimes ? fetch(`${API_URL}/crime-points${suffix}`, { signal: controller.signal }) : null,
        ]
        const [heatmapResponse, crimesResponse] =
          await Promise.all(requests)
        if (controller.signal.aborted) return

        if (heatmapResponse) {
          if (!heatmapResponse.ok) throw new Error('No se pudo cargar el mapa de calor.')
          const heatmapData = await heatmapResponse.json()
          if (controller.signal.aborted) return
          setHeatmapPoints(heatmapData.points ?? [])
        } else {
          setHeatmapPoints([])
        }

        if (crimesResponse) {
          if (!crimesResponse.ok) throw new Error('No se pudieron cargar los delitos.')
          const crimesData = await crimesResponse.json()
          if (controller.signal.aborted) return
          setCrimePoints(crimesData.points ?? [])
          setCrimeTotal(crimesData.total ?? 0)
        } else {
          setCrimePoints([])
          setCrimeTotal(0)
        }

      } catch (requestError) {
        if (requestError.name !== 'AbortError') setMapErrors((current) => ({ ...current, historical: requestError.message }))
      } finally {
        if (!controller.signal.aborted) setHistoricalDataLoading(false)
      }
    }
    loadMapData()
    return () => controller.abort()
  }, [
    crimeFilters,
    mapLayers.crimes,
    mapLayers.heatmap,
  ])

  useEffect(() => {
    const controller = new AbortController()
    setMapErrors((current) => ({ ...current, prediction: '' }))
    setPredictionAvailable(0)
    setPredictionCounts({ bajo: 0, medio: 0, alto: 0 })
    if (!mapLayers.predictionHeatmap || !predictionBounds || modelsLoading) {
      setPredictionSurface(null)
      setPredictionDataLoading(false)
      return () => controller.abort()
    }
    setPredictionDataLoading(true)
    const timer = window.setTimeout(async () => {
      try {
        // Consulta el campo de riesgo del área visible y comprueba su modelo, periodo y versión antes de mostrarlo.
        const query = new URLSearchParams({
          modelo_riesgo: riskModel, bbox: predictionBounds.bounds.join(','), zoom: String(predictionBounds.zoom),
        })
        const response = await fetch(`${API_URL}/risk-surface?${query}`, { signal: controller.signal })
        if (!response.ok) throw new Error('No se pudo cargar el riesgo del modelo seleccionado.')
        const data = await response.json()
        if (controller.signal.aborted) return
        if (data.model_key !== riskModel || data.prediction_period !== activeModel?.prediction_period || data.risk_scope !== 'mensual'
          || data.risk_field?.version !== SHARED_RISK_FIELD_VERSION) {
          throw new Error('El mapa recibió predicciones de un modelo o periodo diferente al seleccionado.')
        }
        setPredictionSurface(data)
        setPredictionAvailable(data.total_segments ?? 0)
        setPredictionCounts(data.counts_by_level ?? { bajo: 0, medio: 0, alto: 0 })
      } catch (requestError) {
        if (requestError.name !== 'AbortError') setMapErrors((current) => ({ ...current, prediction: requestError.message }))
      } finally {
        if (!controller.signal.aborted) setPredictionDataLoading(false)
      }
    }, 250)
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [mapLayers.predictionHeatmap, predictionBounds, riskModel, modelsLoading, activeModel?.prediction_period])

  // Valida origen y destino, solicita las dos rutas y comprueba que usan el mismo campo del mapa.
  async function handleSubmit(event) {
    event.preventDefault()
    if (!origin || !destination) {
      setError('Selecciona un origen y un destino válidos.')
      setStatus('error')
      return
    }

    setStatus('loading')
    setError('')
    routeRequest.current?.abort()
    const controller = new AbortController()
    routeRequest.current = controller
    try {
      const response = await fetch(`${API_URL}/route/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          origin,
          destination,
          alpha: Number(form.safetyWeight),
          turno: null,
          routePreference,
          modelo_riesgo: riskModel,
          beta: 10,
          buffer_m: 200,
          risk_mode: riskMode,
        }),
      })
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}))
        throw new Error(errorBody.detail || 'No se pudo generar la ruta.')
      }
      const data = await response.json()
      if (controller.signal.aborted) return
      if (data.modelo_usado !== activeModel?.name || data.periodo_prediccion !== activeModel?.prediction_period || data.turno_riesgo !== null
        || data.risk_field?.version !== SHARED_RISK_FIELD_VERSION) {
        throw new Error('La ruta recibió predicciones de un modelo o periodo diferente al mostrado en el mapa.')
      }
      setRouteData({ safe: data.safe_route, traditional: data.traditional_route })
      setRouteMeta(data)
      setStatus('success')
      setResultsOpen(true)
      setRouteView('both')
      setRouteFocusRequest((current) => current + 1)
    } catch (requestError) {
      if (requestError.name === 'AbortError') return
      setError(requestError.message)
      setStatus('error')
    }
  }

  function handleAddressChange(type, value) {
    if (type === 'origin') setOriginQuery(value)
    else setDestinationQuery(value)
    setForm((current) => ({ ...current, [`${type}Lat`]: '', [`${type}Lng`]: '' }))
  }

  function handleSelectionModeChange(next) {
    setSelectionMode(next)
    if (window.matchMedia('(max-width: 1024px)').matches) {
      document.getElementById('map-stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  function handlePreferenceChange(preference) {
    setRoutePreference(preference)
    setForm((current) => ({
      ...current,
      safetyWeight: preference === 'safe' ? 0.7 : 0,
    }))
  }

  // Solicita el enfoque de una ruta solo cuando el usuario cambia la vista de resultados.
  function handleRouteViewChange(view) {
    setRouteView(view)
    setRouteFocusRequest((current) => current + 1)
    if (window.matchMedia('(max-width: 1024px)').matches) {
      document.getElementById('map-stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  // Obtiene una etiqueta de dirección para las coordenadas seleccionadas en el mapa.
  async function reverseGeocode(type, latlng) {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latlng.lat}&lon=${latlng.lng}`,
        { headers: { 'Accept-Language': 'es' } },
      )
      if (!response.ok) return
      const result = await response.json()
      if (!result.display_name) return
      if (type === 'origin') setOriginQuery(result.display_name)
      else setDestinationQuery(result.display_name)
    } catch {
      // Las coordenadas siguen siendo válidas si falla el geocodificador.
    }
  }

  // Guarda las coordenadas elegidas y completa su dirección sin impedir el uso del mapa si falla la consulta.
  function handlePickFromMap(type, latlng) {
    const coordinateLabel = `${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}`
    setForm((current) => ({
      ...current,
      [`${type}Lat`]: latlng.lat.toFixed(6),
      [`${type}Lng`]: latlng.lng.toFixed(6),
    }))
    if (type === 'origin') setOriginQuery(coordinateLabel)
    else setDestinationQuery(coordinateLabel)
    setSelectionMode(null)
    reverseGeocode(type, latlng)
  }

  // Busca una dirección en el área de Lima y actualiza el extremo correspondiente del recorrido.
  async function handleGeocode(type) {
    const query = type === 'origin' ? originQuery : destinationQuery
    if (!query.trim()) {
      setGeoStatus((current) => ({
        ...current,
        [type]: 'Ingresa una dirección o referencia válida.',
      }))
      return
    }
    setGeoStatus((current) => ({ ...current, [type]: '' }))
    setGeoLoading((current) => ({ ...current, [type]: true }))
    try {
      const searchParams = new URLSearchParams({ format: 'json', limit: '1', q: query,
        countrycodes: 'pe', viewbox: '-77.30,-11.60,-76.50,-12.50', bounded: '1' })
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?${searchParams}`,
        { headers: { 'Accept-Language': 'es' } },
      )
      if (!response.ok) throw new Error('No se pudo buscar la dirección.')
      const results = await response.json()
      if (!results.length) throw new Error('No se encontró una coincidencia.')
      const result = results[0]
      setForm((current) => ({
        ...current,
        [`${type}Lat`]: Number(result.lat).toFixed(6),
        [`${type}Lng`]: Number(result.lon).toFixed(6),
      }))
      if (type === 'origin') setOriginQuery(result.display_name)
      else setDestinationQuery(result.display_name)
      setMapCenter([Number(result.lat), Number(result.lon)])
      setSelectionMode(null)
    } catch (requestError) {
      setGeoStatus((current) => ({ ...current, [type]: requestError.message }))
    } finally {
      setGeoLoading((current) => ({ ...current, [type]: false }))
    }
  }

  // Limpia un extremo y los mensajes asociados para permitir una nueva selección.
  function handleClearLocation(type) {
    setForm((current) => ({
      ...current,
      [`${type}Lat`]: '',
      [`${type}Lng`]: '',
    }))
    setGeoStatus((current) => ({ ...current, [type]: '' }))
    if (type === 'origin') setOriginQuery('')
    else setDestinationQuery('')
  }

  const estimateMinutes = (distanceKm, speedKmh = 25) =>
    distanceKm ? Math.round((distanceKm / speedKmh) * 60) : null
  const safeMinutes = safeRoute?.time_min ?? estimateMinutes(safeRoute?.distance_km)
  const traditionalMinutes =
    traditionalRoute?.time_min ?? estimateMinutes(traditionalRoute?.distance_km)
  const riskReduction =
    routeMeta?.risk_reduction ??
    (safeRoute && traditionalRoute
      ? Math.max(
          0,
          Math.round(
            (1 - safeRoute.risk_score / Math.max(traditionalRoute.risk_score, 0.01)) * 100,
          ),
        )
      : null)

  return (
    <main className="app-shell">
      <RoutePanel
        originQuery={originQuery}
        destinationQuery={destinationQuery}
        selectionMode={selectionMode}
        geoLoading={geoLoading}
        geoStatus={geoStatus}
        routePreference={routePreference}
        riskModel={riskModel}
        models={models}
        modelsLoading={modelsLoading}
        activeModel={activeModel}
        status={status}
        error={error}
        onOriginQueryChange={(value) => handleAddressChange('origin', value)}
        onDestinationQueryChange={(value) => handleAddressChange('destination', value)}
        onClearLocation={handleClearLocation}
        onSelectionModeChange={handleSelectionModeChange}
        onPreferenceChange={handlePreferenceChange}
        onRiskModelChange={setRiskModel}
        onGeocode={handleGeocode}
        onSubmit={handleSubmit}
      />

      <section className="map-column" aria-label="Mapa y resultados de rutas">
        <header className="map-header">
          <MapPin size={27} aria-hidden="true" />
          <div>
            <h2>Mapa de rutas seguras</h2>
            <p>Lima Metropolitana</p>
          </div>
          <div className="risk-scale" aria-label="Escala del nivel de riesgo">
            {Object.entries(RISK_LEVELS).map(([key, level]) => (
              <span key={key}><i style={{ background: level.color }} />{level.label}</span>
            ))}
          </div>
        </header>
        <div className="map-stage" id="map-stage">
          <MapView
            mapCenter={mapCenter}
            selectionMode={selectionMode}
            onPick={handlePickFromMap}
            origin={origin}
            destination={destination}
            originLabel={originQuery.split(',').slice(0, 2).join(',')}
            destinationLabel={destinationQuery.split(',').slice(0, 2).join(',')}
            heatmapPoints={heatmapPoints}
            crimePoints={crimePoints}
            predictionSurface={predictionSurface?.model_key === riskModel && predictionSurface?.risk_field?.version === SHARED_RISK_FIELD_VERSION ? predictionSurface : null}
            crimeTotal={crimeTotal}
            predictionAvailable={predictionAvailable}
            predictionCounts={predictionCounts}
            onPredictionBoundsChange={handlePredictionBoundsChange}
            modelName={activeModel?.name ?? 'Modelo seleccionado'}
            crimeFilters={crimeFilters}
            filterOptions={filterOptions}
            mapLayers={mapLayers}
            mapDataLoading={mapDataLoading}
            mapError={mapError}
            onFilterChange={(name, value) =>
              setCrimeFilters((current) => ({ ...current, [name]: value }))
            }
            onLayerChange={handleLayerChange}
            onResetFilters={() => setCrimeFilters(DEFAULT_FILTERS)}
            safeRoutePositions={safeRoutePositions}
            traditionalRoutePositions={routeMeta?.misma_ruta ? EMPTY_ROUTE_POSITIONS : traditionalRoutePositions}
            routePreference={routePreference}
            safeSegments={safeRoute?.segments}
            traditionalSegments={traditionalRoute?.segments}
            routeView={routeView}
            routeFocusRequest={routeFocusRequest}
            sameRoute={routeMeta?.misma_ruta === true}
            accessConnectors={routePreference === 'fast' ? traditionalRoute?.access_connectors : safeRoute?.access_connectors}
            resultsOpen={resultsOpen && Boolean(safeRoute)}
          />
          {safeRoute && <section className="results-area" aria-label="Resultado del cálculo">
            <button className="results-toggle" type="button" aria-expanded={resultsOpen}
              aria-controls="route-results" onClick={() => setResultsOpen((current) => !current)}>
              <Route size={24} className="results-heading-icon" aria-hidden="true" />
              <span>Comparación de rutas<small>{activeModel?.name} · Riesgo mensual</small></span>
              {resultsOpen ? <ChevronUp size={17} /> : <ChevronDown size={17} />}
            </button>
            {resultsOpen && <InfoPanel safeRoute={safeRoute} traditionalRoute={traditionalRoute}
              safeMinutes={safeMinutes} traditionalMinutes={traditionalMinutes}
              riskReduction={riskReduction} routeMeta={routeMeta} routePreference={routePreference}
              routeView={routeView} onRouteViewChange={handleRouteViewChange} />}
          </section>}
        </div>
      </section>
    </main>
  )
}

export default App
