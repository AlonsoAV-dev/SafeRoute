import { useEffect, useMemo, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import './App.css'
import RoutePanel from './components/RoutePanel'
import MapView from './components/MapView'
import InfoPanel from './components/InfoPanel'

const API_URL = 'http://127.0.0.1:8000/api'
const LIMA_METRO_CENTER = [-12.0464, -77.0428]
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

const parseCoordinatePair = (latValue, lngValue) => {
  if (latValue === '' || lngValue === '') return null
  const lat = Number(latValue)
  const lng = Number(lngValue)
  return Number.isFinite(lat) && Number.isFinite(lng) ? [lat, lng] : null
}

function App() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [originQuery, setOriginQuery] = useState('')
  const [destinationQuery, setDestinationQuery] = useState('')
  const [routeData, setRouteData] = useState({ safe: null, traditional: null })
  const [routeMeta, setRouteMeta] = useState(null)
  const [heatmapPoints, setHeatmapPoints] = useState([])
  const [crimePoints, setCrimePoints] = useState([])
  const [predictionHeatmapPoints, setPredictionHeatmapPoints] = useState([])
  const [predictionPoints, setPredictionPoints] = useState([])
  const [crimeTotal, setCrimeTotal] = useState(0)
  const [predictionTotal, setPredictionTotal] = useState(0)
  const [filterOptions, setFilterOptions] = useState(DEFAULT_FILTER_OPTIONS)
  const [crimeFilters, setCrimeFilters] = useState(DEFAULT_FILTERS)
  const [mapLayers, setMapLayers] = useState({
    crimes: false,
    heatmap: false,
    predictionHeatmap: true,
    predictionPoints: false,
  })
  const [mapDataLoading, setMapDataLoading] = useState(false)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [selectionMode, setSelectionMode] = useState(null)
  const [geoStatus, setGeoStatus] = useState({ origin: '', destination: '' })
  const [geoLoading, setGeoLoading] = useState({ origin: false, destination: false })
  const [mapCenter, setMapCenter] = useState(LIMA_METRO_CENTER)
  const [routePreference, setRoutePreference] = useState('safe')
  const [riskMode, setRiskMode] = useState('predicted')
  const [isResultsMinimized, setIsResultsMinimized] = useState(false)

  const safeRoute = routeData.safe
  const traditionalRoute = routeData.traditional
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
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = [position.coords.latitude, position.coords.longitude]
        const coordinateLabel = `${coords[0].toFixed(6)}, ${coords[1].toFixed(6)}`
        setMapCenter(coords)
        setForm((current) => ({
          ...current,
          originLat: current.originLat || coords[0].toFixed(6),
          originLng: current.originLng || coords[1].toFixed(6),
        }))
        setOriginQuery((current) => current || coordinateLabel)
      },
      () => setMapCenter(LIMA_METRO_CENTER),
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 },
    )
  }, [])

  useEffect(() => {
    async function loadFilterOptions() {
      try {
        const response = await fetch(`${API_URL}/crime-filters`)
        if (response.ok) setFilterOptions(await response.json())
      } catch {
        // Los valores por defecto mantienen el mapa utilizable.
      }
    }
    loadFilterOptions()
  }, [])

  useEffect(() => {
    const query = new URLSearchParams()
    Object.entries(crimeFilters).forEach(([key, value]) => {
      if (value && value !== 'todos') query.set(key, value)
    })
    const suffix = query.toString() ? `?${query.toString()}` : ''

    async function loadMapData() {
      setMapDataLoading(true)
      try {
        const requests = [
          mapLayers.heatmap ? fetch(`${API_URL}/heatmap${suffix}`) : null,
          mapLayers.crimes ? fetch(`${API_URL}/crime-points${suffix}`) : null,
          mapLayers.predictionHeatmap ? fetch(`${API_URL}/prediction-heatmap`) : null,
          mapLayers.predictionPoints
            ? fetch(`${API_URL}/prediction-points?min_score=0.34&limit=15000`)
            : null,
        ]
        const [heatmapResponse, crimesResponse, predictionHeatmapResponse, predictionPointsResponse] =
          await Promise.all(requests)

        if (heatmapResponse) {
          if (!heatmapResponse.ok) throw new Error('No se pudo cargar el mapa de calor.')
          const heatmapData = await heatmapResponse.json()
          setHeatmapPoints(heatmapData.points ?? [])
        } else {
          setHeatmapPoints([])
        }

        if (crimesResponse) {
          if (!crimesResponse.ok) throw new Error('No se pudieron cargar los delitos.')
          const crimesData = await crimesResponse.json()
          setCrimePoints(crimesData.points ?? [])
          setCrimeTotal(crimesData.total ?? 0)
        } else {
          setCrimePoints([])
          setCrimeTotal(0)
        }

        if (predictionHeatmapResponse) {
          if (!predictionHeatmapResponse.ok) throw new Error('No se pudo cargar el calor RF.')
          const predictionHeatmapData = await predictionHeatmapResponse.json()
          setPredictionHeatmapPoints(predictionHeatmapData.points ?? [])
        } else {
          setPredictionHeatmapPoints([])
        }

        if (predictionPointsResponse) {
          if (!predictionPointsResponse.ok) throw new Error('No se pudieron cargar los tramos RF.')
          const predictionData = await predictionPointsResponse.json()
          setPredictionPoints(predictionData.points ?? [])
          setPredictionTotal(predictionData.total ?? 0)
        } else {
          setPredictionPoints([])
          setPredictionTotal(0)
        }
      } catch (requestError) {
        setError(requestError.message)
      } finally {
        setMapDataLoading(false)
      }
    }
    loadMapData()
  }, [
    crimeFilters,
    mapLayers.crimes,
    mapLayers.heatmap,
    mapLayers.predictionHeatmap,
    mapLayers.predictionPoints,
  ])

  async function handleSubmit(event) {
    event.preventDefault()
    if (!origin || !destination) {
      setError('Selecciona un origen y un destino válidos.')
      setStatus('error')
      return
    }

    setStatus('loading')
    setError('')
    try {
      const response = await fetch(`${API_URL}/route/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin,
          destination,
          alpha: Number(form.safetyWeight),
          datetime: new Date().toISOString().slice(0, 16),
          routePreference,
          modelo_riesgo: 'random_forest',
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
      setRouteData({ safe: data.safe_route, traditional: data.traditional_route })
      setRouteMeta(data)
      setStatus('success')
      setIsResultsMinimized(false)
    } catch (requestError) {
      setError(requestError.message)
      setStatus('error')
    }
  }

  function handlePreferenceChange(preference) {
    setRoutePreference(preference)
    setForm((current) => ({
      ...current,
      safetyWeight: preference === 'safe' ? 0.7 : 0,
    }))
  }

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
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`,
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
        riskMode={riskMode}
        status={status}
        error={error}
        onOriginQueryChange={setOriginQuery}
        onDestinationQueryChange={setDestinationQuery}
        onClearLocation={handleClearLocation}
        onSelectionModeChange={setSelectionMode}
        onPreferenceChange={handlePreferenceChange}
        onRiskModeChange={setRiskMode}
        onGeocode={handleGeocode}
        onSubmit={handleSubmit}
      />

      <section className="map-column">
        <div className="map-stage">
          <MapView
            mapCenter={mapCenter}
            selectionMode={selectionMode}
            onPick={handlePickFromMap}
            origin={origin}
            destination={destination}
            heatmapPoints={heatmapPoints}
            crimePoints={crimePoints}
            predictionHeatmapPoints={predictionHeatmapPoints}
            predictionPoints={predictionPoints}
            crimeTotal={crimeTotal}
            predictionTotal={predictionTotal}
            crimeFilters={crimeFilters}
            filterOptions={filterOptions}
            mapLayers={mapLayers}
            mapDataLoading={mapDataLoading}
            onFilterChange={(name, value) =>
              setCrimeFilters((current) => ({ ...current, [name]: value }))
            }
            onLayerChange={(name, value) =>
              setMapLayers((current) => ({ ...current, [name]: value }))
            }
            onResetFilters={() => setCrimeFilters(DEFAULT_FILTERS)}
            safeRoutePositions={safeRoutePositions}
            traditionalRoutePositions={routeMeta?.misma_ruta ? [] : traditionalRoutePositions}
          />
          <div className={`map-overlay ${isResultsMinimized ? 'is-minimized' : ''}`}>
            <button
              type="button"
              className="map-overlay-toggle"
              onClick={() => setIsResultsMinimized((current) => !current)}
              aria-label={isResultsMinimized ? 'Mostrar resultados' : 'Minimizar resultados'}
            >
              {isResultsMinimized ? '+' : '−'}
            </button>
            {!isResultsMinimized && (
              <InfoPanel
                safeRoute={safeRoute}
                traditionalRoute={traditionalRoute}
                safeMinutes={safeMinutes}
                traditionalMinutes={traditionalMinutes}
                riskReduction={riskReduction}
                routeMeta={routeMeta}
              />
            )}
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
