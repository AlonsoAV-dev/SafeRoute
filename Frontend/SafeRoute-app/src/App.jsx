import { useEffect, useMemo, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import './App.css'
import Sidebar from './components/Sidebar'
import RoutePanel from './components/RoutePanel'
import MapView from './components/MapView'
import InfoPanel from './components/InfoPanel'
import BottomMetrics from './components/BottomMetrics'

const API_URL = 'http://127.0.0.1:8000/api'

const DEFAULT_FORM = {
  originLat: '',
  originLng: '',
  destinationLat: '',
  destinationLng: '',
  turno: 'noche',
  safetyWeight: 0.7,
}

const LIMA_METRO_CENTER = [-12.0464, -77.0428]

const turnoByHour = (hour) => {
  if (hour >= 0 && hour < 6) return 'madrugada'
  if (hour >= 6 && hour < 12) return 'manana'
  if (hour >= 12 && hour < 18) return 'tarde'
  return 'noche'
}

const toNumber = (value) => {
  if (value === '' || value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const parseCoordinatePair = (latValue, lngValue) => {
  const lat = toNumber(latValue)
  const lng = toNumber(lngValue)
  if (lat === null || lng === null) return null
  return [lat, lng]
}

function App() {
  const [form, setForm] = useState(() => ({
    ...DEFAULT_FORM,
    turno: turnoByHour(new Date().getHours()),
  }))
  const [originQuery, setOriginQuery] = useState('')
  const [destinationQuery, setDestinationQuery] = useState('')
  const [routeData, setRouteData] = useState({ safe: null, traditional: null })
  const [riskZones, setRiskZones] = useState([])
  const [heatmapPoints, setHeatmapPoints] = useState([])
  const [systemStats, setSystemStats] = useState({
    model_accuracy: 0,
    zones_count: 0,
    calc_time_ms: 0,
  })
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [selectionMode, setSelectionMode] = useState(null)
  const [geoStatus, setGeoStatus] = useState({ origin: '', destination: '' })
  const [geoLoading, setGeoLoading] = useState({ origin: false, destination: false })
  const [mapCenter, setMapCenter] = useState(LIMA_METRO_CENTER)
  const [useCurrentTime, setUseCurrentTime] = useState(true)
  const [routePreference, setRoutePreference] = useState('safe')
  const [travelDate, setTravelDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [travelTime, setTravelTime] = useState(() => new Date().toTimeString().slice(0, 5))
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isResultsMinimized, setIsResultsMinimized] = useState(false)

  const safeRoute = routeData.safe
  const traditionalRoute = routeData.traditional

  const safeRoutePositions = useMemo(
    () => safeRoute?.route.map((point) => [point.lat, point.lng]) ?? [],
    [safeRoute],
  )
  const traditionalRoutePositions = useMemo(
    () => traditionalRoute?.route.map((point) => [point.lat, point.lng]) ?? [],
    [traditionalRoute],
  )

  const origin = parseCoordinatePair(form.originLat, form.originLng)
  const destination = parseCoordinatePair(form.destinationLat, form.destinationLng)
  const effectiveTurno = useCurrentTime
    ? turnoByHour(new Date().getHours())
    : form.turno

  useEffect(() => {
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = [position.coords.latitude, position.coords.longitude]
        setMapCenter(coords)
        setForm((currentForm) => ({
          ...currentForm,
          originLat: currentForm.originLat || coords[0].toFixed(6),
          originLng: currentForm.originLng || coords[1].toFixed(6),
        }))
      },
      () => {
        setMapCenter(LIMA_METRO_CENTER)
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 },
    )
  }, [])

  useEffect(() => {
    async function loadMapData() {
      try {
        const [zonesResponse, heatmapResponse] = await Promise.all([
          fetch(`${API_URL}/risk-zones?turno=${effectiveTurno}`),
          fetch(`${API_URL}/heatmap?turno=${effectiveTurno}`),
        ])

        if (!zonesResponse.ok) throw new Error('No se pudieron cargar las zonas.')
        if (!heatmapResponse.ok) throw new Error('No se pudo cargar el mapa de riesgo.')

        const zonesData = await zonesResponse.json()
        const heatmapData = await heatmapResponse.json()
        setRiskZones(zonesData.zones ?? [])
        setHeatmapPoints(heatmapData.points ?? [])
      } catch (requestError) {
        setError(requestError.message)
      }
    }

    loadMapData()
  }, [effectiveTurno])

  useEffect(() => {
    async function loadStats() {
      try {
        const response = await fetch(`${API_URL}/stats`)
        if (!response.ok) return
        const data = await response.json()
        setSystemStats(data)
      } catch (requestError) {
        console.error(requestError)
      }
    }

    loadStats()
  }, [])

  async function handleSubmit(event) {
    event.preventDefault()
    if (!origin || !destination) {
      setError('Selecciona un origen y un destino válidos.')
      setStatus('error')
      return
    }
    setStatus('loading')
    setError('')

    const timeValue = useCurrentTime
      ? new Date().toISOString().slice(0, 16)
      : `${travelDate}T${travelTime}`

    try {
      const response = await fetch(`${API_URL}/route/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: [origin[0], origin[1]],
          destination: [destination[0], destination[1]],
          alpha: Number(form.safetyWeight),
          datetime: timeValue,
        }),
      })

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}))
        throw new Error(errorBody.detail || 'No se pudo generar la ruta.')
      }

      const data = await response.json()
      setRouteData({ safe: data.safe_route, traditional: data.traditional_route })
      setStatus('success')
    } catch (requestError) {
      setError(requestError.message)
      setStatus('error')
    }
  }

  function updateField(event) {
    const { name, value } = event.target
    setForm((currentForm) => ({ ...currentForm, [name]: value }))
  }

  function toggleCurrentTime() {
    setUseCurrentTime((current) => {
      if (current) {
        setForm((currentForm) => ({
          ...currentForm,
          turno: turnoByHour(new Date().getHours()),
        }))
      }
      return !current
    })
  }

  const handlePreferenceChange = (preference) => {
    setRoutePreference(preference)
    setForm((currentForm) => ({
      ...currentForm,
      safetyWeight: preference === 'safe' ? 0.7 : 0,
    }))
  }

  const estimateMinutes = (distanceKm, speedKmh = 25) => {
    if (!distanceKm) return null
    return Math.round((distanceKm / speedKmh) * 60)
  }

  const safeMinutes = estimateMinutes(safeRoute?.distance_km)
  const traditionalMinutes = estimateMinutes(traditionalRoute?.distance_km)
  const riskReduction = safeRoute && traditionalRoute
    ? Math.max(
        0,
        Math.round(
          (1 - safeRoute.risk_score / Math.max(traditionalRoute.risk_score, 0.01)) * 100,
        ),
      )
    : null

  function handlePickFromMap(type, latlng) {
    setForm((currentForm) => ({
      ...currentForm,
      [`${type}Lat`]: latlng.lat.toFixed(6),
      [`${type}Lng`]: latlng.lng.toFixed(6),
    }))
    setSelectionMode(null)
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
        `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(
          query,
        )}`,
        { headers: { 'Accept-Language': 'es' } },
      )

      if (!response.ok) throw new Error('No se pudo buscar la dirección.')
      const results = await response.json()

      if (!results.length) {
        throw new Error('No se encontró una coincidencia para esa dirección.')
      }

      const result = results[0]
      setForm((currentForm) => ({
        ...currentForm,
        [`${type}Lat`]: Number(result.lat).toFixed(6),
        [`${type}Lng`]: Number(result.lon).toFixed(6),
      }))

      if (type === 'origin') {
        setOriginQuery(result.display_name)
      } else {
        setDestinationQuery(result.display_name)
      }

      setSelectionMode(null)
    } catch (requestError) {
      setGeoStatus((current) => ({
        ...current,
        [type]: requestError.message,
      }))
    } finally {
      setGeoLoading((current) => ({ ...current, [type]: false }))
    }
  }

  return (
    <main className={`app-shell ${isSidebarOpen ? 'sidebar-open' : 'sidebar-collapsed'}`}>
      <Sidebar isOpen={isSidebarOpen} onToggle={() => setIsSidebarOpen((current) => !current)} />

      <RoutePanel
        form={form}
        originQuery={originQuery}
        destinationQuery={destinationQuery}
        selectionMode={selectionMode}
        geoLoading={geoLoading}
        geoStatus={geoStatus}
        travelDate={travelDate}
        travelTime={travelTime}
        useCurrentTime={useCurrentTime}
        routePreference={routePreference}
        status={status}
        error={error}
        effectiveTurno={effectiveTurno}
        onOriginQueryChange={setOriginQuery}
        onDestinationQueryChange={setDestinationQuery}
        onSelectionModeChange={setSelectionMode}
        onPreferenceChange={handlePreferenceChange}
        onTravelDateChange={setTravelDate}
        onTravelTimeChange={setTravelTime}
        onUpdateField={updateField}
        onToggleCurrentTime={toggleCurrentTime}
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
            riskZones={riskZones}
            heatmapPoints={heatmapPoints}
            safeRoutePositions={safeRoutePositions}
            traditionalRoutePositions={traditionalRoutePositions}
          />
          <div
            className={`map-overlay ${isResultsMinimized ? 'is-minimized' : ''}`}
            aria-live="polite"
          >
            <button
              type="button"
              className="map-overlay-toggle"
              onClick={() => setIsResultsMinimized((current) => !current)}
              aria-label={isResultsMinimized ? 'Mostrar resultados' : 'Minimizar resultados'}
            >
              {isResultsMinimized ? '+' : '-'}
            </button>
            {!isResultsMinimized && (
              <InfoPanel
                safeRoute={safeRoute}
                traditionalRoute={traditionalRoute}
                safeMinutes={safeMinutes}
                traditionalMinutes={traditionalMinutes}
                riskReduction={riskReduction}
              />
            )}
          </div>
        </div>
        <BottomMetrics stats={systemStats} />
      </section>
    </main>
  )
}

export default App
