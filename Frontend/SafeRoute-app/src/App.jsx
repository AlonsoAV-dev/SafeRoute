import { useEffect, useMemo, useState } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const DEFAULT_FORM = {
  originLat: '',
  originLng: '',
  destinationLat: '',
  destinationLng: '',
  turno: 'noche',
  safetyWeight: 4,
}

const zoneColors = {
  bajo: '#15803d',
  medio: '#ca8a04',
  alto: '#dc2626',
}

const clusterColors = ['#ef4444', '#f59e0b', '#2563eb', '#7c3aed', '#0891b2', '#db2777']

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

function App() {
  const [form, setForm] = useState(() => ({
    ...DEFAULT_FORM,
    turno: turnoByHour(new Date().getHours()),
  }))
  const [originQuery, setOriginQuery] = useState('')
  const [destinationQuery, setDestinationQuery] = useState('')
  const [routeData, setRouteData] = useState(null)
  const [riskZones, setRiskZones] = useState([])
  const [crimePoints, setCrimePoints] = useState([])
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [selectionMode, setSelectionMode] = useState(null)
  const [geoStatus, setGeoStatus] = useState({ origin: '', destination: '' })
  const [geoLoading, setGeoLoading] = useState({ origin: false, destination: false })
  const [mapCenter, setMapCenter] = useState(LIMA_METRO_CENTER)
  const [useCurrentTime, setUseCurrentTime] = useState(true)

  const routePositions = useMemo(
    () => routeData?.route.map((point) => [point.lat, point.lng]) ?? [],
    [routeData],
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
        const [zonesResponse, pointsResponse] = await Promise.all([
          fetch(`${API_URL}/risk-zones?turno=${effectiveTurno}`),
          fetch(`${API_URL}/crime-points?turno=${effectiveTurno}`),
        ])

        if (!zonesResponse.ok) throw new Error('No se pudieron cargar las zonas.')
        if (!pointsResponse.ok) throw new Error('No se pudieron cargar los delitos.')

        setRiskZones(await zonesResponse.json())
        setCrimePoints(await pointsResponse.json())
      } catch (requestError) {
        setError(requestError.message)
      }
    }

    loadMapData()
  }, [effectiveTurno])

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
      const response = await fetch(`${API_URL}/route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: { lat: origin[0], lng: origin[1] },
          destination: {
            lat: destination[0],
            lng: destination[1],
          },
          turno: effectiveTurno,
          safety_weight: Number(form.safetyWeight),
        }),
      })

      if (!response.ok) throw new Error('No se pudo generar la ruta.')
      setRouteData(await response.json())
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
    <main className="app-shell">
      <aside className="control-panel">
        <div>
          <p className="eyebrow">SafeRoute Lima</p>
          <h1>Recomendador de rutas seguras</h1>
        </div>

        <form onSubmit={handleSubmit} className="route-form">
          <fieldset>
            <legend>Origen</legend>
            <label className="wide-field">
              Buscar dirección
              <input
                name="originQuery"
                value={originQuery}
                onChange={(event) => setOriginQuery(event.target.value)}
                placeholder="Ej. Av. Arequipa 123, Lima"
              />
            </label>
            <div className="action-row wide-field">
              <button
                type="button"
                className="action-button"
                onClick={() => handleGeocode('origin')}
                disabled={geoLoading.origin}
              >
                {geoLoading.origin ? 'Buscando...' : 'Buscar'}
              </button>
              <button
                type="button"
                className={
                  selectionMode === 'origin'
                    ? 'action-button action-button--active'
                    : 'action-button'
                }
                onClick={() =>
                  setSelectionMode((current) => (current === 'origin' ? null : 'origin'))
                }
              >
                {selectionMode === 'origin' ? 'Seleccionando...' : 'Elegir en mapa'}
              </button>
            </div>
            {geoStatus.origin && (
              <p className="geo-status wide-field">{geoStatus.origin}</p>
            )}
            <label>
              Latitud
              <input
                name="originLat"
                value={form.originLat}
                onChange={updateField}
                inputMode="decimal"
              />
            </label>
            <label>
              Longitud
              <input
                name="originLng"
                value={form.originLng}
                onChange={updateField}
                inputMode="decimal"
              />
            </label>
          </fieldset>

          <fieldset>
            <legend>Destino</legend>
            <label className="wide-field">
              Buscar dirección
              <input
                name="destinationQuery"
                value={destinationQuery}
                onChange={(event) => setDestinationQuery(event.target.value)}
                placeholder="Ej. Plaza San Martín, Lima"
              />
            </label>
            <div className="action-row wide-field">
              <button
                type="button"
                className="action-button"
                onClick={() => handleGeocode('destination')}
                disabled={geoLoading.destination}
              >
                {geoLoading.destination ? 'Buscando...' : 'Buscar'}
              </button>
              <button
                type="button"
                className={
                  selectionMode === 'destination'
                    ? 'action-button action-button--active'
                    : 'action-button'
                }
                onClick={() =>
                  setSelectionMode((current) =>
                    current === 'destination' ? null : 'destination',
                  )
                }
              >
                {selectionMode === 'destination'
                  ? 'Seleccionando...'
                  : 'Elegir en mapa'}
              </button>
            </div>
            {geoStatus.destination && (
              <p className="geo-status wide-field">{geoStatus.destination}</p>
            )}
            <label>
              Latitud
              <input
                name="destinationLat"
                value={form.destinationLat}
                onChange={updateField}
                inputMode="decimal"
              />
            </label>
            <label>
              Longitud
              <input
                name="destinationLng"
                value={form.destinationLng}
                onChange={updateField}
                inputMode="decimal"
              />
            </label>
          </fieldset>

          {selectionMode && (
            <p className="selection-hint">
              Haz clic en el mapa para fijar el{' '}
              {selectionMode === 'origin' ? 'origen' : 'destino'}.
            </p>
          )}

          <label>
            Turno
            <select
              name="turno"
              value={effectiveTurno}
              onChange={updateField}
              disabled={useCurrentTime}
            >
              <option value="manana">Mañana</option>
              <option value="tarde">Tarde</option>
              <option value="noche">Noche</option>
              <option value="madrugada">Madrugada</option>
            </select>
          </label>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={useCurrentTime}
              onChange={toggleCurrentTime}
            />
            Usar turno según hora actual
          </label>

          <label>
            Peso de seguridad: {form.safetyWeight}
            <input
              type="range"
              name="safetyWeight"
              min="0"
              max="10"
              step="1"
              value={form.safetyWeight}
              onChange={updateField}
            />
          </label>

          <button type="submit" disabled={status === 'loading'}>
            {status === 'loading' ? 'Calculando...' : 'Generar ruta'}
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}

        <section className="summary">
          <h2>Resultado</h2>
          {routeData ? (
            <dl>
              <div>
                <dt>Distancia</dt>
                <dd>{routeData.distance_km} km</dd>
              </div>
              <div>
                <dt>Riesgo</dt>
                <dd className={`risk-pill ${routeData.risk_level}`}>
                  {routeData.risk_level}
                </dd>
              </div>
              <div>
                <dt>Puntaje</dt>
                <dd>{routeData.risk_score}</dd>
              </div>
            </dl>
          ) : (
            <p className="muted">Genera una ruta para ver el resumen.</p>
          )}
        </section>

        <section className="map-layers">
          <h2>Capas</h2>
          <div className="layer-row">
            <span className="layer-dot crime-dot"></span>
            <span>Delitos validos</span>
            <strong>{crimePoints.length}</strong>
          </div>
          <div className="layer-row">
            <span className="layer-dot cluster-dot"></span>
            <span>Clusters K-Means</span>
            <strong>{riskZones.length}</strong>
          </div>
        </section>
      </aside>

      <section className="map-area" aria-label="Mapa de rutas seguras">
        <MapContainer center={mapCenter} zoom={13} className="map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapCenterUpdater center={mapCenter} />

          <MapClickPicker selectionMode={selectionMode} onPick={handlePickFromMap} />

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

          {routePositions.length > 1 && (
            <>
              <Polyline
                positions={routePositions}
                pathOptions={{ color: '#0f766e', weight: 6, opacity: 0.9 }}
              />
              <FitRoute route={routePositions} />
            </>
          )}
        </MapContainer>
      </section>
    </main>
  )
}

export default App
