import { useEffect, useMemo, useState } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  useMap,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const DEFAULT_FORM = {
  originLat: '-12.0464',
  originLng: '-77.0428',
  destinationLat: '-12.0905',
  destinationLng: '-77.0068',
  turno: 'noche',
  safetyWeight: 4,
}

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

function App() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [routeData, setRouteData] = useState(null)
  const [riskZones, setRiskZones] = useState([])
  const [crimePoints, setCrimePoints] = useState([])
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')

  const routePositions = useMemo(
    () => routeData?.route.map((point) => [point.lat, point.lng]) ?? [],
    [routeData],
  )

  const origin = [Number(form.originLat), Number(form.originLng)]
  const destination = [Number(form.destinationLat), Number(form.destinationLng)]

  useEffect(() => {
    async function loadMapData() {
      try {
        const [zonesResponse, pointsResponse] = await Promise.all([
          fetch(`${API_URL}/risk-zones?turno=${form.turno}`),
          fetch(`${API_URL}/crime-points`),
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
  }, [form.turno])

  async function handleSubmit(event) {
    event.preventDefault()
    setStatus('loading')
    setError('')

    try {
      const response = await fetch(`${API_URL}/route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: { lat: Number(form.originLat), lng: Number(form.originLng) },
          destination: {
            lat: Number(form.destinationLat),
            lng: Number(form.destinationLng),
          },
          turno: form.turno,
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

          <label>
            Turno
            <select name="turno" value={form.turno} onChange={updateField}>
              <option value="manana">Mañana</option>
              <option value="tarde">Tarde</option>
              <option value="noche">Noche</option>
              <option value="madrugada">Madrugada</option>
            </select>
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
        <MapContainer center={origin} zoom={13} className="map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <CircleMarker center={origin} radius={9} pathOptions={{ color: '#0f766e' }}>
            <Popup>Origen</Popup>
          </CircleMarker>
          <CircleMarker center={destination} radius={9} pathOptions={{ color: '#2563eb' }}>
            <Popup>Destino</Popup>
          </CircleMarker>

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
