import { Calendar, Clock, Shield, Timer, X } from 'lucide-react'

function RoutePanel({
  form,
  originQuery,
  destinationQuery,
  selectionMode,
  geoLoading,
  geoStatus,
  travelDate,
  travelTime,
  useCurrentTime,
  routePreference,
  status,
  error,
  effectiveTurno,
  onOriginQueryChange,
  onDestinationQueryChange,
  onSelectionModeChange,
  onPreferenceChange,
  onTravelDateChange,
  onTravelTimeChange,
  onUpdateField,
  onToggleCurrentTime,
  onGeocode,
  onSubmit,
}) {
  const alphaValue = Number(form.safetyWeight || 0).toFixed(2)

  return (
    <aside className="route-panel">
      <header className="panel-header">
        <p className="panel-title">Nueva ruta</p>
        <h2>Planifica tu viaje</h2>
      </header>
      <form onSubmit={onSubmit} className="route-form">
        <div className="input-group">
          <span className="input-label">Punto de origen</span>
          <div className="input-field">
            <span className="dot dot--green" />
            <input
              name="originQuery"
              value={originQuery}
              onChange={(event) => onOriginQueryChange(event.target.value)}
              placeholder="Av. Arequipa 1234, Lima"
            />
            <button
              type="button"
              className="icon-button"
              onClick={() => onOriginQueryChange('')}
              aria-label="Limpiar origen"
            >
              <X size={14} />
            </button>
          </div>
          <div className="input-actions">
            <button
              type="button"
              className={selectionMode === 'origin' ? 'ghost-button is-active' : 'ghost-button'}
              onClick={() =>
                onSelectionModeChange((current) => (current === 'origin' ? null : 'origin'))
              }
            >
              {selectionMode === 'origin' ? 'Seleccionando...' : 'Elegir en mapa'}
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={() => onGeocode('origin')}
              disabled={geoLoading.origin}
            >
              {geoLoading.origin ? '...' : 'Buscar'}
            </button>
          </div>
          {geoStatus.origin && <p className="inline-status">{geoStatus.origin}</p>}
        </div>

        <div className="input-group">
          <span className="input-label">Punto de destino</span>
          <div className="input-field">
            <span className="dot dot--red" />
            <input
              name="destinationQuery"
              value={destinationQuery}
              onChange={(event) => onDestinationQueryChange(event.target.value)}
              placeholder="Universidad de Lima"
            />
            <button
              type="button"
              className="icon-button"
              onClick={() => onDestinationQueryChange('')}
              aria-label="Limpiar destino"
            >
              <X size={14} />
            </button>
          </div>
          <div className="input-actions">
            <button
              type="button"
              className={selectionMode === 'destination' ? 'ghost-button is-active' : 'ghost-button'}
              onClick={() =>
                onSelectionModeChange((current) =>
                  current === 'destination' ? null : 'destination',
                )
              }
            >
              {selectionMode === 'destination' ? 'Seleccionando...' : 'Elegir en mapa'}
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={() => onGeocode('destination')}
              disabled={geoLoading.destination}
            >
              {geoLoading.destination ? '...' : 'Buscar'}
            </button>
          </div>
          {geoStatus.destination && <p className="inline-status">{geoStatus.destination}</p>}
        </div>

        <div className="date-time-row">
          <label className="input-stack">
            <span>Fecha</span>
            <div className="input-field">
              <Calendar size={14} />
              <input type="date" value={travelDate} onChange={(event) => onTravelDateChange(event.target.value)} />
            </div>
          </label>
          <label className="input-stack">
            <span>Hora</span>
            <div className="input-field">
              <Timer size={14} />
              <input type="time" value={travelTime} onChange={(event) => onTravelTimeChange(event.target.value)} />
            </div>
          </label>
        </div>

        <label className="input-stack">
          <span>Turno</span>
          <select
            name="turno"
            value={effectiveTurno}
            onChange={onUpdateField}
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
            onChange={onToggleCurrentTime}
          />
          Usar turno según hora actual
        </label>

        <div className="preference">
          <p>Preferencia de ruta</p>
          <div className="toggle-buttons">
            <button
              type="button"
              className={routePreference === 'safe' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('safe')}
            >
              <Shield size={14} /> Ruta más segura
            </button>
            <button
              type="button"
              className={routePreference === 'fast' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('fast')}
            >
              <Clock size={14} /> Ruta más rápida
            </button>
          </div>
        </div>

        <div className="slider-block">
          <div className="slider-header">
            <span>Peso de seguridad (α)</span>
            <span className="alpha-badge">{alphaValue}</span>
          </div>
          <input
            type="range"
            name="safetyWeight"
            min="0"
            max="1"
            step="0.05"
            value={form.safetyWeight}
            onChange={onUpdateField}
          />
          <div className="slider-labels">
            <span>Priorizar rapidez</span>
            <span>Priorizar seguridad</span>
          </div>
        </div>

        {selectionMode && (
          <p className="selection-hint">
            Haz clic en el mapa para fijar el{' '}
            {selectionMode === 'origin' ? 'origen' : 'destino'}.
          </p>
        )}

        <button type="submit" className="primary-button" disabled={status === 'loading'}>
          <Shield size={16} />
          {status === 'loading' ? 'Calculando...' : 'Buscar ruta segura'}
        </button>
      </form>
      <div className="risk-legend">
        <h4>Niveles de riesgo</h4>
        <div>
          <span className="risk-dot risk-high" /> Alto riesgo 0.7–1.0
        </div>
        <div>
          <span className="risk-dot risk-medium" /> Medio riesgo 0.3–0.7
        </div>
        <div>
          <span className="risk-dot risk-low" /> Bajo riesgo 0.0–0.3
        </div>
      </div>
      {error && <p className="error-message">{error}</p>}
    </aside>
  )
}

export default RoutePanel
