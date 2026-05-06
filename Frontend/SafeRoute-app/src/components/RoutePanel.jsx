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
  return (
    <div className="route-panel">
      <header>
        <p>Nueva ruta</p>
        <h2>Planifica tu viaje</h2>
      </header>
      <form onSubmit={onSubmit} className="route-form">
        <fieldset>
          <legend>Punto de origen</legend>
          <label className="location-input">
            <span className="dot dot--green"></span>
            <input
              name="originQuery"
              value={originQuery}
              onChange={(event) => onOriginQueryChange(event.target.value)}
              placeholder="Buscar punto de origen"
            />
            <button
              type="button"
              onClick={() => onGeocode('origin')}
              disabled={geoLoading.origin}
            >
              {geoLoading.origin ? '...' : 'Buscar'}
            </button>
          </label>
          <div className="action-row wide-field">
            <button
              type="button"
              className={
                selectionMode === 'origin'
                  ? 'action-button action-button--active'
                  : 'action-button'
              }
              onClick={() =>
                onSelectionModeChange((current) => (current === 'origin' ? null : 'origin'))
              }
            >
              {selectionMode === 'origin' ? 'Seleccionando...' : 'Elegir en mapa'}
            </button>
          </div>
          {geoStatus.origin && <p className="geo-status wide-field">{geoStatus.origin}</p>}
        </fieldset>

        <fieldset>
          <legend>Punto de destino</legend>
          <label className="location-input">
            <span className="dot dot--red"></span>
            <input
              name="destinationQuery"
              value={destinationQuery}
              onChange={(event) => onDestinationQueryChange(event.target.value)}
              placeholder="Buscar punto de destino"
            />
            <button
              type="button"
              onClick={() => onGeocode('destination')}
              disabled={geoLoading.destination}
            >
              {geoLoading.destination ? '...' : 'Buscar'}
            </button>
          </label>
          <div className="action-row wide-field">
            <button
              type="button"
              className={
                selectionMode === 'destination'
                  ? 'action-button action-button--active'
                  : 'action-button'
              }
              onClick={() =>
                onSelectionModeChange((current) =>
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
        </fieldset>

        <div className="two-columns">
          <label>
            Fecha
            <input type="date" value={travelDate} onChange={(event) => onTravelDateChange(event.target.value)} />
          </label>
          <label>
            Hora
            <input type="time" value={travelTime} onChange={(event) => onTravelTimeChange(event.target.value)} />
          </label>
        </div>

        <label>
          Turno
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
              Ruta más segura
            </button>
            <button
              type="button"
              className={routePreference === 'fast' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('fast')}
            >
              Ruta más rápida
            </button>
          </div>
        </div>

        <label>
          Peso de seguridad (α): {form.safetyWeight}
          <input
            type="range"
            name="safetyWeight"
            min="0"
            max="10"
            step="1"
            value={form.safetyWeight}
            onChange={onUpdateField}
          />
        </label>

        {selectionMode && (
          <p className="selection-hint">
            Haz clic en el mapa para fijar el{' '}
            {selectionMode === 'origin' ? 'origen' : 'destino'}.
          </p>
        )}

        <button type="submit" disabled={status === 'loading'}>
          {status === 'loading' ? 'Calculando...' : 'Buscar ruta segura'}
        </button>
      </form>
      {error && <p className="error-message">{error}</p>}
    </div>
  )
}

export default RoutePanel
