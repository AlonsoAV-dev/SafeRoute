import { Clock3, MapPin, Search, ShieldCheck, X } from 'lucide-react'

function RoutePanel({
  originQuery,
  destinationQuery,
  selectionMode,
  geoLoading,
  geoStatus,
  routePreference,
  riskModelSelection,
  status,
  error,
  onOriginQueryChange,
  onDestinationQueryChange,
  onClearLocation,
  onSelectionModeChange,
  onPreferenceChange,
  onRiskModelChange,
  onGeocode,
  onSubmit,
}) {
  const handleAddressKeyDown = (event, type) => {
    if (event.key !== 'Enter') return
    event.preventDefault()
    onGeocode(type)
  }

  const renderLocationStep = ({
    number,
    type,
    title,
    value,
    onChange,
    colorClass,
  }) => (
    <section className="route-step">
      <div className="step-heading">
        <span className="step-number">{number}</span>
        <strong>{title}</strong>
      </div>
      <div className="location-field">
        <span className={`dot ${colorClass}`} />
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => handleAddressKeyDown(event, type)}
          placeholder="Escribe una dirección"
          aria-label={title}
        />
        {value && (
          <button
            type="button"
            className="icon-button"
            onClick={() => onClearLocation(type)}
            aria-label={`Limpiar ${title.toLowerCase()}`}
          >
            <X size={16} />
          </button>
        )}
      </div>
      <div className="input-actions">
        <button
          type="button"
          className={selectionMode === type ? 'secondary-button is-active' : 'secondary-button'}
          onClick={() =>
            onSelectionModeChange((current) => (current === type ? null : type))
          }
        >
          <MapPin size={15} />
          {selectionMode === type ? 'Haz clic en el mapa' : 'Elegir en mapa'}
        </button>
        <button
          type="button"
          className="search-button"
          onClick={() => onGeocode(type)}
          disabled={geoLoading[type]}
          aria-label={`Buscar ${title.toLowerCase()}`}
        >
          <Search size={16} />
        </button>
      </div>
      {geoStatus[type] && <p className="inline-status">{geoStatus[type]}</p>}
    </section>
  )

  return (
    <aside className="route-panel">
      <header className="panel-header">
        <div className="brand-mark">
          <ShieldCheck size={22} />
        </div>
        <div>
          <p className="panel-title">SafeRoute</p>
          <h1>¿A dónde vamos?</h1>
          <p className="panel-subtitle">Elige dos puntos y encuentra la mejor ruta.</p>
        </div>
      </header>

      <form onSubmit={onSubmit} className="route-form">
        {renderLocationStep({
          number: 1,
          type: 'origin',
          title: 'Punto de partida',
          value: originQuery,
          onChange: onOriginQueryChange,
          colorClass: 'dot--green',
        })}
        {renderLocationStep({
          number: 2,
          type: 'destination',
          title: 'Punto de llegada',
          value: destinationQuery,
          onChange: onDestinationQueryChange,
          colorClass: 'dot--red',
        })}

        <section className="route-step">
          <div className="step-heading">
            <span className="step-number">3</span>
            <strong>Preferencia de ruta</strong>
          </div>
          <div className="toggle-buttons">
            <button
              type="button"
              className={routePreference === 'safe' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('safe')}
            >
              <ShieldCheck size={17} />
              Ruta más segura
            </button>
            <button
              type="button"
              className={routePreference === 'fast' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('fast')}
            >
              <Clock3 size={17} />
              Ruta más rápida
            </button>
          </div>
        </section>

        <label className="model-selector">
          <span>Modelo de riesgo</span>
          <select
            value={riskModelSelection}
            onChange={(event) => onRiskModelChange(event.target.value)}
          >
            <option value="auto">Automático</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
          <small>
            Automático selecciona el modelo con mejores métricas de validación.
          </small>
        </label>

        {selectionMode && (
          <p className="selection-hint">
            Haz clic en el mapa para fijar el{' '}
            {selectionMode === 'origin' ? 'punto de partida' : 'punto de llegada'}.
          </p>
        )}

        <button type="submit" className="primary-button" disabled={status === 'loading'}>
          <ShieldCheck size={18} />
          {status === 'loading' ? 'Calculando ruta...' : 'Calcular Ruta'}
        </button>
        {error && <p className="error-message">{error}</p>}
      </form>

      <p className="privacy-note">
        Las rutas consideran datos de riesgo sin mostrar información técnica innecesaria.
      </p>
    </aside>
  )
}

export default RoutePanel
