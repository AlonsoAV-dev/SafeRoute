import { BrainCircuit, Clock3, MapPin, Search, ShieldCheck, X } from 'lucide-react'
import { formatPeriod } from '../lib/presentation'

// Organiza origen, destino, preferencia y modelo, y muestra el estado de la consulta.
function RoutePanel({ originQuery, destinationQuery, selectionMode, geoLoading, geoStatus,
  routePreference, riskModel, models, modelsLoading, activeModel, status, error,
  onOriginQueryChange, onDestinationQueryChange, onClearLocation, onSelectionModeChange,
  onPreferenceChange, onRiskModelChange, onGeocode, onSubmit }) {
  const handleAddressKeyDown = (event, type) => {
    if (event.key !== 'Enter') return
    event.preventDefault()
    onGeocode(type)
  }
  const renderLocation = (number, type, title, value, onChange) => (
    <section className="route-step">
      <div className="step-heading">
        <span className="step-number">{number}</span>
        <label htmlFor={`location-${type}`}>{title}</label>
      </div>
      <div className="location-field">
        <span className={`dot ${type === 'origin' ? 'dot--green' : 'dot--red'}`} />
        <input id={`location-${type}`} value={value} onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => handleAddressKeyDown(event, type)} placeholder="Escribe una dirección" />
        {value && <button type="button" className="icon-button" onClick={() => onClearLocation(type)}
          aria-label={`Limpiar ${title.toLowerCase()}`}><X size={16} /></button>}
      </div>
      <div className="input-actions">
        <button type="button" className={selectionMode === type ? 'secondary-button is-active' : 'secondary-button'}
          onClick={() => onSelectionModeChange((current) => current === type ? null : type)}
          aria-pressed={selectionMode === type}>
          <MapPin size={15} />{selectionMode === type ? 'Selecciona en el mapa' : 'Elegir en el mapa'}
        </button>
        <button type="button" className="search-button" onClick={() => onGeocode(type)}
          disabled={geoLoading[type]} aria-label={`Buscar ${title.toLowerCase()}`}>
          <Search size={17} />
        </button>
      </div>
      {geoStatus[type] && <p className="inline-status" role="status">{geoStatus[type]}</p>}
    </section>
  )
  return (
    <aside className="route-panel" aria-label="Planificación de la ruta">
      <header className="panel-header">
        <div className="brand-mark"><ShieldCheck size={23} aria-hidden="true" /></div>
        <div><p className="panel-title">SafeRoute</p><h1>Planifica tu ruta</h1>
          <p className="panel-subtitle">Compara recorridos según distancia y riesgo estimado.</p></div>
      </header>
      <form onSubmit={onSubmit} className="route-form">
        {renderLocation(1, 'origin', 'Origen', originQuery, onOriginQueryChange)}
        {renderLocation(2, 'destination', 'Destino', destinationQuery, onDestinationQueryChange)}
        <section className="route-step">
          <div className="step-heading"><span className="step-number">3</span><strong>Preferencia de ruta</strong></div>
          <div className="toggle-buttons">
            <button type="button" className={routePreference === 'safe' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('safe')} aria-pressed={routePreference === 'safe'}>
              <ShieldCheck size={16} />Menor riesgo
            </button>
            <button type="button" className={routePreference === 'fast' ? 'chip chip--active' : 'chip'}
              onClick={() => onPreferenceChange('fast')} aria-pressed={routePreference === 'fast'}>
              <Clock3 size={16} />Menor distancia
            </button>
          </div>
        </section>
        <section className="route-step">
          <div className="step-heading"><span className="step-number">4</span><label htmlFor="risk-model">Modelo predictivo</label></div>
          <div className="model-selector">
            <BrainCircuit size={18} aria-hidden="true" />
            <select id="risk-model" value={riskModel} onChange={(event) => onRiskModelChange(event.target.value)}
              disabled={modelsLoading || status === 'loading'} aria-describedby="model-period">
              {models.map((model) => <option key={model.key} value={model.key} disabled={!model.available}>
                {model.name}{!modelsLoading && !model.available ? ' · No disponible' : ''}
              </option>)}
            </select>
          </div>
          <p className="model-period" id="model-period">
            {modelsLoading ? 'Consultando modelos…' : activeModel?.available
              ? `Periodo disponible: ${formatPeriod(activeModel.prediction_period)} · Riesgo mensual`
              : 'El modelo necesita predicciones disponibles para calcular rutas.'}
          </p>
        </section>
        {selectionMode && <p className="selection-hint" role="status">
          Selecciona en el mapa el {selectionMode === 'origin' ? 'origen' : 'destino'}.</p>}
        <button type="submit" className="primary-button" aria-busy={status === 'loading'}
          disabled={status === 'loading' || modelsLoading || !activeModel?.available}>
          <ShieldCheck size={18} />{status === 'loading' ? 'Calculando ruta…' : 'Calcular ruta'}
        </button>
        {error && <p className="error-message" role="alert">{error}</p>}
      </form>
      <p className="privacy-note">El riesgo estimado orienta la elección del recorrido. No garantiza la seguridad de una ruta.</p>
    </aside>
  )
}

export default RoutePanel
