import { ShieldCheck } from 'lucide-react'

function RouteMetrics({ route, minutes }) {
  return (
    <div className="panel-stats grid">
      <div>
        <span>Distancia</span>
        <strong>{route.distance_km} km</strong>
      </div>
      <div>
        <span>Tiempo estimado</span>
        <strong>{minutes ? `${minutes} min` : '-'}</strong>
      </div>
      <div>
        <span>Riesgo total</span>
        <strong>{route.risk_total ?? '-'}</strong>
      </div>
      <div>
        <span>Riesgo promedio</span>
        <strong>{route.risk_average ?? route.risk_score}</strong>
      </div>
      <div>
        <span>Nivel de riesgo</span>
        <strong className={`risk-pill ${route.risk_level}`}>
          {route.risk_level}
        </strong>
      </div>
      <div>
        <span>Tramos críticos</span>
        <strong>{route.high_risk_segments ?? 0}</strong>
      </div>
    </div>
  )
}

function InfoPanel({
  safeRoute,
  traditionalRoute,
  safeMinutes,
  traditionalMinutes,
  riskReduction,
  routeMeta,
}) {
  return (
    <aside className="info-panel" aria-live="polite">
      <div className="panel-card">
        <div className="panel-card-header">
          <ShieldCheck size={17} />
          <h3>Ruta más segura</h3>
        </div>
        {safeRoute ? (
          <RouteMetrics route={safeRoute} minutes={safeMinutes} />
        ) : (
          <p className="muted">
            Calcula una ruta para comparar distancia, tiempo y exposición.
          </p>
        )}
        <div className="route-line safe" />
      </div>

      {traditionalRoute && (
        <div className="panel-card">
          <h3>Ruta más rápida</h3>
          <RouteMetrics route={traditionalRoute} minutes={traditionalMinutes} />
          <div className="route-line traditional" />
        </div>
      )}

      {riskReduction !== null && (
        <div className="panel-card highlight">
          <div className="panel-card-header">
            <ShieldCheck size={18} />
            <h3>Reducción de riesgo</h3>
          </div>
          <strong className="risk-reduction">{riskReduction}%</strong>
          <p>Calculada con riesgo acumulado por segmento.</p>
        </div>
      )}

      {routeMeta && (
        <div className="route-explanation">
          <strong>Modelo: {routeMeta.modelo_usado}</strong>
          <span>{routeMeta.mensaje}</span>
        </div>
      )}
    </aside>
  )
}

export default InfoPanel
