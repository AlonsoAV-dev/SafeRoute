import { ShieldCheck } from 'lucide-react'

function RouteMetrics({ route, minutes }) {
  const averageRisk = route.risk_average ?? route.risk_score
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
        <span>Exposición acumulada</span>
        <strong>{route.risk_total != null ? route.risk_total.toFixed(3) : '-'}</strong>
      </div>
      <div>
        <span>Riesgo promedio</span>
        <strong>{averageRisk != null ? `${(averageRisk * 100).toFixed(1)}%` : '-'}</strong>
      </div>
      <div>
        <span>Nivel de riesgo</span>
        <strong className={`risk-pill ${route.risk_level}`}>
          {route.risk_level}
        </strong>
      </div>
      <div>
        <span>Tramos de riesgo alto</span>
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
  const sameRoute = routeMeta?.misma_ruta === true
  const riskModeLabel = {
    predicted: 'Predicción Random Forest',
    historical: 'Historial delictivo',
    hybrid: 'Combinado (70% histórico + 30% RF)',
  }[routeMeta?.modo_riesgo]
  return (
    <aside className="info-panel" aria-live="polite">
      <div className="panel-card">
        <div className="panel-card-header">
          <ShieldCheck size={17} />
          <h3>{sameRoute ? 'Ruta recomendada' : 'Ruta más segura'}</h3>
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

      {traditionalRoute && !sameRoute && (
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
            <h3>{sameRoute ? 'Resultado de la comparación' : 'Reducción de riesgo'}</h3>
          </div>
          {sameRoute ? (
            <>
              <strong className="comparison-status">Sin alternativa de menor riesgo</strong>
              <p>La ruta más corta también obtuvo el menor costo disponible.</p>
            </>
          ) : (
            <>
              <strong className="risk-reduction">{riskReduction}%</strong>
              <p>Calculada con exposición ponderada por distancia.</p>
            </>
          )}
        </div>
      )}

      {routeMeta && (
        <div className="route-explanation">
          <strong>Modelo: {routeMeta.modelo_usado}</strong>
          <span>Criterio de ruta: {riskModeLabel ?? 'No disponible'}</span>
          <span>Periodo estimado: {routeMeta.periodo_prediccion}</span>
          <span>{routeMeta.mensaje}</span>
        </div>
      )}
    </aside>
  )
}

export default InfoPanel
