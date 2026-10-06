import { ArrowDown, Clock3, Route, ShieldCheck } from 'lucide-react'
import { RISK_LEVELS, formatPeriod } from '../lib/presentation'

const formatNumber = (value, decimals = 1) => Number(value).toLocaleString('es-PE', {
  maximumFractionDigits: decimals,
})

// Presenta distancia, tiempo e índice de riesgo de una alternativa con su exposición comparativa.
function RouteCard({ route, minutes, kind, selected, reduction, sameRoute, onShow }) {
  const risk = RISK_LEVELS[route.risk_level] ?? RISK_LEVELS.bajo
  const score = Math.max(0, Math.min(100, (route.risk_average ?? route.risk_score) * 100))
  const safe = kind === 'safe'
  const Icon = safe ? ShieldCheck : Route
  return <article className={`panel-card ${safe ? 'is-recommended' : ''}`}>
    <div className="panel-card-header">
      <div className="route-card-title"><Icon size={25} aria-hidden="true" />
        <div><h3>{safe ? 'Ruta recomendada' : 'Ruta alternativa'}</h3>
          <p>{sameRoute ? 'Recorrido compartido' : safe ? 'Menor riesgo' : 'Más corta'}</p></div>
      </div>
      {safe && !sameRoute && reduction > 0 && <span className="exposure-badge">
        <ArrowDown size={17} aria-hidden="true" />{formatNumber(reduction)} % menos exposición
      </span>}
    </div>
    <dl className="panel-stats navigation-stats">
      <div><dd>{formatNumber(route.distance_km, 2)} <small>km</small></dd><dt>Distancia</dt></div>
      <div><dd>{minutes != null ? formatNumber(minutes) : '—'} <small>min</small></dd><dt>Tiempo estimado</dt></div>
      <div><dd>{formatNumber(score)} <small>/ 100</small></dd><dt>Índice de riesgo</dt></div>
    </dl>
    <div className="route-risk-meter" role="meter" aria-label={`Índice de riesgo de la ruta ${safe ? 'recomendada' : 'alternativa'}`}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={score} aria-valuetext={`${formatNumber(score)} de 100; riesgo ${risk.label.toLowerCase()}`}>
      <span style={{ left: `${score}%`, background: risk.color }} />
    </div>
    <div className="route-risk-labels"><span>Bajo</span><span>Alto</span></div>
    <details className="route-detail"><summary>Detalle del recorrido</summary>
      <span>{formatNumber(route.red_distance_m ?? 0, 0)} m en rojo · {formatNumber(route.orange_distance_m ?? 0, 0)} m en naranja</span>
      <span>Exposición al campo del mapa: {formatNumber(route.risk_total, 3)}</span>
      <span>Score original del modelo por tramo, promedio: {formatNumber((route.model_risk_average ?? 0) * 100)} / 100</span>
    </details>
    {!sameRoute && <button type="button" className={`route-view-button ${selected ? 'is-active' : ''}`}
      aria-pressed={selected} onClick={onShow}>{selected ? 'Viendo esta ruta' : 'Ver esta ruta por separado'}</button>}
  </article>
}

// Agrupa ambas rutas y explica sus diferencias o los motivos de que sus recorridos coincidan.
function InfoPanel({ safeRoute, traditionalRoute, safeMinutes, traditionalMinutes,
  riskReduction, routeMeta, routeView, onRouteViewChange }) {
  if (!safeRoute) return <div className="results-empty" id="route-results" aria-live="polite">
    <Route size={21} aria-hidden="true" /><p>Elige un origen y un destino para comparar tus rutas.</p>
  </div>
  const sameRoute = routeMeta?.misma_ruta === true
  const cards = [
    { key: 'safe', route: safeRoute, minutes: safeMinutes },
    ...(traditionalRoute ? [{ key: 'fast', route: traditionalRoute, minutes: traditionalMinutes }] : []),
  ]
  return <div className="info-panel" id="route-results" aria-live="polite">
    {routeMeta?.hotspot_policy && <div className={`hotspot-policy-note ${safeRoute.red_distance_m >= 0.01 ? 'has-crossing' : ''}`} role="status">
      <strong>{safeRoute.red_distance_m < 0.01 ? 'Centros rojos evitados' : 'Cruce de rojo condicionado'}</strong>
      {routeMeta.hotspot_policy.crossing_reasons?.length
        ? routeMeta.hotspot_policy.crossing_reasons.map((reason) => <span key={reason}>{reason}</span>)
        : <span>{safeRoute.orange_distance_m < 0.01 ? 'El recorrido también evita las zonas naranjas.' : 'Se prioriza evitar rojo; permanece exposición a naranja.'}</span>}
    </div>}
    <div className="route-cards">
      {cards.map(({ key, ...card }) => <RouteCard key={key} kind={key} {...card}
        selected={routeView === key} reduction={riskReduction} sameRoute={sameRoute}
        onShow={() => onRouteViewChange(key)} />)}
    </div>
    {sameRoute ? <p className="shared-route-note">{routeMeta?.mensaje}</p>
      : traditionalRoute && <button type="button" className="route-view-button compare-routes"
        aria-pressed={routeView === 'both'} onClick={() => onRouteViewChange('both')}>Comparar ambas rutas</button>}
    <div className="result-context">
      <span>{routeMeta?.modelo_usado} · {formatPeriod(routeMeta?.periodo_prediccion)}</span>
      <span>Índice del campo espacial mostrado en el heatmap; no es probabilidad de sufrir un delito. Prioridad: evitar rojo, luego naranja, después exposición y distancia.</span>
      {cards.some((card) => card.route.access_distance_m > 2) && <span>Los accesos punteados a la red vial no tienen riesgo evaluado.</span>}
    </div>
  </div>
}

export default InfoPanel
