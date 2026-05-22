import { ChevronRight, Shield } from 'lucide-react'

function InfoPanel({ safeRoute, traditionalRoute, safeMinutes, traditionalMinutes, riskReduction }) {
  return (
    <aside className="info-panel">
      <div className="panel-card">
        <div className="panel-card-header">
          <Shield size={16} />
          <h3>Ruta recomendada</h3>
        </div>
        {safeRoute ? (
          <div className="panel-stats grid">
            <div>
              <span>Distancia</span>
              <strong>{safeRoute.distance_km} km</strong>
            </div>
            <div>
              <span>Tiempo estimado</span>
              <strong>{safeMinutes ? `${safeMinutes} min` : '-'}</strong>
            </div>
            <div>
              <span>Riesgo promedio</span>
              <strong className={`risk-pill ${safeRoute.risk_level}`}>
                {safeRoute.risk_score} ({safeRoute.risk_level})
              </strong>
            </div>
          </div>
        ) : (
          <p className="muted">Genera una ruta para ver resultados.</p>
        )}
        <div className="route-line safe" />
      </div>

      <div className="panel-card">
        <h3>Ruta tradicional</h3>
        {traditionalRoute ? (
          <div className="panel-stats grid">
            <div>
              <span>Distancia</span>
              <strong>{traditionalRoute.distance_km} km</strong>
            </div>
            <div>
              <span>Tiempo estimado</span>
              <strong>{traditionalMinutes ? `${traditionalMinutes} min` : '-'}</strong>
            </div>
            <div>
              <span>Riesgo promedio</span>
              <strong className={`risk-pill ${traditionalRoute.risk_level}`}>
                {traditionalRoute.risk_score} ({traditionalRoute.risk_level})
              </strong>
            </div>
          </div>
        ) : (
          <p className="muted">Sin datos todavía.</p>
        )}
        <div className="route-line traditional" />
      </div>

      <div className="panel-card highlight">
        <div className="panel-card-header">
          <Shield size={18} />
          <h3>Reducción de riesgo</h3>
        </div>
        <strong className="risk-reduction">
          {riskReduction !== null ? `${riskReduction}%` : '--'}
        </strong>
        <p>Comparado con la ruta tradicional</p>
      </div>

      <button type="button" className="outline-button">
        Ver indicaciones paso a paso
        <ChevronRight size={16} />
      </button>
    </aside>
  )
}

export default InfoPanel
