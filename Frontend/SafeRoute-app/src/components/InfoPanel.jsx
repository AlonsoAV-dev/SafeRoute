function InfoPanel({ safeRoute, traditionalRoute, safeMinutes, traditionalMinutes, riskReduction }) {
  return (
    <aside className="info-panel">
      <div className="panel-card">
        <h3>Ruta recomendada</h3>
        {safeRoute ? (
          <div className="panel-stats">
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
      </div>
      <div className="panel-card">
        <h3>Ruta tradicional</h3>
        {traditionalRoute ? (
          <div className="panel-stats">
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
      </div>
      <div className="panel-card highlight">
        <h3>Reducción de riesgo</h3>
        <strong>{riskReduction !== null ? `${riskReduction}%` : '--'}</strong>
        <p>Comparado con la ruta tradicional</p>
      </div>
    </aside>
  )
}

export default InfoPanel
