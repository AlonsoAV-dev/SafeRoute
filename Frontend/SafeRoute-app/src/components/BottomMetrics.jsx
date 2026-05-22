import { Brain, MapPin, Sparkles } from 'lucide-react'

function BottomMetrics({ stats }) {
  const accuracy = stats?.model_accuracy ? `${(stats.model_accuracy * 100).toFixed(1)}%` : '0.0%'
  const zonesCount = stats?.zones_count ?? 0
  const calcTime = stats?.calc_time_ms ? `${stats.calc_time_ms} ms` : '0 ms'

  return (
    <section className="bottom-metrics">
      <div className="metric-card">
        <div className="metric-icon">
          <Brain size={16} />
        </div>
        <div>
          <h4>Prediccion de riesgo</h4>
          <p>Modelo Random Forest</p>
          <strong>Precision: {accuracy}</strong>
        </div>
      </div>
      <div className="metric-card">
        <div className="metric-icon">
          <MapPin size={16} />
        </div>
        <div>
          <h4>Zonas detectadas</h4>
          <p>Clustering K-Means</p>
          <strong>Activas: {zonesCount}</strong>
        </div>
      </div>
      <div className="metric-card">
        <div className="metric-icon">
          <Sparkles size={16} />
        </div>
        <div>
          <h4>Algoritmo de ruta</h4>
          <p>A* (distancia + riesgo)</p>
          <strong>Tiempo: {calcTime}</strong>
        </div>
      </div>
    </section>
  )
}

export default BottomMetrics
