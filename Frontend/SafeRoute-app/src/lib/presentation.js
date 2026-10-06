// Centraliza etiquetas y colores para mantener una escala de riesgo consistente en la interfaz.
export const RISK_LEVELS = {
  bajo: { label: 'Bajo', color: '#22c55e', textColor: '#166534', background: '#dcfce7' },
  medio: { label: 'Medio', color: '#f59e0b', textColor: '#92400e', background: '#fef3c7' },
  alto: { label: 'Alto', color: '#ef4444', textColor: '#991b1b', background: '#fee2e2' },
}

// Relaciona las claves de la API con los nombres visibles de los tres modelos.
export const MODEL_OPTIONS = [
  { key: 'random_forest', name: 'Random Forest' },
  { key: 'xgboost', name: 'XGBoost' },
  { key: 'lstm', name: 'LSTM' },
]

export const ROUTE_COLOR = '#0879f9'
export const ALTERNATIVE_COLOR = '#334155'

// Presenta el mes de predicción en español sin desplazarlo por la zona horaria del equipo.
export function formatPeriod(period) {
  if (!/^\d{4}-\d{2}$/.test(period ?? '')) return 'No disponible'
  const [year, month] = period.split('-').map(Number)
  return new Intl.DateTimeFormat('es-PE', { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, month - 1, 1)))
}

// Escapa los valores insertados en etiquetas HTML de Leaflet.
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character])
}
