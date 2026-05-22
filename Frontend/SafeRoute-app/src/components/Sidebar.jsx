import {
  AlertTriangle,
  BarChart2,
  Clock,
  Home,
  Info,
  MapPin,
  Settings,
  Shield,
} from 'lucide-react'

const NAV_ITEMS = [
  { label: 'Inicio', icon: Home, active: true },
  { label: 'Nueva ruta', icon: MapPin },
  { label: 'Historial', icon: Clock },
  { label: 'Zonas de riesgo', icon: AlertTriangle },
  { label: 'Estadísticas', icon: BarChart2 },
  { label: 'Configuración', icon: Settings },
  { label: 'Acerca de', icon: Info },
]

function Sidebar({ isOpen, onToggle }) {
  return (
    <aside className={`sidebar ${isOpen ? 'is-open' : 'is-collapsed'}`}>
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={16} />
            <span>K</span>
          </div>
          <div>
            <p className="brand-name">SafeRoute</p>
            <span>Rutas óptimas y seguras</span>
          </div>
        </div>
        <button
          type="button"
          className="hamburger"
          onClick={onToggle}
          aria-label={isOpen ? 'Contraer menú' : 'Expandir menú'}
        >
          <span />
          <span />
          <span />
        </button>
      </div>
      <nav className="nav-menu">
        {NAV_ITEMS.map(({ label, icon: Icon, active }) => (
          <button
            key={label}
            type="button"
            className={active ? 'nav-item nav-item--active' : 'nav-item'}
          >
            <Icon size={16} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-card">
        <div className="sidebar-card-icon">
          <Shield size={28} />
        </div>
        <div>
          <h3>Tu seguridad es nuestra prioridad</h3>
          <p>Rutas inteligentes con datos reales de Lima Metropolitana.</p>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
