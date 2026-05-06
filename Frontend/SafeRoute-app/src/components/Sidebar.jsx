function Sidebar({ isOpen, onToggle }) {
  return (
    <aside className={`sidebar ${isOpen ? 'is-open' : 'is-collapsed'}`}>
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-icon">SR</div>
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
        <button type="button" className="nav-item nav-item--active">Inicio</button>
        <button type="button" className="nav-item">Nueva ruta</button>
        <button type="button" className="nav-item">Historial</button>
        <button type="button" className="nav-item">Zonas de riesgo</button>
        <button type="button" className="nav-item">Estadísticas</button>
        <button type="button" className="nav-item">Configuración</button>
        <button type="button" className="nav-item">Acerca de</button>
      </nav>
      <div className="sidebar-card">
        <h3>Tu seguridad</h3>
        <p>Es nuestra prioridad en Lima Metropolitana.</p>
      </div>
    </aside>
  )
}

export default Sidebar
