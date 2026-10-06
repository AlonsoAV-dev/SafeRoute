import { useEffect } from 'react'
import { useMap } from 'react-leaflet'
import L from 'leaflet'
import { rasterizeRiskHotspots } from '../lib/riskSurface'

// Dibuja el campo espacial recibido de la API y sincroniza el canvas con los movimientos de Leaflet.
export default function RiskSurfaceLayer({ surface, enabled }) {
  const map = useMap()
  useEffect(() => {
    if (!enabled || !surface?.cells?.length) return undefined
    let frame = null
    const Layer = L.Layer.extend({
      onAdd(target) {
        this.canvas = L.DomUtil.create('canvas', 'risk-surface-canvas leaflet-layer')
        this.canvas.style.pointerEvents = 'none'
        target.getPane('risk-heatmap').appendChild(this.canvas)
        this.schedule = () => {
          if (frame !== null) window.cancelAnimationFrame(frame)
          frame = window.requestAnimationFrame(() => this.draw(target))
        }
        this.hide = () => { this.canvas.style.visibility = 'hidden' }
        target.on('zoomstart', this.hide)
        target.on('moveend resize zoomend', this.schedule)
        this.schedule()
      },
      draw(target) {
        frame = null
        const size = target.getSize()
        if (!size.x || !size.y) return
        this.canvas.width = size.x
        this.canvas.height = size.y
        L.DomUtil.setPosition(this.canvas, target.containerPointToLayerPoint([0, 0]))
        const metersPerPixel = 156543.03392 * Math.cos(target.getCenter().lat * Math.PI / 180) / (2 ** target.getZoom())
        // Radio físico constante: el tamaño de la zona no cambia con el zoom.
        const radius = (surface.risk_field?.radius_m ?? 250) / metersPerPixel
        const raster = rasterizeRiskHotspots(surface.cells, size.x, size.y,
          (lat, lng) => target.latLngToContainerPoint([lat, lng]),
          { radius, fieldConfig: surface.risk_field })
        const buffer = document.createElement('canvas')
        buffer.width = raster.width
        buffer.height = raster.height
        buffer.getContext('2d').putImageData(new ImageData(raster.pixels, raster.width, raster.height), 0, 0)
        const context = this.canvas.getContext('2d')
        context.imageSmoothingEnabled = true
        context.drawImage(buffer, 0, 0, size.x, size.y)
        this.canvas.style.visibility = 'visible'
      },
      onRemove(target) {
        target.off('moveend resize zoomend', this.schedule)
        target.off('zoomstart', this.hide)
        if (frame !== null) window.cancelAnimationFrame(frame)
        this.canvas.remove()
      },
    })
    const layer = new Layer().addTo(map)
    return () => map.removeLayer(layer)
  }, [map, surface, enabled])
  return null
}
