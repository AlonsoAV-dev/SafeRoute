// Escala continua fija, independiente del área visible y de la cantidad de calles.
const DEFAULT_STOPS = [
  [0, [187, 247, 208]], [0.20, [74, 222, 128]], [0.34, [163, 230, 53]],
  [0.50, [250, 204, 21]], [0.66, [249, 115, 22]], [0.80, [239, 68, 68]], [1, [185, 28, 28]],
]

// Interpola el color entre los puntos de la escala recibida del backend.
function colorAt(score, stops) {
  for (let index = 1; index < stops.length; index += 1) {
    const [end, to] = stops[index]
    if (score > end) continue
    const [start, from] = stops[index - 1]
    const t = Math.max(0, Math.min(1, (score - start) / (end - start)))
    return from.map((value, channel) => value + (to[channel] - value) * t)
  }
  return stops.at(-1)[1]
}

// Rasteriza los focos con el kernel compartido y conserva áreas transparentes cuando la influencia es tenue.
export function rasterizeRiskHotspots(cells, width, height, project,
  { step = 3, radius = 24, fieldConfig = {} } = {}) {
  const focusFloor = fieldConfig.focus_floor ?? 0.6
  const exponent = fieldConfig.opacity_exponent ?? 2
  const cutoff = fieldConfig.min_influence ?? 0.025
  const kernelExponent = fieldConfig.kernel_exponent ?? 4.5
  const stops = fieldConfig.color_stops ?? DEFAULT_STOPS
  const columns = Math.ceil(width / step)
  const rows = Math.ceil(height / step)
  const size = columns * rows
  const intensity = new Float32Array(size)
  const localScores = new Float32Array(size)
  const pixelWidth = width / columns
  const pixelHeight = height / rows
  for (const [lat, lng, score] of cells) {
    if (!Number.isFinite(score)) continue
    // La opacidad tiene una transformación fija y conserva el score original para el campo compartido.
    const strength = Math.max(0, Math.min(1, (score - focusFloor) / (1 - focusFloor))) ** exponent
    if (strength <= cutoff) continue
    const location = project(lat, lng)
    const left = Math.max(0, Math.floor((location.x - radius) / pixelWidth))
    const right = Math.min(columns - 1, Math.ceil((location.x + radius) / pixelWidth))
    const top = Math.max(0, Math.floor((location.y - radius) / pixelHeight))
    const bottom = Math.min(rows - 1, Math.ceil((location.y + radius) / pixelHeight))
    for (let py = top; py <= bottom; py += 1) {
      for (let px = left; px <= right; px += 1) {
        const squaredDistance = ((px + 0.5) * pixelWidth - location.x) ** 2 + ((py + 0.5) * pixelHeight - location.y) ** 2
        if (squaredDistance > radius * radius) continue
        const contribution = Math.exp(-kernelExponent * squaredDistance / (radius * radius))
        const index = py * columns + px
        // La combinación por máximo evita saturar de rojo el mapa al superponer focos.
        const influence = strength * contribution
        const scoreHere = score * Math.sqrt(contribution)
        if (influence > intensity[index] || (influence === intensity[index] && scoreHere > localScores[index])) {
          intensity[index] = influence
          localScores[index] = scoreHere
        }
      }
    }
  }
  const pixels = new Uint8ClampedArray(size * 4)
  for (let index = 0; index < size; index += 1) {
    const value = intensity[index]
    if (value <= cutoff) continue
    const color = colorAt(localScores[index], stops)
    for (let channel = 0; channel < 3; channel += 1) pixels[index * 4 + channel] = color[channel]
    const fade = Math.min(1, (value - cutoff) / 0.065)
    pixels[index * 4 + 3] = Math.round(255 * 0.76 * Math.sqrt(value) * fade)
  }
  return { pixels, width: columns, height: rows }
}
