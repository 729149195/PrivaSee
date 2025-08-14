import React, { useEffect, useRef, useState } from 'react'
import styles from './ImgTest.module.css'

export default function ImgTest() {
  const [file, setFile] = useState(null)
  const [imgUrl, setImgUrl] = useState('')
  const [meta, setMeta] = useState(null)
  const [clipTags, setClipTags] = useState([])
  const [clipProbs, setClipProbs] = useState(null)
  const [clipLabels, setClipLabels] = useState(null)
  const [geo, setGeo] = useState([])
  const [geoUnc, setGeoUnc] = useState(null)
  const [geoProb, setGeoProb] = useState(null)
  const [boxes, setBoxes] = useState([])
  const [faces, setFaces] = useState([])
  const [plates, setPlates] = useState([])
  const [ocr, setOcr] = useState([])
  const [heat, setHeat] = useState([])
  const [loading, setLoading] = useState(false)

  const canvasRef = useRef(null)
  const heatRef = useRef(null)

  const drawBoxes = (canvas, image, list, color) => {
    const ctx = canvas.getContext('2d')
    canvas.width = image.width
    canvas.height = image.height
    ctx.drawImage(image, 0, 0)
    ctx.lineWidth = Math.max(2, Math.min(6, Math.round(Math.max(image.width, image.height) / 600)))
    const baseFont = Math.max(12, Math.round(Math.max(image.width, image.height) / 40))
    ctx.font = `${baseFont}px sans-serif`
    const piiColor = (types) => {
      if (!types || !types.length) return '#fa8c16'
      const priority = [
        'CREDIT_CARD','US_SSN','US_PASSPORT','US_BANK_NUMBER','IBAN_CODE','EMAIL_ADDRESS','PHONE_NUMBER','IP_ADDRESS','URL','LOCATION','PERSON','ORGANIZATION'
      ]
      const colorMap = {
        CREDIT_CARD: '#ff4d4f',
        US_SSN: '#f759ab',
        US_PASSPORT: '#d48806',
        US_BANK_NUMBER: '#ff7a45',
        IBAN_CODE: '#73d13d',
        EMAIL_ADDRESS: '#faad14',
        PHONE_NUMBER: '#52c41a',
        IP_ADDRESS: '#fa8c16',
        URL: '#1677ff',
        LOCATION: '#9254de',
        PERSON: '#13c2c2',
        ORGANIZATION: '#2f54eb',
      }
      const t = priority.find((p) => types.includes(p)) || types[0]
      return colorMap[t] || '#fa8c16'
    }
    for (const b of list) {
      // decide color per box
      let stroke = color
      if (b.label === 'FACE') stroke = '#52c41a'
      else if (b.label === 'PLATE') stroke = '#722ed1'
      else if (b.text) stroke = piiColor(b.pii_types)
      ctx.strokeStyle = stroke
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
      if (b.label) {
        ctx.fillStyle = stroke
        let text = `${b.label} ${(b.score ?? 0).toFixed(2)}`
        if (b.text) text += ` ${b.text}`
        if (b.pii_types && b.pii_types.length) text += ` [${b.pii_types.join(',')}]`
        if (b.uncert != null) text += ` u=${(b.uncert).toFixed(2)}`
        const padY = Math.round(baseFont * 0.3)
        const barH = baseFont + padY
        const w = ctx.measureText(text).width + 6
        ctx.fillRect(b.x1, Math.max(0, b.y1 - barH), w, barH)
        ctx.fillStyle = '#fff'
        ctx.fillText(text, b.x1 + 3, Math.max(baseFont, b.y1 - Math.round(padY * 0.5)))
      }
    }
  }

  const drawHeat = (canvas, image, pixelHeat, ocrList, faceList) => {
    const ctx = canvas.getContext('2d')
    canvas.width = image.width
    canvas.height = image.height
    ctx.drawImage(image, 0, 0)
    // pixel-level heat map if available
    if (pixelHeat && Array.isArray(pixelHeat) && pixelHeat.length) {
      const hH = pixelHeat.length
      const hW = pixelHeat[0].length
      const temp = document.createElement('canvas')
      temp.width = hW
      temp.height = hH
      const tctx = temp.getContext('2d')
      const imgData = tctx.createImageData(hW, hH)
      // color map: blue->yellow->red
      const colorMap = (v) => {
        // stronger gamma to boost visibility
        const x = Math.pow(Math.max(0, Math.min(1, v)), 0.35)
        const r = Math.round(255 * Math.max(0, 1.5 * x - 0.5))
        const g = Math.round(255 * Math.max(0, Math.min(1.5 * x, 1.5 * (1 - x))))
        const b = Math.round(255 * Math.max(0, 1.5 * (1 - x) - 0.5))
        return [r, g, b]
      }
      let p = 0
      for (let y = 0; y < hH; y++) {
        for (let x = 0; x < hW; x++) {
          const v = pixelHeat[y][x]
          const [r, g, b] = colorMap(v)
          imgData.data[p++] = r
          imgData.data[p++] = g
          imgData.data[p++] = b
          const a = Math.pow(Math.max(0, Math.min(1, v)), 0.35)
          imgData.data[p++] = Math.round(245 * a)
        }
      }
      tctx.putImageData(imgData, 0, 0)
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(temp, 0, 0, image.width, image.height)
    }
    // overlay text uncertainty regions (light blue, stronger alpha)
    for (const t of (ocrList || [])) {
      if (t.uncert == null) continue
      const a = Math.min(0.75, Math.max(0.15, t.uncert))
      ctx.fillStyle = `rgba(24,144,255,${a})`
      ctx.fillRect(t.x1, t.y1, t.x2 - t.x1, t.y2 - t.y1)
    }

    // overlay FACE regions with opacity based on (1 - score)
    for (const f of (faceList || [])) {
      if (!f || String(f.label).toUpperCase() !== 'FACE') continue
      const u = 1 - Math.max(0, Math.min(1, f.score ?? 0))
      const a = Math.min(0.75, Math.max(0.15, u))
      // green tint for faces to区分于OCR
      ctx.fillStyle = `rgba(82,196,26,${a})`
      ctx.fillRect(f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1)
      // add border to separate from heatmap
      ctx.lineWidth = Math.max(2, Math.min(6, Math.round(Math.max(image.width, image.height) / 600)))
      ctx.strokeStyle = 'rgba(82,196,26,1)'
      ctx.strokeRect(f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1)
    }
  }

  useEffect(() => {
    if (!imgUrl) return
    const img = new Image()
    img.onload = () => {
      if (canvasRef.current) drawBoxes(canvasRef.current, img, [...boxes, ...faces, ...plates, ...ocr], '#1677ff')
      if (heatRef.current) drawHeat(heatRef.current, img, meta?.pixel_heat || null, ocr, faces)
    }
    img.src = imgUrl
  }, [imgUrl, boxes, faces, plates, ocr, heat, meta])

  const onUpload = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    // Clear previous results
    setClipTags([])
    setClipProbs(null)
    setClipLabels(null)
    setMeta(null)
    setBoxes([])
    setFaces([])
    setPlates([])
    setOcr([])
    setHeat([])
    setGeo([])
    setGeoUnc(null)
    setGeoProb(null)
    setFile(f)
    const url = URL.createObjectURL(f)
    setImgUrl(url)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const resp = await fetch('http://localhost:8000/image/analyze', {
        method: 'POST',
        body: form,
      })
      if (!resp.ok) throw new Error(`Image analyze failed: ${resp.status}`)
      const data = await resp.json()
      setMeta({ width: data.width, height: data.height, exif: data.exif || {}, pixel_heat: data.pixel_heat || null })
      setClipTags(data.clip_top || [])
      setClipProbs(data.clip_probs || null)
      setClipLabels(data.clip_labels || null)
      setBoxes(data.detections || [])
      setFaces(data.faces || [])
      setPlates(data.plates || [])
      setOcr(data.ocr_boxes || [])
      setHeat(data.heat_boxes || [])
      setGeo(data.geo_candidates || [])
      setGeoUnc(data.geo_uncertainty || null)
      setGeoProb(data.geo_prob ?? null)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Image Test</h1>
      <div className={styles.panel}>
        <div className={styles.panelSection}>
          <div className={styles.panelTitle}>Image Analysis</div>
          <div className={styles.panelGrid}>
            <label className={styles.label}>Upload</label>
            <input className={styles.input} type="file" accept="image/*" onChange={onUpload} />
            <label className={styles.label}>CLIP tags</label>
            <div>{clipTags.join(', ') || '-'}</div>
            <label className={styles.label}>CLIP indoor/outdoor</label>
            <div>
              {clipProbs && clipLabels && clipLabels.indoor_outdoor && clipProbs.indoor_outdoor
                ? clipLabels.indoor_outdoor.map((t, i) => `${t}: ${(clipProbs.indoor_outdoor[i] ?? 0).toFixed(2)}`).join(', ')
                : '-'}
            </div>
            <label className={styles.label}>CLIP landmark set</label>
            <div>
              {clipProbs && clipLabels && clipLabels.landmark_set && clipProbs.landmark_set
                ? clipLabels.landmark_set.map((t, i) => `${t}: ${(clipProbs.landmark_set[i] ?? 0).toFixed(2)}`).join(', ')
                : '-'}
            </div>
            <label className={styles.label}>Size</label>
            <div>{meta ? `${meta.width} × ${meta.height}` : '-'}</div>
            <label className={styles.label}>EXIF GPS</label>
            <div>{meta && (meta.exif?.GPSLatitudeDecimal != null ? `${meta.exif.GPSLatitudeDecimal.toFixed(6)}, ${meta.exif.GPSLongitudeDecimal.toFixed(6)}` : '无')}</div>
            <label className={styles.label}>All EXIF</label>
            <div style={{ maxHeight: 120, overflow: 'auto', gridColumn: '2 / -1', fontSize: 12 }}>
              <pre style={{ margin: 0 }}>{meta ? JSON.stringify(meta.exif, null, 2) : '-'}</pre>
            </div>
            <label className={styles.label}>Geo probability</label>
            <div>{geoProb != null ? `${(geoProb*100).toFixed(1)}%` : '-'}</div>
            <div style={{ gridColumn: '1 / -1' }}>
              <button className={styles.button} onClick={analyze} disabled={!file || loading}>Analyze</button>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.row}>
        <div className={styles.col}>
          <div className={styles.colTitle}>Detections (PII/Objects/Faces/Plates/OCR)</div>
          <div className={styles.box}>
            {imgUrl ? (
              <canvas ref={canvasRef} className={styles.canvas} />
            ) : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            <div className={`${styles.overlayLoading} ${loading ? styles.show : ''}`}><div className={styles.spinner}></div></div>
          </div>
        </div>
        <div className={styles.col}>
          <div className={styles.colTitle}>Uncertainty heatmap</div>
          <div className={styles.box}>
            {imgUrl ? (
              <canvas ref={heatRef} className={styles.canvas} />
            ) : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            <div className={`${styles.overlayLoading} ${loading ? styles.show : ''}`}><div className={styles.spinner}></div></div>
          </div>
        </div>
        <div className={styles.col}>
          <div className={styles.colTitle}>Preview</div>
          <div className={styles.box}>
            {imgUrl ? <img src={imgUrl} className={styles.canvas} /> : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            {/* Preview 不进入加载态 */}
          </div>
        </div>
      </div>
    </div>
  )
}


