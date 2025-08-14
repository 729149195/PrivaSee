import React, { useEffect, useRef, useState } from 'react'
import styles from './AudioTest.module.css'

export default function AudioTest() {
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [transcribeLoading, setTranscribeLoading] = useState(false)
  const audioRef = useRef(null)

  const [sampleRate, setSampleRate] = useState(null)
  const [durationSec, setDurationSec] = useState(null)
  const [numChannels, setNumChannels] = useState(null)

  const [wave, setWave] = useState([])
  const [onsets, setOnsets] = useState([])
  const [beats, setBeats] = useState([])
  const [segments, setSegments] = useState([])
  const [uncertainty, setUncertainty] = useState([])
  const [currentTime, setCurrentTime] = useState(0)
  const rafRef = useRef(0)

  const waveformRef = useRef(null)
  const uncertRef = useRef(null)
  const [isSeeking, setIsSeeking] = useState(false)

  // Mic recording
  const [isRecording, setIsRecording] = useState(false)
  const [recError, setRecError] = useState('')
  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])

  // Transcript + PII + token-uncertainty for recognized text
  const [transcript, setTranscript] = useState('')
  const [transSegments, setTransSegments] = useState([])
  const [piiHtml, setPiiHtml] = useState('')
  const [tokUncertHtml, setTokUncertHtml] = useState('')
  const [transcribeError, setTranscribeError] = useState('')

  const escaped = (value) =>
    (value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')

  const buildHighlightedHtml = (rawText, entities) => {
    if (!rawText) return ''
    if (!entities || entities.length === 0) return escaped(rawText)
    const accepted = []
    let lastEnd = -1
    for (const ent of entities) {
      if (ent.start >= lastEnd) {
        accepted.push(ent)
        lastEnd = ent.end
      }
    }
    let html = ''
    let cursor = 0
    for (const ent of accepted) {
      const before = rawText.slice(cursor, ent.start)
      const chunk = rawText.slice(ent.start, ent.end)
      html += escaped(before)
      const conf = (ent.score ?? 0).toFixed(2)
      const title = `${ent.entity_type} (${conf})`
      const typeClass = styles[`type-${ent.entity_type}`] || ''
      const className = `${styles.pii} ${typeClass}`.trim()
      const label = `${ent.entity_type} ${conf}`
      html += `<span class="${className}" data-entity="${escaped(label)}" title="${escaped(title)}">${escaped(chunk)}</span>`
      cursor = ent.end
    }
    html += escaped(rawText.slice(cursor))
    return html
  }

  const buildTokenUncertaintyHtml = (rawText, tokens) => {
    if (!rawText) return ''
    if (!tokens || tokens.length === 0) return escaped(rawText)
    const scores = new Array(rawText.length).fill(0)
    for (const t of tokens) {
      const s = Math.max(t.score_mean ?? 0, t.score_std ?? 0)
      for (let i = t.start; i < t.end && i < scores.length; i++) scores[i] = Math.max(scores[i], s)
    }
    let html = ''
    for (let i = 0; i < rawText.length;) {
      const base = scores[i]
      if (base <= 0) { html += escaped(rawText[i]); i += 1; continue }
      let j = i + 1
      while (j < rawText.length && scores[j] === base) j += 1
      const chunk = rawText.slice(i, j)
      const alpha = Math.min(0.85, Math.max(0.1, base))
      const bg = `rgba(255, 77, 79, ${alpha})`
      html += `<span class="${styles.heatSpan}" style="background:${bg}">${escaped(chunk)}</span>`
      i = j
    }
    return html
  }

  const clearState = () => {
    setSampleRate(null)
    setDurationSec(null)
    setNumChannels(null)
    setWave([])
    setOnsets([])
    setBeats([])
    setSegments([])
    setUncertainty([])
  }

  const onUpload = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    clearState()
    setFile(f)
    const url = URL.createObjectURL(f)
    setAudioUrl(url)
    setTranscript('')
    setTransSegments([])
    setPiiHtml('')
    setTokUncertHtml('')
    setTranscribeError('')
  }

  // Start/stop mic recording（最小改动：使用 MediaRecorder；Safari/Chrome 兼容尽量自适配）
  const canRecord = typeof window !== 'undefined' && typeof navigator !== 'undefined' && !!(navigator.mediaDevices) && typeof window.MediaRecorder !== 'undefined'

  const cleanupStream = () => {
    try {
      const s = streamRef.current
      if (s) {
        s.getTracks().forEach((t) => {
          try { t.stop() } catch { }
        })
      }
    } catch { }
    streamRef.current = null
  }

  const stopRecording = () => {
    try {
      const mr = mediaRecorderRef.current
      if (mr && mr.state !== 'inactive') mr.stop()
    } catch { }
    setIsRecording(false)
    // stream 停止在 onstop 之后统一清理
  }

  const startRecording = async () => {
    if (!canRecord || isRecording) return
    setRecError('')
    try {
      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // 选择可用 mimeType（尽量兼容 Safari/Chrome）
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/mp4',
        'audio/mpeg'
      ]
      let chosen = ''
      if (typeof window.MediaRecorder !== 'undefined' && typeof MediaRecorder.isTypeSupported === 'function') {
        for (const c of candidates) {
          try { if (MediaRecorder.isTypeSupported(c)) { chosen = c; break } } catch { }
        }
      }

      const opts = chosen ? { mimeType: chosen } : undefined
      const mr = new MediaRecorder(stream, opts)
      chunksRef.current = []
      mr.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data) }
      // 将录音在前端转换为 WAV，避免后端对 webm/mp4 解码失败
      const convertBlobToWavFile = async (blob) => {
        try {
          const arrayBuf = await blob.arrayBuffer()
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuf)
          const numChannels = audioBuffer.numberOfChannels
          const sampleRateLocal = audioBuffer.sampleRate
          const length = audioBuffer.length
          // 16-bit PCM WAV
          const bytesPerSample = 2
          const blockAlign = numChannels * bytesPerSample
          const wavBuffer = new ArrayBuffer(44 + length * blockAlign)
          const view = new DataView(wavBuffer)
          // RIFF header
          let offset = 0
          const writeString = (str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); offset += str.length }
          const writeUint32 = (v) => { view.setUint32(offset, v, true); offset += 4 }
          const writeUint16 = (v) => { view.setUint16(offset, v, true); offset += 2 }
          writeString('RIFF')
          writeUint32(36 + length * blockAlign)
          writeString('WAVE')
          writeString('fmt ')
          writeUint32(16) // PCM subchunk size
          writeUint16(1) // PCM format
          writeUint16(numChannels)
          writeUint32(sampleRateLocal)
          writeUint32(sampleRateLocal * blockAlign)
          writeUint16(blockAlign)
          writeUint16(8 * bytesPerSample)
          writeString('data')
          writeUint32(length * blockAlign)
          // interleave and write PCM
          const channelData = []
          for (let ch = 0; ch < numChannels; ch++) channelData.push(audioBuffer.getChannelData(ch))
          for (let i = 0; i < length; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
              let s = channelData[ch][i]
              s = Math.max(-1, Math.min(1, s))
              view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
              offset += 2
            }
          }
          const wavBlob = new Blob([view], { type: 'audio/wav' })
          return new File([wavBlob], 'recording.wav', { type: 'audio/wav' })
        } catch (err) {
          // 回退：直接返回原始 blob 文件
          return new File([blob], 'recording.webm', { type: blob.type || 'audio/webm' })
        }
      }

      mr.onstop = () => {
        (async () => {
          try {
            const mime = mr.mimeType || chosen || 'audio/webm'
            const blob = new Blob(chunksRef.current, { type: mime })
            const wavFile = await convertBlobToWavFile(blob)

            // 对齐上传流程的状态重置
            clearState()
            setFile(wavFile)
            const url = URL.createObjectURL(wavFile)
            setAudioUrl(url)
            setTranscript('')
            setTransSegments([])
            setPiiHtml('')
            setTokUncertHtml('')
            setTranscribeError('')
          } catch (err) {
            setRecError((err && err.message) ? err.message : 'Recording failed')
          } finally {
            cleanupStream()
            mediaRecorderRef.current = null
            chunksRef.current = []
          }
        })()
      }
      mediaRecorderRef.current = mr
      mr.start(100) // 分片间隔，降低内存占用
      setIsRecording(true)
    } catch (err) {
      setRecError((err && err.message) ? err.message : 'Microphone permission denied or unavailable')
      cleanupStream()
      mediaRecorderRef.current = null
      chunksRef.current = []
      setIsRecording(false)
    }
  }

  const analyze = async () => {
    if (!file) return
    setAnalyzeLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const resp = await fetch('http://localhost:8000/audio/analyze', {
        method: 'POST',
        body: form,
      })
      if (!resp.ok) throw new Error(`Audio analyze failed: ${resp.status}`)
      const data = await resp.json()
      setSampleRate(data.sample_rate || null)
      setDurationSec(data.duration_sec || null)
      setNumChannels(data.num_channels || null)
      setWave(Array.isArray(data.waveform_downsample) ? data.waveform_downsample : [])
      setOnsets(Array.isArray(data.onsets_sec) ? data.onsets_sec : [])
      setBeats(Array.isArray(data.beats_sec) ? data.beats_sec : [])
      setSegments(Array.isArray(data.segments) ? data.segments : [])
      setUncertainty(Array.isArray(data.frame_uncertainty) ? data.frame_uncertainty : [])
      setAnalyzeLoading(false)
      // kick off transcription + PII + token uncertainty (best-effort)
      try {
        setTranscribeLoading(true)
        const [trResp] = await Promise.all([
          fetch('http://localhost:8000/audio/transcribe', { method: 'POST', body: form }),
        ])
        if (trResp && trResp.ok) {
          const tr = await trResp.json()
          const fullText = tr?.text || ''
          const segs = Array.isArray(tr?.segments) ? tr.segments : []
          setTranscript(fullText)
          setTransSegments(segs)
          setTranscribeError('')
          // PII detection
          const piiResp = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullText, language: 'en', model: 'spacy/en_core_web_sm' }),
          })
          if (piiResp.ok) {
            const pii = await piiResp.json()
            setPiiHtml(buildHighlightedHtml(fullText, pii.entities || []))
          } else { setPiiHtml(escaped(fullText)) }
          // token uncertainty via LM
          const tuResp = await fetch('http://localhost:8000/uncertainty', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullText, model: 'distilgpt2', samples: 8, ig_steps: 8, max_input_tokens: 256 })
          })
          if (tuResp.ok) {
            const tu = await tuResp.json()
            setTokUncertHtml(buildTokenUncertaintyHtml(fullText, tu.tokens || []))
          } else { setTokUncertHtml(escaped(fullText)) }
        } else {
          setTranscript('')
          setTransSegments([])
          setPiiHtml('')
          setTokUncertHtml('')
          try {
            const err = await trResp.json()
            setTranscribeError(err?.detail || `Transcription unavailable (${trResp.status})`)
          } catch {
            setTranscribeError(`Transcription unavailable (${trResp?.status || 'error'})`)
          }
        }
      } catch (err) {
        setTranscript('')
        setTransSegments([])
        setPiiHtml('')
        setTokUncertHtml('')
        setTranscribeError('Transcription unavailable')
      } finally {
        setTranscribeLoading(false)
      }
    } catch (e) {
      console.error(e)
      setAnalyzeLoading(false)
    } finally {
      // no-op; analyzeLoading was cleared earlier or on error
    }
  }

  const drawWaveform = (canvas, waveArr, onsetsSec, beatsSec, segs, durSec, curT) => {
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const cssWidth = canvas.clientWidth || 800
    const cssHeight = 200
    canvas.width = Math.max(1, Math.floor(cssWidth * dpr))
    canvas.height = Math.max(1, Math.floor(cssHeight * dpr))
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, cssWidth, cssHeight)

    // background
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, cssWidth, cssHeight)

    // time ticks at bottom
    if (durSec && durSec > 0) {
      const y = cssHeight - 14
      ctx.strokeStyle = '#d9d9d9'
      ctx.fillStyle = '#8c8c8c'
      ctx.lineWidth = 1
      const approxTicks = Math.min(10, Math.max(4, Math.floor(cssWidth / 80)))
      const step = durSec / approxTicks
      const niceStep = step < 1 ? 0.5 : step < 2 ? 1 : step < 5 ? 2 : step < 10 ? 5 : 10
      for (let t = 0; t <= durSec + 1e-6; t += niceStep) {
        const x = Math.max(0, Math.min(cssWidth, (t / durSec) * cssWidth))
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 8); ctx.stroke()
        const label = `${t.toFixed(0)}s`
        ctx.fillText(label, x + 2, y + 8)
      }
    }

    // segments shading
    if (Array.isArray(segs) && segs.length && durSec) {
      ctx.fillStyle = 'rgba(22, 119, 255, 0.10)'
      for (const s of segs) {
        const x1 = Math.max(0, Math.min(cssWidth, (s.start_sec / durSec) * cssWidth))
        const x2 = Math.max(0, Math.min(cssWidth, (s.end_sec / durSec) * cssWidth))
        ctx.fillRect(x1, 0, Math.max(1, x2 - x1), cssHeight)
      }
    }

    // axes/grid
    ctx.strokeStyle = '#f0f0f0'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * cssHeight
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(cssWidth, y)
      ctx.stroke()
    }

    // waveform
    if (Array.isArray(waveArr) && waveArr.length) {
      const N = waveArr.length
      const centerY = cssHeight / 2
      const amp = cssHeight * 0.42
      ctx.strokeStyle = '#1677ff'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i < N; i++) {
        const x = (i / (N - 1)) * cssWidth
        const y = centerY - (waveArr[i] || 0) * amp
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    // onsets
    if (Array.isArray(onsetsSec) && onsetsSec.length && durSec) {
      ctx.strokeStyle = '#ff4d4f'
      ctx.lineWidth = 1
      for (const t of onsetsSec) {
        const x = Math.max(0, Math.min(cssWidth, (t / durSec) * cssWidth))
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, cssHeight)
        ctx.stroke()
      }
    }

    // beats
    if (Array.isArray(beatsSec) && beatsSec.length && durSec) {
      ctx.strokeStyle = '#52c41a'
      ctx.lineWidth = 1
      for (const t of beatsSec) {
        const x = Math.max(0, Math.min(cssWidth, (t / durSec) * cssWidth))
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, cssHeight)
        ctx.stroke()
      }
    }
    // playhead
    if (durSec && curT != null) {
      const x = Math.max(0, Math.min(cssWidth, (curT / durSec) * cssWidth))
      ctx.strokeStyle = '#222'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, cssHeight)
      ctx.stroke()
    }
  }

  const drawUncertainty = (canvas, uncArr, curT) => {
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const cssWidth = canvas.clientWidth || 800
    const cssHeight = 200
    canvas.width = Math.max(1, Math.floor(cssWidth * dpr))
    canvas.height = Math.max(1, Math.floor(cssHeight * dpr))
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, cssWidth, cssHeight)

    // background
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, cssWidth, cssHeight)

    if (!Array.isArray(uncArr) || !uncArr.length) return
    const N = uncArr.length
    const barW = Math.max(1, cssWidth / N)
    for (let i = 0; i < N; i++) {
      const v = Math.max(0.0, Math.min(1.0, uncArr[i] || 0))
      const alpha = Math.max(0.1, Math.min(0.85, v))
      ctx.fillStyle = `rgba(255,77,79,${alpha})`
      const x = i * barW
      ctx.fillRect(x, 0, barW + 0.5, cssHeight)
    }

    // time ticks
    if (durationSec && durationSec > 0) {
      const y = cssHeight - 14
      ctx.strokeStyle = '#d9d9d9'
      ctx.fillStyle = '#8c8c8c'
      ctx.lineWidth = 1
      const approxTicks = Math.min(10, Math.max(4, Math.floor(cssWidth / 80)))
      const step = durationSec / approxTicks
      const niceStep = step < 1 ? 0.5 : step < 2 ? 1 : step < 5 ? 2 : step < 10 ? 5 : 10
      for (let t = 0; t <= durationSec + 1e-6; t += niceStep) {
        const x = Math.max(0, Math.min(cssWidth, (t / durationSec) * cssWidth))
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 8); ctx.stroke()
        const label = `${t.toFixed(0)}s`
        ctx.fillText(label, x + 2, y + 8)
      }
    }

    // playhead
    if (durationSec && curT != null) {
      const x = Math.max(0, Math.min(cssWidth, (curT / durationSec) * cssWidth))
      ctx.strokeStyle = '#222'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, cssHeight)
      ctx.stroke()
    }
  }

  useEffect(() => {
    const c = waveformRef.current
    if (!c) return
    drawWaveform(c, wave, onsets, beats, segments, durationSec || 0, currentTime)
  }, [wave, onsets, beats, segments, durationSec, currentTime])

  useEffect(() => {
    const c = uncertRef.current
    if (!c) return
    drawUncertainty(c, uncertainty, currentTime)
  }, [uncertainty, currentTime])

  // Sync playhead: rebind whenever audio element/url changes
  useEffect(() => {
    const el = audioRef.current
    if (!el) return
    const onTime = () => setCurrentTime(el.currentTime || 0)
    const onSeek = () => setCurrentTime(el.currentTime || 0)
    let running = true
    const tick = () => {
      if (!running) return
      setCurrentTime((prev) => {
        const ct = el.currentTime || 0
        return Math.abs(ct - prev) > 0.016 ? ct : prev
      })
      rafRef.current = requestAnimationFrame(tick)
    }
    const onPlay = () => { running = true }
    const onPause = () => { running = false }
    el.addEventListener('timeupdate', onTime)
    el.addEventListener('seeking', onSeek)
    el.addEventListener('seeked', onSeek)
    el.addEventListener('play', onPlay)
    el.addEventListener('playing', onPlay)
    el.addEventListener('pause', onPause)
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      el.removeEventListener('timeupdate', onTime)
      el.removeEventListener('seeking', onSeek)
      el.removeEventListener('seeked', onSeek)
      el.removeEventListener('play', onPlay)
      el.removeEventListener('playing', onPlay)
      el.removeEventListener('pause', onPause)
      cancelAnimationFrame(rafRef.current)
    }
  }, [audioUrl])

  // 组件卸载时清理录音与流
  useEffect(() => {
    return () => {
      try { if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop() } catch { }
      cleanupStream()
    }
  }, [])

  // Seek by clicking/dragging on canvases
  const posToTime = (canvas, clientX) => {
    if (!canvas || !durationSec) return 0
    const rect = canvas.getBoundingClientRect()
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left))
    const t = (x / rect.width) * durationSec
    return Math.max(0, Math.min(durationSec, t))
  }

  const onPointerDown = (e, which) => {
    setIsSeeking(true)
    const canvas = which === 'wave' ? waveformRef.current : uncertRef.current
    const t = posToTime(canvas, e.clientX)
    if (audioRef.current) audioRef.current.currentTime = t
    setCurrentTime(t)
  }
  const onPointerMove = (e, which) => {
    if (!isSeeking) return
    const canvas = which === 'wave' ? waveformRef.current : uncertRef.current
    const t = posToTime(canvas, e.clientX)
    if (audioRef.current) audioRef.current.currentTime = t
    setCurrentTime(t)
  }
  const onPointerUp = () => setIsSeeking(false)

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Audio Test</h1>
      <div className={styles.panel}>
        <div className={styles.panelSection}>
          <div className={styles.panelTitle}>Audio Analysis</div>
          <div className={styles.panelGrid}>
            <label className={styles.label}>Upload</label>
            <input className={styles.input} type="file" accept="audio/*" onChange={onUpload} />
            <label className={styles.label}>Microphone</label>
            <div>
              <button className={styles.button} onClick={isRecording ? stopRecording : startRecording} disabled={!canRecord}>
                {isRecording ? 'Stop' : 'Record'}
              </button>
              {!canRecord ? (
                <div className={styles.legend} style={{ color: '#8c8c8c', marginTop: 6 }}>Recording not supported in this browser</div>
              ) : null}
              {!!recError && canRecord ? (
                <div className={styles.legend} style={{ color: '#cf1322', marginTop: 6 }}>{recError}</div>
              ) : null}
            </div>
            <label className={styles.label}>Sample rate</label>
            <div>{sampleRate != null ? `${sampleRate} Hz` : '-'}</div>
            <label className={styles.label}>Duration</label>
            <div>{durationSec != null ? `${durationSec.toFixed(2)} s` : '-'}</div>
            <label className={styles.label}>Channels</label>
            <div>{numChannels != null ? `${numChannels}` : '-'}</div>
            <div style={{ gridColumn: '1 / -1' }}>
              <button className={styles.button} onClick={analyze} disabled={!file || analyzeLoading}>Analyze</button>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.row}>
        <div className={styles.col}>
          <div className={styles.colHeader}>
            <div className={styles.colTitle}>Waveform & recognized events</div>
            <div className={styles.legend} style={{ marginTop: 6 }}>
              <span>Onset</span>
              <span style={{ display: 'inline-block', width: 24, height: 4, background: '#ff4d4f' }}></span>
              <span>Beat</span>
              <span style={{ display: 'inline-block', width: 24, height: 4, background: '#52c41a' }}></span>
              <span>Active segment</span>
              <span style={{ display: 'inline-block', width: 24, height: 8, background: 'rgba(22,119,255,0.10)' }}></span>
            </div>
          </div>
          <div className={styles.box}
            onPointerDown={(e) => onPointerDown(e, 'wave')}
            onPointerMove={(e) => onPointerMove(e, 'wave')}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}>
            {audioUrl ? (
              <canvas ref={waveformRef} className={styles.canvas} />
            ) : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            <div className={`${styles.overlayLoading} ${analyzeLoading ? styles.show : ''}`}><div className={styles.spinner}></div></div>
          </div>

        </div>
        <div className={styles.col}>
          <div className={styles.colHeader}>
            <div className={styles.colTitle}>Uncertainty heatmap</div>
            <div className={styles.legend} style={{ marginTop: 6 }}>
              <span>Low</span>
              <div className={styles.legendBar}></div>
              <span>High</span>
            </div>
          </div>
          <div className={styles.box}
            onPointerDown={(e) => onPointerDown(e, 'unc')}
            onPointerMove={(e) => onPointerMove(e, 'unc')}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}>
            {audioUrl ? (
              <canvas ref={uncertRef} className={styles.canvas} />
            ) : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            <div className={`${styles.overlayLoading} ${analyzeLoading ? styles.show : ''}`}><div className={styles.spinner}></div></div>
          </div>

        </div>
        <div className={styles.col}>
          <div className={styles.colTitle_audio}>Original audio</div>
          <div className={styles.box}>
            {audioUrl ? (
              <audio ref={audioRef} src={audioUrl} controls style={{ width: '100%' }} />
            ) : (
              <div className={styles.empty}><div className={styles.emptyIcon}></div></div>
            )}
            {/* Preview does not show loading overlay */}
          </div>
        </div>
      </div>
      {/* Transcript + PII + token uncertainty */}
      <div className={styles.subRow}>
        <div className={styles.col}>
          <div className={styles.subTitle}>Transcript PII highlight</div>
          <div className={styles.textBox}>
            {transcribeLoading && !transcribeError ? (
              <div>
                <div className={styles.skeletonLine} style={{ width: '85%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '70%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '92%' }}></div>
              </div>
            ) : null}
            {!!transcribeError && (
              <div className={styles.legend} style={{ color: '#cf1322', marginBottom: 6 }}>
                {transcribeError} · 请安装 faster-whisper 以启用转写
              </div>
            )}
            {(transcribeLoading || (!!transcript && !piiHtml)) && !transcribeError ? (
              <div>
                <div className={styles.skeletonLine} style={{ width: '85%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '70%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '92%' }}></div>
              </div>
            ) : null}
            {!!piiHtml && !transcribeLoading && !transcribeError ? (
              <div className={styles.rich} dangerouslySetInnerHTML={{ __html: piiHtml }} />
            ) : null}
            <div className={styles.tagBar}>
              {durationSec != null ? (
                <span className={styles.tag}>t = {currentTime.toFixed(2)}s / {durationSec.toFixed(2)}s</span>
              ) : null}
              {Array.isArray(transSegments) && transSegments.length ? (
                <span className={styles.tag}>segments: {transSegments.length}</span>
              ) : null}
            </div>
          </div>
        </div>
        <div className={styles.col}>
          <div className={styles.subTitle}>Transcript token uncertainty</div>
          <div className={styles.textBox}>
            {transcribeLoading && !transcribeError ? (
              <div>
                <div className={styles.skeletonLine} style={{ width: '80%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '60%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '78%' }}></div>
              </div>
            ) : null}
            {(transcribeLoading || (!!transcript && !tokUncertHtml)) && !transcribeError ? (
              <div>
                <div className={styles.skeletonLine} style={{ width: '80%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '60%' }}></div>
                <div className={styles.skeletonLine} style={{ width: '78%' }}></div>
              </div>
            ) : null}
            {!!tokUncertHtml && !transcribeLoading && !transcribeError ? (
              <div className={styles.rich} dangerouslySetInnerHTML={{ __html: tokUncertHtml }} />
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}


