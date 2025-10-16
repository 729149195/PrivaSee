import React, { useState, useRef, useEffect } from 'react'
import { Button, Upload, message, Popover, Space } from 'antd'
import { AudioOutlined, LoadingOutlined, CloseOutlined } from '@ant-design/icons'
import styles from './AudioRecorder.module.css'

/**
 * 语音录制和上传组件
 * @param {function} onAudioAdded - 添加音频的回调 (audioData: {id, blob, transcript, duration})
 * @param {boolean} disabled - 是否禁用
 */
const AudioRecorder = ({ onAudioAdded, disabled = false }) => {
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [popoverVisible, setPopoverVisible] = useState(false)
  
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const streamRef = useRef(null)
  const isCancelledRef = useRef(false) // 用于标记是否是用户取消的

  // 清理录音资源
  const cleanupRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    mediaRecorderRef.current = null
    chunksRef.current = []
    setRecordingTime(0)
  }

  // 组件卸载时清理
  useEffect(() => {
    return () => cleanupRecording()
  }, [])

  // 开始录音
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []
      isCancelledRef.current = false // 重置取消标志
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }
      
      mediaRecorder.onstop = async () => {
        // 只有在非取消状态下才处理音频
        if (!isCancelledRef.current) {
          const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
          await processAudio(audioBlob)
        }
        cleanupRecording()
      }
      
      mediaRecorder.start()
      setIsRecording(true)
      setPopoverVisible(false)
      
      // 录音计时
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
      
      message.success('开始录音')
    } catch (error) {
      message.error(`无法访问麦克风: ${error.message}`)
      console.error('[AudioRecorder] 麦克风访问失败:', error)
    }
  }

  // 停止录音
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  // 取消录音
  const cancelRecording = () => {
    isCancelledRef.current = true // 标记为取消状态
    cleanupRecording()
    setIsRecording(false)
    message.info('已取消录音')
  }

  // 获取音频时长
  const getAudioDuration = (blob) => {
    return new Promise((resolve) => {
      const audio = new Audio()
      const url = URL.createObjectURL(blob)
      
      audio.addEventListener('loadedmetadata', () => {
        URL.revokeObjectURL(url)
        resolve(audio.duration || 0)
      })
      
      audio.addEventListener('error', () => {
        URL.revokeObjectURL(url)
        resolve(recordingTime || 0) // 出错时使用录音时长
      })
      
      audio.src = url
    })
  }

  // 将Blob转换为Base64（用于持久化存储）
  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result)
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  // 处理音频：发送到后端转文本
  const processAudio = async (audioBlob) => {
    setIsProcessing(true)
    const hideLoading = message.loading('正在转换语音为文本...', 0)
    
    try {
      // 转换为WAV格式（Whisper更兼容）
      const wavBlob = await convertToWav(audioBlob)
      
      // 获取真实的音频时长
      const duration = await getAudioDuration(wavBlob)
      
      // 将音频转换为base64（用于持久化）
      const base64Data = await blobToBase64(wavBlob)
      
      // 发送到后端
      const formData = new FormData()
      formData.append('audio', wavBlob, 'recording.wav')
      formData.append('language', 'auto') // 自动检测中英文
      
      const response = await fetch('/whisper/transcribe', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`转换失败: ${response.statusText}`)
      }
      
      const result = await response.json()
      
      if (!result.text || result.text.trim() === '') {
        throw new Error('未识别到语音内容')
      }
      
      // 生成音频数据对象
      const audioData = {
        id: `audio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        url: base64Data, // 使用base64而不是blob URL，刷新后仍可播放
        transcript: result.text.trim(),
        duration: duration, // 使用真实的音频时长
        language: result.language || 'unknown',
        timestamp: Date.now()
      }
      
      hideLoading()
      message.success('语音转换成功')
      onAudioAdded?.(audioData)
      
    } catch (error) {
      hideLoading()
      message.error(`语音处理失败: ${error.message}`)
      console.error('[AudioRecorder] 处理失败:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  // 转换音频格式为WAV
  const convertToWav = async (blob) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const arrayBuffer = await blob.arrayBuffer()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
      
      // 转换为WAV
      const wavBuffer = audioBufferToWav(audioBuffer)
      return new Blob([wavBuffer], { type: 'audio/wav' })
    } catch (error) {
      console.warn('[AudioRecorder] WAV转换失败，使用原始格式:', error)
      return blob
    }
  }

  // AudioBuffer转WAV格式
  const audioBufferToWav = (audioBuffer) => {
    const numOfChan = audioBuffer.numberOfChannels
    const length = audioBuffer.length * numOfChan * 2 + 44
    const buffer = new ArrayBuffer(length)
    const view = new DataView(buffer)
    const channels = []
    let offset = 0
    let pos = 0

    // WAV头
    const setUint16 = (data) => {
      view.setUint16(pos, data, true)
      pos += 2
    }
    const setUint32 = (data) => {
      view.setUint32(pos, data, true)
      pos += 4
    }

    // RIFF identifier
    setUint32(0x46464952)
    // file length
    setUint32(length - 8)
    // RIFF type
    setUint32(0x45564157)
    // format chunk identifier
    setUint32(0x20746d66)
    // format chunk length
    setUint32(16)
    // sample format (raw)
    setUint16(1)
    // channel count
    setUint16(numOfChan)
    // sample rate
    setUint32(audioBuffer.sampleRate)
    // byte rate (sample rate * block align)
    setUint32(audioBuffer.sampleRate * 2 * numOfChan)
    // block align (channel count * bytes per sample)
    setUint16(numOfChan * 2)
    // bits per sample
    setUint16(16)
    // data chunk identifier
    setUint32(0x61746164)
    // data chunk length
    setUint32(length - pos - 4)

    // 写入音频数据
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
      channels.push(audioBuffer.getChannelData(i))
    }

    while (pos < length) {
      for (let i = 0; i < numOfChan; i++) {
        let sample = Math.max(-1, Math.min(1, channels[i][offset]))
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
        view.setInt16(pos, sample, true)
        pos += 2
      }
      offset++
    }

    return buffer
  }

  // 处理文件上传
  const handleFileUpload = async (file) => {
    // 检查文件大小（最大20MB）
    if (file.size > 20 * 1024 * 1024) {
      message.error(`音频文件过大 (${(file.size / 1024 / 1024).toFixed(2)}MB)，最大支持20MB`)
      return Upload.LIST_IGNORE
    }
    
    // 检查文件类型
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/webm', 'audio/m4a', 'audio/flac']
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|webm|m4a|flac)$/i)) {
      message.error('不支持的音频格式，请上传 WAV、MP3、OGG、WEBM、M4A 或 FLAC 格式')
      return Upload.LIST_IGNORE
    }
    
    setPopoverVisible(false)
    await processAudio(file)
    return Upload.LIST_IGNORE
  }

  // 格式化录音时间
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Popover内容
  const popoverContent = (
    <div className={styles.popoverContent}>
      <Space>
        <Button 
          type="primary" 
          icon={<AudioOutlined />} 
          onClick={startRecording}
          style={{ width: '90px' }}
        >
          录音
        </Button>
        <Upload
          accept="audio/*"
          showUploadList={false}
          beforeUpload={handleFileUpload}
        >
          <Button style={{ width: '90px' }}>上传</Button>
        </Upload>
      </Space>
    </div>
  )

  // 录音中显示
  if (isRecording) {
    return (
      <div className={styles.recordingContainer}>
        <div className={styles.recordingIndicator}>
          <div className={styles.recordingDot} />
          <span className={styles.recordingTime}>{formatTime(recordingTime)}</span>
        </div>
        <Space>
          <Button 
            type="primary" 
            onClick={stopRecording}
            size="small"
          >
            完成
          </Button>
          <Button 
            danger 
            icon={<CloseOutlined />} 
            onClick={cancelRecording}
            size="small"
          >
            取消
          </Button>
        </Space>
      </div>
    )
  }

  // 处理中显示
  if (isProcessing) {
    return (
      <Button 
        icon={<LoadingOutlined />} 
        disabled
      >
        处理中...
      </Button>
    )
  }

  // 默认按钮
  return (
    <Popover
      content={popoverContent}
      title="添加语音"
      trigger="click"
      open={popoverVisible}
      onOpenChange={setPopoverVisible}
      placement="topRight"
    >
      <Button 
        icon={<AudioOutlined />} 
        disabled={disabled}
        title="添加语音"
      />
    </Popover>
  )
}

export default AudioRecorder

