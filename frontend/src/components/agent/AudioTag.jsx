import React, { useState, useRef, useEffect } from 'react'
import { Space, Tooltip, Input } from 'antd'
import { PlayCircleOutlined, PauseCircleOutlined, CloseOutlined, SoundOutlined } from '@ant-design/icons'
import styles from './AudioTag.module.css'

const { TextArea } = Input

/**
 * 音频可视化标签组件
 * @param {object} audioData - 音频数据 {id, url, transcript, duration}
 * @param {function} onRemove - 移除回调
 * @param {boolean} removable - 是否可移除
 * @param {string} variant - 样式变体 'input' | 'message'
 * @param {function} onTranscriptChange - 转录文本修改回调
 * @param {boolean} editable - 转录文本是否可编辑
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 */
const AudioTag = ({ audioData, onRemove, removable = true, variant = 'input', onTranscriptChange, editable = true, renderHighlightedText }) => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isEditingTranscript, setIsEditingTranscript] = useState(false)
  const [editedTranscript, setEditedTranscript] = useState(audioData?.transcript || '')
  const audioRef = useRef(null)
  const animationRef = useRef(null)
  const transcriptInputRef = useRef(null)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleEnded = () => setIsPlaying(false)
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime)
    
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    
    return () => {
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  const togglePlay = async (e) => {
    e.stopPropagation()
    const audio = audioRef.current
    if (!audio) return

    if (isPlaying) {
      audio.pause()
      setIsPlaying(false)
    } else {
      try {
        await audio.play()
        setIsPlaying(true)
      } catch (error) {
        console.error('[AudioTag] 播放失败:', error)
      }
    }
  }

  const handleRemove = (e) => {
    e.stopPropagation()
    const audio = audioRef.current
    if (audio) {
      audio.pause()
      audio.currentTime = 0
    }
    onRemove?.()
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const progress = audioData?.duration > 0 ? (currentTime / audioData.duration) * 100 : 0
  
  // 根据variant选择className
  const tagClassName = variant === 'message' ? styles.audioTagMessage : styles.audioTagInput

  // 转录文本编辑相关
  const handleTranscriptClick = () => {
    if (editable && !isEditingTranscript) {
      setIsEditingTranscript(true)
      setTimeout(() => {
        // Ant Design TextArea 的 select 方法在 resizableTextArea.textArea 上
        const textArea = transcriptInputRef.current?.resizableTextArea?.textArea
        if (textArea) {
          textArea.focus()
          textArea.select()
        }
      }, 0)
    }
  }

  const handleTranscriptBlur = () => {
    setIsEditingTranscript(false)
    if (editedTranscript !== audioData?.transcript) {
      onTranscriptChange?.(audioData.id, editedTranscript)
    }
  }

  const handleTranscriptKeyDown = (e) => {
    // Ctrl+Enter 或 Cmd+Enter 保存
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleTranscriptBlur()
    } else if (e.key === 'Escape') {
      setEditedTranscript(audioData?.transcript || '')
      setIsEditingTranscript(false)
    }
  }

  return (
    <div className={tagClassName}>
      <audio ref={audioRef} src={audioData?.url} preload="metadata" />
      
      <div className={styles.audioContent}>
        {/* 播放按钮 */}
        <button 
          className={styles.playButton}
          onClick={togglePlay}
          type="button"
          title={isPlaying ? '暂停' : '播放'}
        >
          {isPlaying ? (
            <PauseCircleOutlined style={{ fontSize: 14, color: '#595959' }} />
          ) : (
            <PlayCircleOutlined style={{ fontSize: 14, color: '#595959' }} />
          )}
        </button>
        
        {/* 音频信息 */}
        <div className={styles.audioInfo}>
          <div className={styles.audioWaveform}>
            <SoundOutlined className={isPlaying ? styles.soundIconActive : styles.soundIcon} />
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${progress}%` }} />
            </div>
            <span className={styles.duration}>
              {formatTime(currentTime)} / {formatTime(audioData?.duration || 0)}
            </span>
          </div>
          
          {/* 转录文本 - 可编辑或高亮显示 */}
          {audioData?.transcript && (
            <div className={styles.transcriptContainer}>
              {isEditingTranscript ? (
                <TextArea
                  ref={transcriptInputRef}
                  className={styles.transcriptInput}
                  value={editedTranscript}
                  onChange={(e) => setEditedTranscript(e.target.value)}
                  onBlur={handleTranscriptBlur}
                  onKeyDown={handleTranscriptKeyDown}
                  autoSize={{ minRows: 1, maxRows: 6 }}
                  placeholder="输入转录文本"
                />
              ) : (
                <span 
                  className={styles.transcript} 
                  onClick={handleTranscriptClick}
                  title={editable ? '点击编辑转录文本 (Ctrl+Enter保存)' : audioData.transcript}
                  style={{ cursor: editable ? 'text' : 'default' }}
                >
                  {renderHighlightedText ? renderHighlightedText(audioData.transcript) : audioData.transcript}
                </span>
              )}
            </div>
          )}
        </div>

        {/* 删除按钮 */}
        {removable && (
          <button 
            className={styles.removeButton}
            onClick={handleRemove}
            type="button"
            title="删除"
          >
            <CloseOutlined style={{ fontSize: 9 }} />
          </button>
        )}
      </div>
    </div>
  )
}

export default AudioTag

