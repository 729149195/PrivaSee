import React, { useState } from 'react'
import { Select, Button, Modal, Input, message } from 'antd'
import { SettingOutlined, FileTextOutlined, PictureOutlined, SoundOutlined, ThunderboltOutlined, KeyOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import { isModelMultimodal, getModelModalities, supportsChainOfThought } from '../../utils/modelUtils'
import ModelConfigPanel from './ModelConfigPanel'

/**
 * 模型选择工具栏组件
 * @param {string} model - 当前选中的模型
 * @param {Array} models - 可用模型列表
 * @param {object} customProviders - 自定义提供商配置
 * @param {function} setModel - 切换模型的回调函数
 * @param {function} addApiModel - 添加 API 模型的回调函数
 * @param {boolean} contextHasImages - 上下文是否包含图片
 * @param {number} selectedImagesCount - 已选择的图片数量
 */
const ModelPickerToolbar = ({ 
  model, 
  models, 
  customProviders,
  setModel, 
  addApiModel,
  contextHasImages,
  selectedImagesCount
}) => {
  const [apiModalOpen, setApiModalOpen] = useState(false)
  const [configPanelOpen, setConfigPanelOpen] = useState(false)
  const [apiModelId, setApiModelId] = useState('')
  const [apiBaseUrl, setApiBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')

  const handleAddApiModel = () => {
    try {
      addApiModel({ id: apiModelId.trim(), baseUrl: apiBaseUrl.trim(), apiKey: apiKey.trim() })
      setApiModalOpen(false)
      setApiModelId('')
      setApiBaseUrl('')
      setApiKey('')
    } catch (_) { }
  }

  const requireMultimodal = Boolean(contextHasImages || (selectedImagesCount > 0))

  return (
    <>
      <div className={styles.toolbar}>
        <div className={styles.modelPicker}>
          <Select
            style={{ minWidth: 220 }}
            value={model}
            onChange={(v) => {
              if (requireMultimodal && !isModelMultimodal(v, customProviders)) {
                message.warning('Cannot switch to a non-multimodal model when images exist in context or pending')
                return
              }
              setModel?.(v)
            }}
            options={(() => {
              return [model, ...(models || [])]
                .filter((v, i, a) => v && a.indexOf(v) === i)
                .map((v) => {
                  const modalities = getModelModalities(v, customProviders)
                  const cot = supportsChainOfThought(v, customProviders)
                  const isApiModel = customProviders?.[v]
                  
                  return {
                    label: (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'space-between' }}>
                        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', gap: 6 }}>
                          {isApiModel && (
                            <KeyOutlined style={{ fontSize: 12, color: '#8b5cf6' }} title="API Key 模型" />
                          )}
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{v}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          {modalities.text && (
                            <FileTextOutlined style={{ fontSize: 12, color: '#3b82f6' }} title="支持文本" />
                          )}
                          {modalities.image && (
                            <PictureOutlined style={{ fontSize: 12, color: '#10b981' }} title="支持图像" />
                          )}
                          {modalities.audio && (
                            <SoundOutlined style={{ fontSize: 12, color: '#f59e0b' }} title="支持音频" />
                          )}
                          {cot && (
                            <ThunderboltOutlined style={{ fontSize: 12, color: '#ef4444' }} title="支持思维链" />
                          )}
                        </div>
                      </div>
                    ),
                    value: v,
                    disabled: (requireMultimodal && !isModelMultimodal(v, customProviders))
                  }
                })
            })()}
          />
          <Button onClick={() => setApiModalOpen(true)}>添加 API 模型</Button>
        </div>
        <Button 
          icon={<SettingOutlined />} 
          onClick={() => setConfigPanelOpen(true)}
        >
          设置
        </Button>
      </div>
      <ModelConfigPanel 
        visible={configPanelOpen} 
        onClose={() => setConfigPanelOpen(false)} 
      />
      <Modal
        title="添加 API 模型"
        open={apiModalOpen}
        onCancel={() => setApiModalOpen(false)}
        onOk={handleAddApiModel}
        okText="添加"
        cancelText="取消"
      >
        <div style={{ display: 'grid', gap: 8 }}>
          <Input placeholder="模型 ID" value={apiModelId} onChange={(e) => setApiModelId(e.target.value)} />
          <Input placeholder="Base URL" value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
          <Input.Password placeholder="API Key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
        </div>
      </Modal>
    </>
  )
}

export default ModelPickerToolbar

