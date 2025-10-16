import React, { useState } from 'react'
import { Select, Button, Modal, Input, message } from 'antd'
import styles from '../AgentPage.module.css'
import { isModelMultimodal } from '../../utils/modelUtils'

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
                .map((v) => ({
                  label: `${v}${isModelMultimodal(v, customProviders) ? ' (multimodal)' : ' (text-only)'}`,
                  value: v,
                  disabled: (requireMultimodal && !isModelMultimodal(v, customProviders))
                }))
            })()}
          />
          <Button onClick={() => setApiModalOpen(true)}>Add API model</Button>
        </div>
      </div>
      <Modal
        title="Add API model"
        open={apiModalOpen}
        onCancel={() => setApiModalOpen(false)}
        onOk={handleAddApiModel}
      >
        <div style={{ display: 'grid', gap: 8 }}>
          <Input placeholder="Model ID" value={apiModelId} onChange={(e) => setApiModelId(e.target.value)} />
          <Input placeholder="Base URL" value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
          <Input.Password placeholder="API Key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
        </div>
      </Modal>
    </>
  )
}

export default ModelPickerToolbar

