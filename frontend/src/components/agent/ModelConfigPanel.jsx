import React, { useState } from 'react'
import { Modal, Select, Divider, Button, message, Popconfirm, Input, Space, Upload } from 'antd'
import { DeleteOutlined, FileTextOutlined, PictureOutlined, SoundOutlined, ThunderboltOutlined, PlusOutlined, DownloadOutlined, UploadOutlined, KeyOutlined, UndoOutlined } from '@ant-design/icons'
import { useStore } from '../../store'
import { useUserStore } from '../../users/userStore'
import { getModelModalities, supportsChainOfThought } from '../../utils/modelUtils'
import styles from './ModelConfigPanel.module.css'

/**
 * 模型配置面板组件
 * 用于配置各阶段使用的模型、管理自定义模型和数据
 */
const ModelConfigPanel = ({ visible, onClose }) => {
  const {
    models,
    customProviders,
    directInferenceModel,
    infonExtractionModel,
    infonPrivacyInferenceModel,
    imageParsingModel,
    protectionSuggestionModel,
    setDirectInferenceModel,
    setInfonExtractionModel,
    setInfonPrivacyInferenceModel,
    setImageParsingModel,
    setProtectionSuggestionModel,
    resetToDefaultModels,
    removeApiModel,
    addApiModel,
    sessions,
    infonSessions,
    privacyInferences,
    customPrivacyItems,
    selectedLawIdx,
    selectedPrivacyItems,
  } = useStore()
  
  const { currentUser } = useUserStore()
  
  // 添加自定义模型的状态
  const [showAddModel, setShowAddModel] = useState(false)
  const [newModelId, setNewModelId] = useState('')
  const [newBaseUrl, setNewBaseUrl] = useState('')
  const [newApiKey, setNewApiKey] = useState('')

  // 获取所有可用模型（去重）
  const allModels = Array.from(new Set([...models]))

  // 只获取API key模型（用于Protection Suggestion）
  const apiKeyModels = allModels.filter(id => customProviders?.[id])
  
  // 添加自定义模型
  const handleAddCustomModel = () => {
    if (!newModelId.trim() || !newBaseUrl.trim() || !newApiKey.trim()) {
      message.warning('请填写完整的模型信息')
      return
    }
    
    try {
      addApiModel({ id: newModelId.trim(), baseUrl: newBaseUrl.trim(), apiKey: newApiKey.trim() })
      message.success(`模型 ${newModelId} 已添加`)
      setNewModelId('')
      setNewBaseUrl('')
      setNewApiKey('')
      setShowAddModel(false)
    } catch (err) {
      message.error('添加模型失败')
    }
  }
  
  // 导出用户数据
  const handleExportData = () => {
    if (!currentUser?.id) {
      message.warning('请先登录后再导出数据')
      return
    }
    
    try {
      // 清理不可序列化的字段
      const cleanInfonSessions = {}
      Object.keys(infonSessions || {}).forEach(sessionId => {
        const session = infonSessions[sessionId]
        if (session?.runs) {
          cleanInfonSessions[sessionId] = {
            runs: session.runs.map(run => {
              const { controller, ...rest } = run
              return rest
            })
          }
        }
      })
      
      const cleanPrivacyInferences = {}
      Object.keys(privacyInferences || {}).forEach(sessionId => {
        const inference = privacyInferences[sessionId]
        if (inference) {
          const { abortController, ...rest } = inference
          cleanPrivacyInferences[sessionId] = rest
        }
      })
      
      const exportData = {
        version: '1.0.0',
        exportTime: new Date().toISOString(),
        userId: currentUser.id,
        username: currentUser.username,
        sessions: sessions || [],
        infonSessions: cleanInfonSessions,
        privacyInferences: cleanPrivacyInferences,
        customPrivacyItems: customPrivacyItems || [],
        selectedLawIdx: selectedLawIdx || 0,
        selectedPrivacyItems: selectedPrivacyItems || [],
        directInferenceModel,
        infonExtractionModel,
        infonPrivacyInferenceModel,
        imageParsingModel,
        protectionSuggestionModel,
      }
      
      const dataStr = JSON.stringify(exportData, null, 2)
      const blob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `privasee_data_${currentUser.username}_${Date.now()}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      
      message.success('数据导出成功')
    } catch (err) {
      console.error('导出失败:', err)
      message.error('数据导出失败')
    }
  }
  
  // 导入用户数据
  const handleImportData = (file) => {
    const reader = new FileReader()
    
    reader.onload = (e) => {
      try {
        const importedData = JSON.parse(e.target.result)
        
        // 验证数据格式
        if (!importedData.version || !importedData.sessions) {
          message.error('数据格式不正确')
          return
        }
        
        // 确认导入操作
        Modal.confirm({
          title: '确认导入数据',
          content: `即将导入用户 ${importedData.username || '未知'} 的数据，这将覆盖当前所有会话和设置。确定继续吗？`,
          okText: '确定',
          cancelText: '取消',
          onOk: () => {
            try {
              // 导入数据到 store
              useStore.setState({
                sessions: importedData.sessions || [],
                infonSessions: importedData.infonSessions || {},
                privacyInferences: importedData.privacyInferences || {},
                customPrivacyItems: importedData.customPrivacyItems || [],
                selectedLawIdx: importedData.selectedLawIdx || 0,
                selectedPrivacyItems: importedData.selectedPrivacyItems || [],
                currentSessionId: importedData.sessions?.[0]?.id || null,
                directInferenceModel: importedData.directInferenceModel || 'deepseek-chat',
                infonExtractionModel: importedData.infonExtractionModel || 'deepseek-chat',
                infonPrivacyInferenceModel: importedData.infonPrivacyInferenceModel || 'deepseek-chat',
                imageParsingModel: importedData.imageParsingModel || 'gemma3:12b',
                protectionSuggestionModel: importedData.protectionSuggestionModel || 'deepseek-chat',
              })
              
              message.success('数据导入成功')
            } catch (err) {
              console.error('导入失败:', err)
              message.error('数据导入失败')
            }
          }
        })
      } catch (err) {
        console.error('解析失败:', err)
        message.error('文件格式错误')
      }
    }
    
    reader.readAsText(file)
    return false // 阻止默认上传行为
  }
  
  // 恢复默认模型配置
  const handleResetToDefaultModels = () => {
    resetToDefaultModels()
    message.success('已恢复默认模型配置')
  }

  // 渲染模型选项（带模态和思维链标记）
  const renderModelOption = (id) => {
    const modalities = getModelModalities(id, customProviders)
    const cot = supportsChainOfThought(id, customProviders)
    const isApiModel = customProviders?.[id]

    return {
      value: id,
      label: (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'space-between' }}>
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center', gap: 6 }}>
            {isApiModel && (
              <KeyOutlined style={{ fontSize: 12, color: '#8b5cf6' }} title="API Key 模型" />
            )}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{id}</span>
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
      )
    }
  }

  return (
    <Modal
      title="设置"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={700}
      className={styles.modelConfigModal}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* 图标说明 */}
        <div style={{ 
          padding: 12, 
          background: 'var(--color-bg-secondary)', 
          borderRadius: 8,
          fontSize: 11,
          color: 'var(--color-text-tertiary)'
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>图标说明：</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <FileTextOutlined style={{ fontSize: 12, color: '#3b82f6' }} />
              <span>文本</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <PictureOutlined style={{ fontSize: 12, color: '#10b981' }} />
              <span>图像</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <SoundOutlined style={{ fontSize: 12, color: '#f59e0b' }} />
              <span>音频</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <ThunderboltOutlined style={{ fontSize: 12, color: '#ef4444' }} />
              <span>思维链</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <KeyOutlined style={{ fontSize: 12, color: '#8b5cf6' }} />
              <span>API Key 模型</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <DeleteOutlined style={{ fontSize: 11, color: '#94a3b8' }} />
              <span>删除</span>
            </div>
          </div>
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 直接推理模式 */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: 'var(--color-text-primary)' }}>
            直接推理模式
          </div>
          <div style={{ display: 'grid', gap: 12 }}>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
                隐私推理模型
              </div>
              <Select
                style={{ width: '100%' }}
                value={directInferenceModel}
                onChange={setDirectInferenceModel}
                options={allModels.map(renderModelOption)}
              />
            </div>
          </div>
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 提取信息元模式 */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: 'var(--color-text-primary)' }}>
            提取信息元模式
          </div>
          <div style={{ display: 'grid', gap: 12 }}>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
                信息元提取模型
              </div>
              <Select
                style={{ width: '100%' }}
                value={infonExtractionModel}
                onChange={setInfonExtractionModel}
                options={allModels.map(renderModelOption)}
              />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
                隐私推理模型
              </div>
              <Select
                style={{ width: '100%' }}
                value={infonPrivacyInferenceModel}
                onChange={setInfonPrivacyInferenceModel}
                options={allModels.map(renderModelOption)}
              />
            </div>
          </div>
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 共用模型 */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: 'var(--color-text-primary)' }}>
            共用模型
          </div>
          <div style={{ display: 'grid', gap: 12 }}>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
                图片解析模型
              </div>
              <Select
                style={{ width: '100%' }}
                value={imageParsingModel}
                onChange={setImageParsingModel}
                options={allModels.map(renderModelOption)}
              />
            </div>
            <div>
              <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
                隐私保护建议模型
              </div>
              <Select
                style={{ width: '100%' }}
                value={protectionSuggestionModel}
                onChange={setProtectionSuggestionModel}
                options={apiKeyModels.map(renderModelOption)}
                placeholder={apiKeyModels.length === 0 ? '没有可用的 API Key 模型' : '选择模型'}
                disabled={apiKeyModels.length === 0}
              />
              {apiKeyModels.length === 0 && (
                <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>
                  请至少添加一个 API Key 模型以使用隐私保护建议功能
                </div>
              )}
            </div>
          </div>
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 恢复默认模型配置 */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: 'var(--color-text-primary)' }}>
            模型配置管理
          </div>
          <Popconfirm
            title="确认恢复默认配置"
            description="这将把所有模型配置重置为默认值，确定继续吗？"
            onConfirm={handleResetToDefaultModels}
            okText="确定"
            cancelText="取消"
          >
            <Button 
              icon={<UndoOutlined />}
              style={{ width: '100%' }}
            >
              恢复默认模型配置
            </Button>
          </Popconfirm>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
            点击此按钮可将所有模型配置恢复为系统默认值
          </div>
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 自定义模型管理 */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--color-text-primary)' }}>
              自定义模型
            </div>
            <Button 
              type="dashed" 
              icon={<PlusOutlined />} 
              size="small"
              onClick={() => setShowAddModel(!showAddModel)}
            >
              添加 API 模型
            </Button>
          </div>
          
          {showAddModel && (
            <div style={{ marginBottom: 16, padding: 12, background: 'var(--color-bg-secondary)', borderRadius: 8 }}>
              <Input
                placeholder="模型 ID (如 gpt-4)"
                value={newModelId}
                onChange={(e) => setNewModelId(e.target.value)}
                style={{ marginBottom: 8 }}
              />
              <Input
                placeholder="Base URL (如 https://api.openai.com/v1)"
                value={newBaseUrl}
                onChange={(e) => setNewBaseUrl(e.target.value)}
                style={{ marginBottom: 8 }}
              />
              <Input.Password
                placeholder="API Key"
                value={newApiKey}
                onChange={(e) => setNewApiKey(e.target.value)}
                style={{ marginBottom: 8 }}
              />
              <Space>
                <Button type="primary" size="small" onClick={handleAddCustomModel}>
                  添加
                </Button>
                <Button size="small" onClick={() => {
                  setShowAddModel(false)
                  setNewModelId('')
                  setNewBaseUrl('')
                  setNewApiKey('')
                }}>
                  取消
                </Button>
              </Space>
            </div>
          )}
          
          {Object.keys(customProviders || {}).length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {Object.entries(customProviders).map(([id, config]) => (
                <div key={id} style={{ 
                  padding: 8, 
                  background: 'var(--color-bg-secondary)', 
                  borderRadius: 6,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div style={{ fontWeight: 500, fontSize: 12 }}>{id}</div>
                    <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {config.baseUrl}
                    </div>
                  </div>
                  <Popconfirm
                    title="确定删除这个模型吗？"
                    onConfirm={() => {
                      removeApiModel(id)
                      message.success(`已删除模型: ${id}`)
                    }}
                    okText="确定"
                    cancelText="取消"
                  >
                    <Button 
                      type="text" 
                      danger 
                      size="small"
                      icon={<DeleteOutlined />}
                    />
                  </Popconfirm>
                </div>
              ))}
            </div>
          )}
        </div>

        <Divider style={{ margin: 0 }} />

        {/* 数据管理 */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: 'var(--color-text-primary)' }}>
            数据管理
          </div>
          
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <Button 
              icon={<DownloadOutlined />} 
              onClick={handleExportData}
              disabled={!currentUser?.id}
              style={{ flex: 1 }}
            >
              导出我的数据
            </Button>
            
            <Upload
              accept=".json"
              beforeUpload={handleImportData}
              showUploadList={false}
              disabled={!currentUser?.id}
            >
              <Button 
                icon={<UploadOutlined />}
                disabled={!currentUser?.id}
                style={{ width: '100%' }}
              >
                导入历史数据
              </Button>
            </Upload>
          </div>
          
          {!currentUser?.id && (
            <div style={{ fontSize: 11, color: '#ef4444', marginTop: 8 }}>
              请先登录后才能使用数据导入/导出功能
            </div>
          )}
          
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
            数据以 JSON 格式保存在浏览器本地，导出后可备份或迁移到其他设备
          </div>
        </div>
      </div>
    </Modal>
  )
}

export default ModelConfigPanel

