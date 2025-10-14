import React, { useState, useEffect } from 'react'
import { Modal, Select, Button, message, Divider, Space, Upload, Input } from 'antd'
import { DownloadOutlined, UploadOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { useStore } from '../store'
import { useUserStore } from './userStore'
import styles from './SettingsModal.module.css'

export default function SettingsModal({ open, onClose }) {
  const {
    models,
    customProviders,
    addApiModel,
    infonExtractionModel,
    privacyInferenceModel,
    setInfonExtractionModel,
    setPrivacyInferenceModel,
    sessions,
    infonSessions,
    privacyInferences,
    customPrivacyItems,
    selectedLawIdx,
    selectedPrivacyItems
  } = useStore()
  
  const { currentUser } = useUserStore()
  
  // 本地状态：模型配置
  const [localInfonModel, setLocalInfonModel] = useState('')
  const [localPrivacyModel, setLocalPrivacyModel] = useState('')
  
  // 本地状态：添加自定义模型
  const [showAddModel, setShowAddModel] = useState(false)
  const [newModelId, setNewModelId] = useState('')
  const [newBaseUrl, setNewBaseUrl] = useState('')
  const [newApiKey, setNewApiKey] = useState('')
  
  // 初始化：从 store 加载当前配置
  useEffect(() => {
    if (open) {
      setLocalInfonModel(infonExtractionModel || 'deepseek-chat')
      setLocalPrivacyModel(privacyInferenceModel || 'deepseek-chat')
    }
  }, [open, infonExtractionModel, privacyInferenceModel])
  
  // 构建模型选项列表：包括 ollama 模型和自定义 API 模型
  const modelOptions = React.useMemo(() => {
    const allModels = [...new Set([...(models || []), ...Object.keys(customProviders || {})])]
    return allModels.map(id => {
      const isCustom = customProviders?.[id]
      return {
        label: isCustom ? `${id} (API)` : id,
        value: id
      }
    })
  }, [models, customProviders])
  
  // 保存配置
  const handleSave = () => {
    setInfonExtractionModel(localInfonModel)
    setPrivacyInferenceModel(localPrivacyModel)
    message.success('设置已保存')
    onClose()
  }
  
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
          // 移除 controller 和 _hash
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
        infonExtractionModel: infonExtractionModel || 'deepseek-chat',
        privacyInferenceModel: privacyInferenceModel || 'deepseek-chat'
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
                currentSessionId: importedData.sessions?.[0]?.id || null
              })
              
              // 导入模型配置
              if (importedData.infonExtractionModel) {
                setInfonExtractionModel(importedData.infonExtractionModel)
                setLocalInfonModel(importedData.infonExtractionModel)
              }
              if (importedData.privacyInferenceModel) {
                setPrivacyInferenceModel(importedData.privacyInferenceModel)
                setLocalPrivacyModel(importedData.privacyInferenceModel)
              }
              
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
  
  return (
    <Modal
      title="设置"
      open={open}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          取消
        </Button>,
        <Button key="save" type="primary" onClick={handleSave}>
          保存
        </Button>
      ]}
      width={600}
      className={styles.settingsModal}
    >
      <div className={styles.settingsContent}>
        {/* 模型配置 */}
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>模型配置</h3>
          
          <div className={styles.formItem}>
            <label className={styles.label}>信息元提取模型</label>
            <Select
              style={{ width: '100%' }}
              value={localInfonModel}
              onChange={setLocalInfonModel}
              options={modelOptions}
              placeholder="选择用于提取信息元的模型"
            />
            <div className={styles.hint}>
              用于从用户输入中提取隐私相关信息元（DESC、SCEN、REL、SIT）
            </div>
          </div>
          
          <div className={styles.formItem}>
            <label className={styles.label}>隐私推理模型</label>
            <Select
              style={{ width: '100%' }}
              value={localPrivacyModel}
              onChange={setLocalPrivacyModel}
              options={modelOptions}
              placeholder="选择用于隐私推理的模型"
            />
            <div className={styles.hint}>
              用于基于提取的信息元进行隐私风险推理和分析
            </div>
          </div>
        </div>
        
        <Divider />
        
        {/* 自定义模型管理 */}
        <div className={styles.section}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 className={styles.sectionTitle}>自定义模型</h3>
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
            <div className={styles.addModelForm}>
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
            <div className={styles.modelList}>
              {Object.entries(customProviders).map(([id, config]) => (
                <div key={id} className={styles.modelItem}>
                  <div>
                    <div className={styles.modelName}>{id}</div>
                    <div className={styles.modelUrl}>{config.baseUrl}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        <Divider />
        
        {/* 数据管理 */}
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>数据管理</h3>
          
          <div className={styles.dataManagement}>
            <div className={styles.buttonRow}>
              <Button 
                icon={<DownloadOutlined />} 
                onClick={handleExportData}
                disabled={!currentUser?.id}
                className={styles.halfButton}
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
                  className={styles.halfButton}
                >
                  导入历史数据
                </Button>
              </Upload>
            </div>
            
            {!currentUser?.id && (
              <div className={styles.hint} style={{ color: '#ff4d4f', marginTop: 12 }}>
                请先登录后才能使用数据导入/导出功能
              </div>
            )}
            
            <div className={styles.hint} style={{ marginTop: 8 }}>
              数据以 JSON 格式保存在浏览器本地，导出后可备份或迁移到其他设备
            </div>
          </div>
        </div>
      </div>
    </Modal>
  )
}

