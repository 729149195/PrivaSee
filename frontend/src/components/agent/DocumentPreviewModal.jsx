import React, { useState, useEffect } from 'react'
import { Modal, Table, Spin, Carousel } from 'antd'
import { FileTextOutlined, DownloadOutlined, LeftOutlined, RightOutlined } from '@ant-design/icons'
import mammoth from 'mammoth'
import * as XLSX from 'xlsx'
import JSZip from 'jszip'

/**
 * 解析 PPTX 文件
 * @param {ArrayBuffer} arrayBuffer - PPTX 文件的 ArrayBuffer
 * @returns {Promise<Array>} - 幻灯片数组
 */
async function parsePPTX(arrayBuffer) {
  const zip = await JSZip.loadAsync(arrayBuffer)
  const slides = []
  
  // 读取所有幻灯片文件
  const slideFiles = []
  zip.folder('ppt/slides').forEach((relativePath, file) => {
    if (relativePath.match(/^slide\d+\.xml$/)) {
      slideFiles.push({ name: relativePath, file })
    }
  })
  
  // 按数字排序
  slideFiles.sort((a, b) => {
    const numA = parseInt(a.name.match(/\d+/)[0])
    const numB = parseInt(b.name.match(/\d+/)[0])
    return numA - numB
  })
  
  // 读取媒体文件（图片）
  const mediaFiles = {}
  const mediaFolder = zip.folder('ppt/media')
  if (mediaFolder) {
    for (const [relativePath, file] of Object.entries(mediaFolder.files)) {
      if (!file.dir && relativePath.match(/\.(png|jpg|jpeg|gif|bmp|svg)$/i)) {
        const blob = await file.async('blob')
        const url = URL.createObjectURL(blob)
        const fileName = relativePath.split('/').pop()
        mediaFiles[fileName] = url
      }
    }
  }
  
  // 解析每个幻灯片
  for (let i = 0; i < slideFiles.length; i++) {
    const slideFile = slideFiles[i].file
    const xmlText = await slideFile.async('text')
    
    // 提取文本内容
    const texts = extractTextsFromSlideXML(xmlText)
    
    // 提取图片引用
    const imageRefs = extractImageRefsFromSlideXML(xmlText)
    const images = []
    
    // 读取对应的关系文件以获取图片文件名
    const slideNum = slideFiles[i].name.match(/\d+/)[0]
    const relsPath = `ppt/slides/_rels/slide${slideNum}.xml.rels`
    const relsFile = zip.file(relsPath)
    
    if (relsFile && imageRefs.length > 0) {
      const relsXml = await relsFile.async('text')
      
      for (const ref of imageRefs) {
        const match = relsXml.match(new RegExp(`Id="${ref}"[^>]*Target="[^"]*\/media\/([^"]+)"`))
        if (match && match[1]) {
          const mediaFileName = match[1]
          if (mediaFiles[mediaFileName]) {
            images.push(mediaFiles[mediaFileName])
          }
        }
      }
    }
    
    slides.push({
      number: i + 1,
      texts,
      images
    })
  }
  
  return slides
}

/**
 * 从幻灯片 XML 中提取文本内容
 */
function extractTextsFromSlideXML(xmlText) {
  const texts = []
  const parser = new DOMParser()
  const xmlDoc = parser.parseFromString(xmlText, 'text/xml')
  
  // 查找所有文本元素 <a:t>
  const textElements = xmlDoc.getElementsByTagName('a:t')
  for (let i = 0; i < textElements.length; i++) {
    const text = textElements[i].textContent.trim()
    if (text) {
      texts.push(text)
    }
  }
  
  return texts
}

/**
 * 从幻灯片 XML 中提取图片引用 ID
 */
function extractImageRefsFromSlideXML(xmlText) {
  const refs = []
  const blipMatches = xmlText.matchAll(/<a:blip[^>]*r:embed="([^"]+)"/g)
  
  for (const match of blipMatches) {
    refs.push(match[1])
  }
  
  return refs
}

/**
 * 文档预览 Modal
 * @param {object} file - 文件数据 {id, name, size, type}
 * @param {File} fileObject - File 对象（用于生成预览）
 * @param {function} onClose - 关闭回调
 */
const DocumentPreviewModal = ({ file, fileObject, onClose }) => {
  const [previewUrl, setPreviewUrl] = useState(null)
  const [previewContent, setPreviewContent] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [previewError, setPreviewError] = useState(null)

  useEffect(() => {
    if (!fileObject || !(fileObject instanceof File)) {
      setPreviewUrl(null)
      setPreviewContent(null)
      setPreviewError(null)
      return
    }

    const processFile = async () => {
      setIsLoading(true)
      setPreviewError(null)
      setPreviewContent(null)
      setPreviewUrl(null)

      try {
        const fileType = file?.type || fileObject.type
        
        // PDF 文件：使用 blob URL
        if (fileType === 'application/pdf') {
          const objectUrl = URL.createObjectURL(fileObject)
          setPreviewUrl(objectUrl)
        }
        // 图片文件：使用 blob URL
        else if (fileType.startsWith('image/')) {
          const objectUrl = URL.createObjectURL(fileObject)
          setPreviewUrl(objectUrl)
        }
        // Word 文档：使用 mammoth
        else if (
          fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || // .docx
          fileType === 'application/msword' || // .doc
          file?.name?.toLowerCase().endsWith('.docx') ||
          file?.name?.toLowerCase().endsWith('.doc')
        ) {
          const arrayBuffer = await fileObject.arrayBuffer()
          const result = await mammoth.convertToHtml({ arrayBuffer })
          setPreviewContent({ type: 'word', html: result.value })
        }
        // Excel 文档：使用 xlsx
        else if (
          fileType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || // .xlsx
          fileType === 'application/vnd.ms-excel' || // .xls
          file?.name?.toLowerCase().endsWith('.xlsx') ||
          file?.name?.toLowerCase().endsWith('.xls')
        ) {
          const arrayBuffer = await fileObject.arrayBuffer()
          const workbook = XLSX.read(arrayBuffer, { type: 'array' })
          const sheets = []
          
          workbook.SheetNames.forEach(sheetName => {
            const worksheet = workbook.Sheets[sheetName]
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '' })
            sheets.push({ name: sheetName, data: jsonData })
          })
          
          setPreviewContent({ type: 'excel', sheets })
        }
        // PowerPoint 文档：使用 jszip 解析
        else if (
          fileType === 'application/vnd.openxmlformats-officedocument.presentationml.presentation' || // .pptx
          file?.name?.toLowerCase().endsWith('.pptx')
        ) {
          const arrayBuffer = await fileObject.arrayBuffer()
          const slides = await parsePPTX(arrayBuffer)
          setPreviewContent({ type: 'pptx', slides })
        }
        // 旧版 PPT 不支持
        else if (
          fileType === 'application/vnd.ms-powerpoint' || // .ppt
          file?.name?.toLowerCase().endsWith('.ppt')
        ) {
          setPreviewContent({ type: 'unsupported', message: '旧版 .ppt 格式不支持预览，请转换为 .pptx 格式或下载后查看' })
        }
        // 其他类型
        else {
          setPreviewContent({ type: 'unsupported', message: '此文件类型暂不支持预览' })
        }
      } catch (error) {
        console.error('文档预览失败:', error)
        setPreviewError(error.message || '预览加载失败')
      } finally {
        setIsLoading(false)
      }
    }

    processFile()

    // Cleanup
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [fileObject, file])

  if (!file) return null

  const getFileSizeText = (fileData) => {
    const bytes = fileData.size || 0
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`
  }

  return (
    <Modal
      open={!!file}
      onCancel={onClose}
      footer={null}
      title={file?.name || 'Document preview'}
      width="80%"
      style={{ maxWidth: '1200px', top: 20 }}
      styles={{
        content: { borderRadius: 12, overflow: 'hidden' },
        header: { padding: '10px 16px', margin: 0, borderBottom: '1px solid #f0f0f0' },
        body: { padding: 0, height: 'calc(100vh - 180px)', overflow: 'hidden' }
      }}
      maskClosable
    >
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* 文件预览内容 - 固定高度、无额外滚动 */}
        <div style={{ flex: 1, overflow: 'auto', padding: '0', backgroundColor: '#fff' }}>
          {isLoading ? (
            <div style={{ 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '100%',
              padding: '60px 20px'
            }}>
              <Spin size="large" />
              <p style={{ marginTop: '16px', color: '#8c8c8c' }}>正在加载预览...</p>
            </div>
          ) : previewError ? (
            <div style={{ 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '100%',
              textAlign: 'center', 
              padding: '60px 20px', 
              color: '#ff4d4f' 
            }}>
              <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px' }} />
              <p>预览加载失败</p>
              <p style={{ fontSize: '12px', marginTop: '8px' }}>
                {previewError}
              </p>
            </div>
          ) : previewUrl ? (
            file.type === 'application/pdf' ? (
              <embed
                src={previewUrl}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ minHeight: '100%', border: 'none', display: 'block' }}
              />
            ) : file.type.startsWith('image/') ? (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                height: '100%',
                padding: '20px',
                overflow: 'auto'
              }}>
                <img 
                  src={previewUrl} 
                  alt={file.name} 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    height: 'auto',
                    objectFit: 'contain'
                  }} 
                />
              </div>
            ) : null
          ) : previewContent ? (
            previewContent.type === 'word' ? (
              <div 
                style={{ 
                  padding: '40px',
                  maxWidth: '900px',
                  margin: '0 auto',
                  backgroundColor: '#fff',
                  minHeight: '100%',
                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                  fontSize: '14px',
                  lineHeight: '1.8',
                  color: '#262626'
                }}
                dangerouslySetInnerHTML={{ __html: previewContent.html }}
              />
            ) : previewContent.type === 'excel' ? (
              <div style={{ padding: '20px' }}>
                {previewContent.sheets.map((sheet, sheetIdx) => (
                  <div key={sheetIdx} style={{ marginBottom: '30px' }}>
                    <h3 style={{ 
                      marginBottom: '12px', 
                      fontSize: '16px', 
                      fontWeight: 600,
                      color: '#262626',
                      borderBottom: '2px solid #1890ff',
                      paddingBottom: '8px'
                    }}>
                      {sheet.name}
                    </h3>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '12px',
                        backgroundColor: '#fff',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                      }}>
                        <tbody>
                          {sheet.data.map((row, rowIdx) => (
                            <tr key={rowIdx} style={{
                              backgroundColor: rowIdx === 0 ? '#fafafa' : rowIdx % 2 === 0 ? '#fff' : '#fafafa'
                            }}>
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} style={{
                                  border: '1px solid #e8e8e8',
                                  padding: '8px 12px',
                                  fontWeight: rowIdx === 0 ? 600 : 400,
                                  color: rowIdx === 0 ? '#262626' : '#595959',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word'
                                }}>
                                  {cell !== null && cell !== undefined ? String(cell) : ''}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            ) : previewContent.type === 'pptx' ? (
              <div style={{ padding: '20px', backgroundColor: '#f5f5f5' }}>
                <div style={{ 
                  marginBottom: '16px', 
                  textAlign: 'center',
                  color: '#8c8c8c',
                  fontSize: '13px'
                }}>
                  共 {previewContent.slides.length} 页幻灯片
                </div>
                {previewContent.slides.map((slide, slideIdx) => (
                  <div 
                    key={slideIdx} 
                    style={{ 
                      marginBottom: '24px',
                      backgroundColor: '#fff',
                      borderRadius: '8px',
                      padding: '20px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
                    }}
                  >
                    {/* 幻灯片页码 */}
                    <div style={{
                      fontSize: '14px',
                      fontWeight: 600,
                      color: '#1890ff',
                      marginBottom: '16px',
                      paddingBottom: '8px',
                      borderBottom: '2px solid #e8e8e8'
                    }}>
                      第 {slide.number} 页
                    </div>
                    
                    {/* 幻灯片图片 */}
                    {slide.images && slide.images.length > 0 && (
                      <div style={{ marginBottom: '16px' }}>
                        {slide.images.map((imageUrl, imgIdx) => (
                          <img 
                            key={imgIdx}
                            src={imageUrl}
                            alt={`Slide ${slide.number} - Image ${imgIdx + 1}`}
                            style={{
                              maxWidth: '100%',
                              height: 'auto',
                              display: 'block',
                              margin: '8px 0',
                              borderRadius: '4px',
                              border: '1px solid #e8e8e8'
                            }}
                          />
                        ))}
                      </div>
                    )}
                    
                    {/* 幻灯片文本内容 */}
                    {slide.texts && slide.texts.length > 0 && (
                      <div style={{
                        backgroundColor: '#fafafa',
                        padding: '16px',
                        borderRadius: '4px',
                        border: '1px solid #e8e8e8'
                      }}>
                        {slide.texts.map((text, textIdx) => (
                          <div 
                            key={textIdx}
                            style={{
                              marginBottom: textIdx < slide.texts.length - 1 ? '8px' : '0',
                              fontSize: '13px',
                              lineHeight: '1.6',
                              color: '#262626',
                              wordBreak: 'break-word'
                            }}
                          >
                            {text}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* 如果没有内容 */}
                    {(!slide.texts || slide.texts.length === 0) && 
                     (!slide.images || slide.images.length === 0) && (
                      <div style={{
                        textAlign: 'center',
                        padding: '40px',
                        color: '#bfbfbf',
                        fontSize: '13px'
                      }}>
                        此页无可显示的内容
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : previewContent.type === 'unsupported' ? (
              <div style={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100%',
                textAlign: 'center', 
                padding: '60px 20px', 
                color: '#8c8c8c' 
              }}>
                <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px', color: '#d9d9d9' }} />
                <p style={{ fontSize: '14px', marginBottom: '8px' }}>{previewContent.message}</p>
                <p style={{ fontSize: '12px', color: '#bfbfbf' }}>
                  文件名: {file.name}
                </p>
              </div>
            ) : null
          ) : (
            <div style={{ 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '100%',
              textAlign: 'center', 
              padding: '60px 20px', 
              color: '#8c8c8c' 
            }}>
              <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px' }} />
              <p>预览不可用</p>
              <p style={{ fontSize: '12px', marginTop: '8px' }}>
                {!fileObject ? '文件内容仅在会话期间可用，刷新页面后将不可用。' : '正在加载...'}
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

export default DocumentPreviewModal

