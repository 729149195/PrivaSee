import React from 'react'
import styles from '../AgentPage.module.css'
import { getInfonColor, getMatchKeywords, getRelatedInfons } from '../../utils/infonUtils'

/**
 * 关系标签组件
 * 显示关系信息元及其关联的信息元
 * @param {Array} relations - 关系信息元数组 [{ infon, run }]
 * @param {object} infonIndex - 信息元索引对象
 * @param {object} style - 自定义样式
 */
const RelationTags = ({ relations, infonIndex, style }) => {
  if (!relations || relations.length === 0) return null

  return (
    <div className={styles.relationTags} style={style}>
      {relations.map(({ infon }, idx) => {
        const relatedInfons = getRelatedInfons(infon, infonIndex)
        const color = getInfonColor('REL')
        
        return (
          <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
            <span className={styles.relationTagName} style={{ color: color }}>
              {infon.relation_name || 'Relation'}
            </span>
            <span className={styles.relationTagArgs}>
              {relatedInfons.map((rel, ri) => {
                const relColor = getInfonColor(rel.infon_type)
                const keywords = getMatchKeywords(rel)
                const label = keywords[0] || rel.iid
                return (
                  <React.Fragment key={ri}>
                    {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                    <span className={styles.relationTagArg} style={{ color: relColor }}>
                      {label}
                    </span>
                  </React.Fragment>
                )
              })}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default RelationTags

