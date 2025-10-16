import React from 'react'
import styles from '../AgentPage.module.css'
import { getInfonColor } from '../../utils/infonUtils'

/**
 * 信息元类型图例组件
 * 显示不同信息元类型及其对应颜色
 */
const InfonLegend = () => {
  return (
    <div className={styles.infonLegend}>
      <div className={styles.legendItem}>
        <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('DESC') }}></span>
        <span className={styles.legendLabel}>Description (DESC)</span>
      </div>
      <div className={styles.legendItem}>
        <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('SCEN') }}></span>
        <span className={styles.legendLabel}>Scenario (SCEN)</span>
      </div>
      <div className={styles.legendItem}>
        <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('REL') }}></span>
        <span className={styles.legendLabel}>Relation (REL)</span>
      </div>
    </div>
  )
}

export default InfonLegend

