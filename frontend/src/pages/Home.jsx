import React from 'react'
import { Divider } from 'antd';
import styles from './Home.module.css'
import AgentPage from '../components/AgentPage'

export default function Home() {
  return (
    <div className={styles.container}>
      <AgentPage />
    </div>
  )
}

