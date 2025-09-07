import React from 'react'
import { Divider } from 'antd';
import styles from './Home.module.css'
import TextTest from '../components/TextTest'
import ImgTest from '../components/ImgTest'
import AudioTest from '../components/AudioTest'
import ConversationViz from '../components/ConversationViz'
import AgentPage from '../components/AgentPage'

export default function Home() {
  return (
    <div className={styles.container}>
      <AgentPage />
    </div>
  )
}