import React from 'react'
import { Divider } from 'antd';
import styles from './Home.module.css'
import TextTest from '../components/TextTest'
import ImgTest from '../components/ImgTest'
import AudioTest from '../components/AudioTest'

export default function Home() {
  return (
    <div className={styles.container}>
      <TextTest />
      <Divider />
      <ImgTest />
      <Divider />
      <AudioTest />
    </div>
  )
}