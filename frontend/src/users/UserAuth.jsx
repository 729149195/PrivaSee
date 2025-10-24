import React, { useState } from 'react'
import { Modal, Input, Button, Tabs, message, Avatar, Dropdown } from 'antd'
import { UserOutlined, LoginOutlined, LogoutOutlined, DeleteOutlined } from '@ant-design/icons'
import { useUserStore, validateLogin } from './userStore'
import { useStore } from '../store'
import styles from './UserAuth.module.css'

export default function UserAuth() {
  const { currentUser, isLoggedIn, login, logout, register } = useUserStore()
  const { clearAllData } = useStore()
  const [modalOpen, setModalOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('login')
  
  // 登录表单
  const [loginUsername, setLoginUsername] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [loginLoading, setLoginLoading] = useState(false)
  
  // 注册表单
  const [registerUsername, setRegisterUsername] = useState('')
  const [registerPassword, setRegisterPassword] = useState('')
  const [registerConfirmPassword, setRegisterConfirmPassword] = useState('')
  const [registerLoading, setRegisterLoading] = useState(false)

  // 处理登录
  const handleLogin = async () => {
    if (!loginUsername || !loginPassword) {
      message.warning('请填写完整的登录信息')
      return
    }
    
    setLoginLoading(true)
    try {
      // 模拟异步验证
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const result = validateLogin(loginUsername, loginPassword)
      if (result.success) {
        login(result.user)
        message.success('登录成功！')
        setModalOpen(false)
        setLoginUsername('')
        setLoginPassword('')
      } else {
        message.error(result.error)
      }
    } catch (error) {
      message.error('登录失败，请重试')
    } finally {
      setLoginLoading(false)
    }
  }

  // 处理注册
  const handleRegister = async () => {
    if (!registerUsername || !registerPassword || !registerConfirmPassword) {
      message.warning('请填写完整的注册信息')
      return
    }
    
    if (registerPassword !== registerConfirmPassword) {
      message.error('两次输入的密码不一致')
      return
    }
    
    if (registerPassword.length < 6) {
      message.error('密码长度至少为6位')
      return
    }
    
    setRegisterLoading(true)
    try {
      // 模拟异步注册
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // 检查用户名是否已存在
      const users = JSON.parse(localStorage.getItem('privasee_users') || '{}')
      if (users[registerUsername]) {
        message.error('该用户名已被注册')
        setRegisterLoading(false)
        return
      }
      
      register(registerUsername, registerPassword)
      message.success('注册成功！')
      setModalOpen(false)
      setRegisterUsername('')
      setRegisterPassword('')
      setRegisterConfirmPassword('')
    } catch (error) {
      message.error('注册失败，请重试')
    } finally {
      setRegisterLoading(false)
    }
  }

  // 处理退出
  const handleLogout = () => {
    Modal.confirm({
      title: '确认退出',
      content: '退出后未保存的对话历史将会丢失，确定要退出吗？',
      okText: '确定',
      cancelText: '取消',
      onOk: () => {
        logout()
        message.success('已退出登录')
      }
    })
  }
  
  // 处理清除全部记录
  const handleClearAll = () => {
    Modal.confirm({
      title: '确认清除全部记录',
      content: '此操作将清除所有会话、信息元、隐私推理结果等数据，且无法恢复。确定要继续吗？',
      okText: '确定',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: () => {
        clearAllData()
        message.success('已清除全部记录')
      }
    })
  }

  // 未登录状态的下拉菜单
  const guestMenuItems = [
    {
      key: 'login',
      icon: <LoginOutlined />,
      label: '登录',
      onClick: () => {
        setActiveTab('login')
        setModalOpen(true)
      }
    }
  ]

  // 已登录状态的下拉菜单
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: <div>
        <div style={{ fontWeight: 600 }}>{currentUser?.username}</div>
        <div style={{ fontSize: '12px', color: '#94a3b8' }}>ID: {currentUser?.id?.slice(0, 15)}...</div>
      </div>,
      disabled: true
    },
    {
      type: 'divider'
    },
    {
      key: 'clear',
      icon: <DeleteOutlined />,
      label: '清除全部记录',
      onClick: handleClearAll,
      danger: true
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout
    }
  ]

  return (
    <>
      <div className={styles.userAuth}>
        <Dropdown 
          menu={{ items: isLoggedIn ? userMenuItems : guestMenuItems }}
          placement="bottomRight"
          trigger={['click']}
        >
          {isLoggedIn ? (
            <div className={styles.userAvatar}>
              <Avatar 
                src={currentUser?.avatar} 
                size={32}
                style={{ cursor: 'pointer', border: '2px solid #e2e8f0' }}
              />
            </div>
          ) : (
            <Button 
              type="text" 
              icon={<UserOutlined />}
              className={styles.loginBtn}
              onClick={() => setModalOpen(true)}
            >
              登录
            </Button>
          )}
        </Dropdown>
      </div>

      <Modal
        title={null}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        footer={null}
        width={400}
        centered
        className={styles.authModal}
      >
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          centered
          className={styles.authTabs}
          items={[
            {
              key: 'login',
              label: '登录',
              children: (
                <div className={styles.authForm}>
                  <div className={styles.formGroup}>
                    <label>用户名</label>
                    <Input
                      size="large"
                      placeholder="请输入用户名"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      onPressEnter={handleLogin}
                      prefix={<UserOutlined />}
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label>密码</label>
                    <Input.Password
                      size="large"
                      placeholder="请输入密码"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      onPressEnter={handleLogin}
                    />
                  </div>
                  <Button 
                    type="primary" 
                    size="large" 
                    block
                    loading={loginLoading}
                    onClick={handleLogin}
                    className={styles.submitBtn}
                  >
                    登录
                  </Button>
                  <div className={styles.formHint}>
                    还没有账号？<a onClick={() => setActiveTab('register')}>立即注册</a>
                  </div>
                </div>
              )
            },
            {
              key: 'register',
              label: '注册',
              children: (
                <div className={styles.authForm}>
                  <div className={styles.formGroup}>
                    <label>用户名</label>
                    <Input
                      size="large"
                      placeholder="请输入用户名"
                      value={registerUsername}
                      onChange={(e) => setRegisterUsername(e.target.value)}
                      prefix={<UserOutlined />}
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label>密码</label>
                    <Input.Password
                      size="large"
                      placeholder="请输入密码（至少6位）"
                      value={registerPassword}
                      onChange={(e) => setRegisterPassword(e.target.value)}
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label>确认密码</label>
                    <Input.Password
                      size="large"
                      placeholder="请再次输入密码"
                      value={registerConfirmPassword}
                      onChange={(e) => setRegisterConfirmPassword(e.target.value)}
                      onPressEnter={handleRegister}
                    />
                  </div>
                  <Button 
                    type="primary" 
                    size="large" 
                    block
                    loading={registerLoading}
                    onClick={handleRegister}
                    className={styles.submitBtn}
                  >
                    注册
                  </Button>
                  <div className={styles.formHint}>
                    已有账号？<a onClick={() => setActiveTab('login')}>立即登录</a>
                  </div>
                </div>
              )
            }
          ]}
        />
      </Modal>
    </>
  )
}

