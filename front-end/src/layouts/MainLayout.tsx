import React from 'react'
import { Outlet } from 'react-router-dom'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css';
import NavBar from '../components/NavBar'

const MainLayout = () => {
  return (
    <div className='flex h-screen bg-[#f7f5eb] text-gray-900 overflow-hidden'>
    <div className='fixed inset-0 z-0'>
        <div className='absolute inset-0 bg-gradient-to-br from-[#f7f5eb] via-[#f2f0e6] to-[#f7f5eb] opacity-80' />
        <div className='absolute inset-0 backdrop-blur-sm' />
    </div>
     <NavBar />
      <Outlet />
      <ToastContainer />
    </div>
  )
}

export default MainLayout
