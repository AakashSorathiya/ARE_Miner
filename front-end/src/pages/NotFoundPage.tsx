import { Link } from 'react-router-dom'
import { FaExclamationTriangle } from 'react-icons/fa'
import { motion } from 'framer-motion'

const NotFoundPage = () => {
  return (
    <motion.div className='flex-1 flex flex-col items-center justify-center p-6 z-10'>
        <FaExclamationTriangle className='text-yellow-400 text-6xl mb-4' />
        <h1 className='text-6xl font-bold mb-4'>404 Not Found</h1>
        <p className='text-xl mb-5'>This page does not exist</p>

        <Link
        to="/"
        className="text-white bg-indigo-700 hover:bg-indigo-900 rounded-md px-3 py-2 mt-4"
        
        >
        
        Go Back
        
        </Link>

    </motion.div>
  )
}

export default NotFoundPage
