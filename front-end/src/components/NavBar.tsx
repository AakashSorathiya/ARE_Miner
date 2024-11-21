import { BarChart2, BarChart3, Menu, Upload } from "lucide-react";
import { useState, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const NavBar = () => {
  const [isSideBarOpen, setIsSideBarOpen] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();

  // Check for available data and file
  const hasFile = () => localStorage.getItem('currentFile') !== null;
  const hasEDAData = () => localStorage.getItem('edaData') !== null;
  const hasRequirementsData = () => localStorage.getItem('requirementsData') !== null;

  const navbarItems = [
    {
      name: 'Upload CSV File',
      icon: Upload,
      color: "#2B82F6",
      href: '/',
      isActive: () => true, // Always active
      onClick: () => navigate('/')
    },
    {
      name: 'Exploratory Data Analysis',
      icon: BarChart2,
      color: "#6366f1",
      href: '/charts',
      isActive: () => hasFile(), // Active if file exists
      onClick: () => {
        if (!hasFile()) {
          toast.error('Please upload a file first');
          navigate('/');
          return;
        }
        if (hasEDAData()) {
          navigate('/charts');
        } else {
          navigate('/charts', { state: { needsData: true } });
        }
      }
    },
    {
      name: 'Extracted Requirements',
      icon: BarChart3,
      color: "#6366f1",
      href: '/extracted-requirements',
      isActive: () => hasFile(), // Active if file exists
      onClick: () => {
        if (!hasFile()) {
          toast.error('Please upload a file first');
          navigate('/');
          return;
        }
        if (hasRequirementsData()) {
          navigate('/extracted-requirements');
        } else {
          navigate('/extracted-requirements', { state: { needsData: true } });
        }
      }
    },
  ];

  // Indicator component for nav items
  const NavItemIndicator = ({ hasData }) => {
    if (!hasData) return null;
    
    return (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        className="w-2 h-2 rounded-full bg-green-400 absolute right-2 top-1/2 transform -translate-y-1/2"
      />
    );
  };

  const renderNavItem = (item) => {
    const isCurrentPath = location.pathname === item.href;
    const hasData = item.href === '/charts' ? hasEDAData() : 
                   item.href === '/extracted-requirements' ? hasRequirementsData() : 
                   false;

    return (
      <motion.div
        key={item.href}
        className={`relative mb-2 ${!item.isActive() ? 'opacity-50 cursor-not-allowed' : ''}`}
        whileHover={item.isActive() ? { scale: 1.02 } : {}}
        whileTap={item.isActive() ? { scale: 0.98 } : {}}
      >
        <button
          onClick={item.isActive() ? item.onClick : undefined}
          className={`w-full flex items-center p-4 text-sm font-medium rounded-lg transition-colors relative
            ${isCurrentPath ? 'bg-gray-700 bg-opacity-50' : ''}
            ${item.isActive() ? 'hover:bg-gray-700' : ''}`}
          disabled={!item.isActive()}
        >
          <item.icon size={20} style={{ color: item.color, minWidth: "20px" }} />
          <AnimatePresence>
            {isSideBarOpen && (
              <motion.span
                className='ml-4 whitespace-nowrap'
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
              >
                {item.name}
              </motion.span>
            )}
          </AnimatePresence>
          {/* Show data indicator if data exists */}
          {isSideBarOpen && <NavItemIndicator hasData={hasData} />}
        </button>
      </motion.div>
    );
  };

  return (
    <motion.div
      className="relative z-10 transition-all duration-300 ease-in-out flex-shrink-0"
      animate={{ width: isSideBarOpen ? 256 : 80 }}
    >
      <div className="h-full bg-gray-400 bg-opacity-50 backdrop-blur-md p-4 flex flex-col border-r border-gray-700">
        {/* Toggle Button */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setIsSideBarOpen(!isSideBarOpen)}
          className='p-2 rounded-full hover:bg-gray-700 transition-colors max-w-fit'
        >
          <Menu size={24} className="text-white" />
        </motion.button>

        {/* Navigation Items */}
        <nav className='mt-8 flex-grow'>
          {navbarItems.map(item => renderNavItem(item))}
        </nav>

        {/* Optional: Add a data status indicator at the bottom */}
        {isSideBarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-xs text-gray-400 mt-4 pb-2"
          >
            {hasFile() && (
              <div className="flex flex-col gap-1 text-black">
                <span>• File uploaded</span>
                {hasEDAData() && <span>• EDA data available</span>}
                {hasRequirementsData() && <span>• Requirements data available</span>}
              </div>
            )}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default NavBar;