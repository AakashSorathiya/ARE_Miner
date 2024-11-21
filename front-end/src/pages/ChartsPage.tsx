import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import Spinner from "../components/Spinner";
import StatCard from "../components/StatCard";
import { Zap } from "lucide-react";
import BarChartComponent from "../components/BarChartComponent";
import LineChartComponent from "../components/LineChartComponent";
import PieChartComponent from "../components/PieChartComponent";

const ChartsPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const [isLoading, setIsLoading] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        // First try to get data from navigation state
        if (location.state?.data) {
            localStorage.setItem('edaData', JSON.stringify(location.state.data));
            setData(location.state.data);
            return;
        }

        // If no navigation state, try to get from localStorage
        const savedData = localStorage.getItem('edaData');
        if (savedData) {
            try {
                setData(JSON.parse(savedData));
            } catch (err) {
                console.error('Error parsing saved data:', err);
                setError('Error loading saved data');
                localStorage.removeItem('edaData'); // Clear invalid data
            }
        }
    }, [location.state]);

    const dataToBlob = async (dataUrl) => {
        try {
            const response = await fetch(dataUrl);
            const blob = await response.blob();
            return new File([blob], 'upload.csv', { type: 'text/csv' });
        } catch (error) {
            console.error('Error converting data to blob:', error);
            throw new Error('Failed to process file');
        }
    };

    const getEDA = async () => {
        const currentFile = localStorage.getItem('currentFile');
        if (!currentFile) {
            toast.error('No file available. Please upload a file first.');
            navigate('/');
            return;
        }

        setIsLoading(true);
        setError(null);
        const formData = new FormData();

        try {
            const file = await dataToBlob(currentFile);
            formData.append('csv_file', file);

            const response = await fetch('/api/eda', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to get EDA');
            }

            const edaData = await response.json();
            localStorage.setItem('edaData', JSON.stringify(edaData));
            setData(edaData);
            toast.success('EDA analysis completed successfully');
        } catch (error) {
            console.error('EDA error:', error);
            setError(error.message);
            toast.error(`Failed to get EDA: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRefreshData = async () => {
        await getEDA();
    };

    const handleResetData = () => {
        localStorage.removeItem('edaData');
        setData(null);
        toast.info('EDA data has been cleared');
    };

    if (isLoading) {
        return (
            <div className="flex-1 flex items-center justify-center">
                <Spinner loading={true} />
            </div>
        );
    }

    if (error) {
        return (
            <motion.div 
                className="flex-1 flex flex-col items-center justify-center p-6 z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
            >
                <div className="bg-white rounded-lg shadow-md p-6 max-w-md w-full text-center">
                    <h2 className="text-xl font-semibold mb-4 text-red-600">Error Loading Data</h2>
                    <p className="text-gray-600 mb-6">{error}</p>
                    <div className="flex flex-col gap-4">
                        <button
                            onClick={handleRefreshData}
                            className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                        >
                            Retry Analysis
                        </button>
                        <button
                            onClick={() => navigate('/')}
                            className="w-full px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                        >
                            Return to Upload
                        </button>
                    </div>
                </div>
            </motion.div>
        );
    }

    if (!data) {
        return (
            <motion.div 
                className="flex-1 flex flex-col items-center justify-center p-6 z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
            >
                <div className="bg-white rounded-lg shadow-md p-6 max-w-md w-full text-center">
                    <h2 className="text-xl font-semibold mb-4">No Data Available</h2>
                    <p className="text-gray-600 mb-6">
                        Would you like to perform exploratory data analysis?
                    </p>
                    <div className="flex flex-col gap-4">
                        <button
                            onClick={getEDA}
                            className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                        >
                            Get EDA Data
                        </button>
                        <button
                            onClick={() => navigate('/')}
                            className="w-full px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                        >
                            Upload New File
                        </button>
                    </div>
                </div>
            </motion.div>
        );
    }

    // Transform data for charts
    const timeData = Object.entries(data.time_distribution || {}).map(([date, count]) => ({
        name: date,
        value: count
    }));

    const avg_word_count = Math.round(data.avg_word_count || 0);
    const sentimentData = Object.entries(data.sentiment_distribution || {}).map(([sentiment, count]) => ({
        name: sentiment,
        value: count
    }));

    const appData = Object.entries(data.app_distribution || {}).map(([app, count]) => ({
        name: app,
        value: count
    }));

    return (
        <div className='flex-1 overflow-auto relative z-10'>
            <main className='max-w-7xl mx-auto py-6 px-4 lg:px-8'>
                {/* Header with actions */}
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-2xl font-semibold text-gray-800">Exploratory Data Analysis</h1>
                    <div className="flex gap-3">
                        <button
                            onClick={handleRefreshData}
                            className="px-4 py-2 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                        >
                            Refresh Data
                        </button>
                        <button
                            onClick={handleResetData}
                            className="px-4 py-2 text-sm text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                        >
                            Clear Data
                        </button>
                    </div>
                </div>

                {/* Charts Grid */}
                <motion.div 
                    className='grid grid-cols-1 lg:grid-cols-2 gap-8'
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    
                        <PieChartComponent 
                            data={sentimentData} 
                            title='Sentiment Distribution' 
                            isSentiment={true}
                        />
               

                   
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <StatCard 
                                name='Average Number of Words' 
                                icon={Zap} 
                                value={avg_word_count} 
                                color='#6366F1' 
                            />
                        </motion.div>
                 

                    
                        <LineChartComponent 
                            data={timeData} 
                            title='Distribution Over Time' 
                        />
          
                        <BarChartComponent 
                            data={appData} 
                            title='App Distribution'
                        />
    
                </motion.div>
            </main>
        </div>
    );
};

export default ChartsPage;