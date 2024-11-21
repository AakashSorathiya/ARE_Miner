import { motion } from "framer-motion";
import { useState, useEffect, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import Spinner from "../components/Spinner";
import BarChartComponent from "../components/BarChartComponent";
import LineChartComponent from "../components/LineChartComponent";
import PieChartComponent from "../components/PieChartComponent";
import ReqTable from "../components/ReqTable";

const ExtractedReqPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const [isLoading, setIsLoading] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        // First try to get data from navigation state
        if (location.state?.data) {
            localStorage.setItem('requirementsData', JSON.stringify(location.state.data));
            setData(location.state.data);
            return;
        }

        // If no navigation state, try to get from localStorage
        const savedData = localStorage.getItem('requirementsData');
        if (savedData) {
            try {
                setData(JSON.parse(savedData));
            } catch (err) {
                console.error('Error parsing saved data:', err);
                setError('Error loading saved data');
                localStorage.removeItem('requirementsData'); // Clear invalid data
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

    const extractRequirements = async () => {
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

            const response = await fetch('/api/extract_requirements', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to extract requirements');
            }
            const requirementsData = await response.json();
            localStorage.setItem('requirementsData', JSON.stringify(requirementsData));
            setData(requirementsData);
            toast.success('Requirements extracted successfully');
        } catch (error) {
            console.error('Extract requirements error:', error);
            setError(error.message);
            toast.error(`Failed to extract requirements: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRefreshData = async () => {
        await extractRequirements();
    };

    const handleResetData = () => {
        localStorage.removeItem('requirementsData');
        setData(null);
        toast.info('Requirements data has been cleared');
    };

    // Memoized data transformations
    const {
        distributionOverApps,
        wordCountDistribution,
        sentimentDistribution,
        distributionOverReviews,
        distributionOverTime,
        topRequirements
    } = useMemo(() => {
        if (!data) return {};

        return {
            distributionOverApps: Object.entries(data.distribution_over_apps || {})
                .map(([app, count]) => ({
                    name: app,
                    value: count
                })),

            wordCountDistribution: Object.entries(data.word_count_distribution || {})
                .map(([word_count, num_req]) => ({
                    name: word_count,
                    value: num_req
                })),

            sentimentDistribution: Object.entries(data.sentiment_distribution || {})
                .map(([sentiment, count]) => ({
                    name: sentiment,
                    value: count
                })),

            distributionOverReviews: Object.entries(data.distribution_over_reviews || {})
                .map(([num_req, num_rev]) => ({
                    name: num_req,
                    value: num_rev
                })),

            distributionOverTime: Object.entries(data.distribution_over_time || {})
                .map(([year_month, num_req]) => ({
                    name: year_month,
                    value: num_req
                })),

            topRequirements: Object.entries(data.records || {})
                .flatMap(([_, record]) => record.requirements)
                .reduce((acc, { requirement }) => {
                    const req = requirement.toLowerCase().trim();
                    acc[req] = (acc[req] || 0) + 1;
                    return acc;
                }, {})
        };
    }, [data]);

    // Memoized top requirements data transformation
    const topRequirementsData = useMemo(() => {
        if (!topRequirements) return [];
        
        return Object.entries(topRequirements)
            .sort(([_, countA], [__, countB]) => countB - countA)
            .slice(0, 5)
            .map(([requirement, count]) => ({
                name: requirement,
                value: count
            }));
    }, [topRequirements]);

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
                            className="w-full px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
                        >
                            Retry Extraction
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
                    <h2 className="text-xl font-semibold mb-4">No Requirements Data Available</h2>
                    <p className="text-gray-600 mb-6">
                        Would you like to extract requirements from your file?
                    </p>
                    <div className="flex flex-col gap-4">
                        <button
                            onClick={extractRequirements}
                            className="w-full px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
                        >
                            Extract Requirements
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

    return (
        <div className='flex-1 overflow-auto relative z-10'>
            <main className='max-w-7xl mx-auto py-6 px-4 lg:px-8'>
                {/* Header with actions */}
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-2xl font-semibold text-gray-800">Extracted Requirements Analysis</h1>
                    <div className="flex gap-3">
                        <button
                            onClick={handleRefreshData}
                            className="px-4 py-2 text-sm bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
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

                <motion.div 
                    className='grid grid-cols-1 lg:grid-cols-2 gap-8'
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
             
                <PieChartComponent data={sentimentDistribution} title='Sentiment Distribution' isSentiment={true} />
            
                <PieChartComponent data={topRequirementsData} title='Top 5 Extracted Requirements'/>
                
                <LineChartComponent data={distributionOverTime} title='Distribution over Time' />
              
                <BarChartComponent data={distributionOverReviews} title='Distribution Over Reviews' />
             
                <BarChartComponent data={distributionOverApps} title='Distribution Over Apps' />
             
                <BarChartComponent data={wordCountDistribution} title='Word Count Distribution'  />

                <ReqTable data={data.records} />
               
                </motion.div>
                
                
    
            </main>
        </div>
    );
};

export default ExtractedReqPage;