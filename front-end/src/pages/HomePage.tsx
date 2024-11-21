import { Upload } from "lucide-react";
import { useState, useEffect } from "react";
import { toast } from 'react-toastify';
import Spinner from "../components/Spinner";
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isGettingEda, setIsGettingEda] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);
  const navigate = useNavigate();

  // Check for existing file on component mount
  useEffect(() => {
    const existingFile = localStorage.getItem('currentFile');
    if (existingFile) {
      setFileUploaded(true);
    }
  }, []);

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

  const handleFileUpload = (file) => {
    if (file && file.name.endsWith('.csv')) {
      // Clear existing data when uploading new file
      localStorage.removeItem('edaData');
      localStorage.removeItem('requirementsData');
      
      const reader = new FileReader();
      reader.onload = (e) => {
        localStorage.setItem('currentFile', e.target.result);
        setFileUploaded(true);
        toast.success('File uploaded successfully!');
      };
      reader.onerror = () => {
        toast.error('Error reading file');
      };
      reader.readAsDataURL(file);
    } else {
      toast.error('Please upload a CSV file');
    }
  };

  const getEDA = async () => {
    const currentFile = localStorage.getItem('currentFile');
    if (!currentFile) {
      toast.error('No file available for EDA');
      return;
    }

    setIsGettingEda(true);
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

      const data = await response.json();
      localStorage.setItem('edaData', JSON.stringify(data));
      navigate('/charts', { state: { data } });
    } catch (error) {
      console.error('EDA error:', error);
      toast.error(`Failed to get EDA: ${error.message}`);
    } finally {
      setIsGettingEda(false);
    }
  };

  const extractRequirements = async () => {
    const currentFile = localStorage.getItem('currentFile');
    if (!currentFile) {
      toast.error('No file available for extraction');
      return;
    }

    setIsExtracting(true);
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
      navigate('/extracted-requirements', { state: { data: requirementsData } });
    } catch (error) {
      console.error('Extract requirements error:', error);
      toast.error(`Failed to extract requirements: ${error.message}`);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const handleInputChange = (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
  };

  const handleResetFile = () => {
    setFileUploaded(false);
    localStorage.removeItem('currentFile');
    localStorage.removeItem('edaData');
    localStorage.removeItem('requirementsData');
  };

  // Check if existing data is available
  const hasExistingEDA = () => {
    return localStorage.getItem('edaData') !== null;
  };

  const hasExistingRequirements = () => {
    return localStorage.getItem('requirementsData') !== null;
  };

  return (
    <div className="flex-1 overflow-auto relative z-10">
      <div className="w-full max-w-2xl mx-auto p-4 mt-11">
        {!fileUploaded ? (
          // File Upload Section
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-4 border-b">
              <h2 className="text-xl font-semibold">Upload Your CSV file</h2>
            </div>
  
            <div className="p-6">
              <div
                className={`border-2 border-dashed rounded-lg p-6 transition-colors
                  ${isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300"}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="csvUpload"
                  className="hidden"
                  onChange={handleInputChange}
                  accept=".csv"
                />
                <label
                  htmlFor="csvUpload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  <Upload 
                    className={`h-12 w-12 mb-2 transition-colors
                      ${isDragging ? "text-blue-500" : "text-gray-400"}`} 
                  />
                  <span className="text-sm text-gray-600 text-center">
                    Drag and drop your CSV file here or click to upload
                  </span>
                </label>
              </div>
            </div>
          </div>
        ) : (
          // Analysis Options Section
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-4 border-b">
              <h2 className="text-xl font-semibold">Choose Analysis Type</h2>
              {(hasExistingEDA() || hasExistingRequirements()) && (
                <p className="text-sm text-gray-500 mt-1">
                  Previous analysis data is available
                </p>
              )}
            </div>
  
            {isGettingEda || isExtracting ? (
              <div className="flex justify-center items-center p-6">
                <Spinner loading={true} />
              </div>
            ) : (
              <div className="p-6">
                <div className="flex flex-col space-y-4">
                  {/* EDA Button with indicator */}
                  <button
                    onClick={getEDA}
                    disabled={isGettingEda}
                    className="w-full flex justify-center items-center px-4 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed relative"
                  >
                    {hasExistingEDA() && (
                      <span className="absolute right-2 top-2 w-2 h-2 bg-green-400 rounded-full"></span>
                    )}
                    Get Exploratory Data Analysis
                  </button>
  
                  {/* Requirements Button with indicator */}
                  <button
                    onClick={extractRequirements}
                    disabled={isExtracting}
                    className="w-full flex justify-center items-center px-4 py-3 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed relative"
                  >
                    {hasExistingRequirements() && (
                      <span className="absolute right-2 top-2 w-2 h-2 bg-blue-400 rounded-full"></span>
                    )}
                    Extract Requirements
                  </button>
                </div>
                
                <button
                  onClick={handleResetFile}
                  className="w-full mt-4 px-4 py-2 text-gray-600 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                >
                  Upload Different File
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;