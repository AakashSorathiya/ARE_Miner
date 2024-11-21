import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Search, Eye, Download, ChevronLeft, ChevronRight } from "lucide-react";

const MotionDiv = motion.div;
const MotionTr = motion.tr;

const ReqTable = ({ data = {}, title = "Extracted Requirements" }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [filteredRecords, setFilteredRecords] = useState([]);
  const recordsPerPage = 10;

  // Flatten records once when component mounts or data changes
  const flattenedRecords = Object.entries(data).map(([key, record]) => ({
    id: key,
    app: record.App,
    review: record.Review,
    date: record.Date,
    requirements: record.requirements.length === 0 ? [] : record.requirements,
    total_requirements: record.requirements.length
  }));

  // Update filtered records whenever search term changes
  useEffect(() => {
    const filtered = flattenedRecords.filter((record) => 
      record.app.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.review.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.date.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.requirements.some(req => 
        req.requirement.toLowerCase().includes(searchTerm.toLowerCase()) ||
        req.sentiment.toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
    setFilteredRecords(filtered);
    setCurrentPage(1); // Reset to first page when search term changes
  }, [searchTerm, data]);

  // Pagination logic
  const totalPages = Math.ceil(filteredRecords.length / recordsPerPage);
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  const currentRecords = filteredRecords.slice(indexOfFirstRecord, indexOfLastRecord);

  // Navigation functions
  const nextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  const prevPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const goToPage = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  // Generate page numbers
  const getPageNumbers = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= maxVisiblePages; i++) {
          pages.push(i);
        }
      } else if (currentPage >= totalPages - 2) {
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        for (let i = currentPage - 2; i <= currentPage + 2; i++) {
          pages.push(i);
        }
      }
    }
    
    return pages;
  };

  const downloadCSV = () => {
    // Use all records if no filter is applied, otherwise use filtered records
    const recordsToExport = searchTerm.trim() === '' ? flattenedRecords : filteredRecords;
      
    // Create CSV from the appropriate records
    const csvData = recordsToExport.map(record => ({
      App: record.app,
      Review: record.review,
      Date: record.date,
      Total_Requirements: record.total_requirements,
      Requirements: record.requirements.map(req => req.requirement).join('; '),
      Sentiments: record.requirements.map(req => req.sentiment).join('; ')
    }));
  
    // Define CSV headers
    const headers = ["App", "Review", "Date", "Total_Requirements", "Requirements", "Sentiments"];
    
    // Convert to CSV rows
    const csvRows = [
      headers.join(","),
      ...csvData.map(record => [
        `"${record.App}"`,
        `"${record.Review.replace(/"/g, '""')}"`,
        `"${record.Date}"`,
        `"${record.Total_Requirements}"`,
        `"${record.Requirements}"`,
        `"${record.Sentiments}"`
      ].join(","))
    ];
  
    // Create and download CSV file
    const csvContent = csvRows.join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    
    // Add timestamp to filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = searchTerm.trim() === '' 
      ? `all_requirements_${timestamp}.csv`
      : `filtered_requirements_${timestamp}.csv`;
      
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url); // Clean up to avoid memory leaks
  };

  return (
    <motion.div
      className="bg-gray-100 bg-opacity-50 backdrop-blur-md shadow-lg rounded-xl p-6 lg:col-span-2 border border-gray-700"
      whileHover={{ y: -5, boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-700">{title}</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <input
              type="text"
              placeholder="Search requirements..."
              className="bg-gray-700 text-white placeholder-gray-200 rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Search className="absolute left-3 top-2.5 text-gray-700" size={18} />
          </div>
          <button
            onClick={downloadCSV}
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
          >
            <Download size={18} />
            <span>Download CSV</span>
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-700">
          <thead>
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                App
              </th>
			  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Review
			</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Total Requirements
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Requirement
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Sentiment
              </th>
              {/* <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Actions
              </th> */}
            </tr>
          </thead>

		  <tbody className="divide divide-gray-700">
			{currentRecords.map((record) => (
				<MotionTr
				key={record.id}
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ duration: 0.3 }}
				>
				<td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500">
					{record.app}
				</td>
				<td className="px-6 py-4 whitespace-normal text-sm font-medium text-gray-500">
					{record.review}
				</td>
				<td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500">
					{record.date}
				</td>
				<td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500">
					{record.total_requirements}
				</td>
				<td className="px-6 py-4 whitespace-normal text-sm font-medium text-gray-500">
					<ul className="list-disc pl-4 space-y-1">
					{record.requirements.map((req, index) => (
						<li key={index}>{req.requirement}</li>
					))}
					</ul>
				</td>
				<td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
					<ul className="space-y-1"> {/* Changed to ul to match requirements structure */}
						{record.requirements.map((req, index) => (
						<li key={index}>
							<span
							className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
								req.sentiment === "positive"
								? "bg-green-100 text-green-800"
								: req.sentiment === "negative"
								? "bg-red-100 text-red-800"
								: "bg-yellow-100 text-yellow-800"
							}`}
							>
							{req.sentiment}
							</span>
						</li>
						))}
					</ul>
					</td>
				</MotionTr>
			))}
			</tbody>

        </table>

        {/* Pagination Controls */}
        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {indexOfFirstRecord + 1} to {Math.min(indexOfLastRecord, filteredRecords.length)} of{" "}
            {filteredRecords.length} results
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={prevPage}
              disabled={currentPage === 1}
              className="px-3 py-1 rounded-md bg-gray-200 text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={20} />
            </button>

            {getPageNumbers().map((pageNum) => (
              <button
                key={pageNum}
                onClick={() => goToPage(pageNum)}
                className={`px-3 py-1 rounded-md ${
                  currentPage === pageNum
                    ? "bg-blue-500 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                {pageNum}
              </button>
            ))}

            <button
              onClick={nextPage}
              disabled={currentPage === totalPages}
              className="px-3 py-1 rounded-md bg-gray-200 text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight size={20} />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ReqTable;