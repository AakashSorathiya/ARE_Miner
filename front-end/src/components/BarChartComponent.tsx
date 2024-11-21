import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from "recharts";

const COLORS = ["#6366F1", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"];

const BarChartComponent = ({ data, title='', xAxisLabel='X Axis', yAxisLabel='Y Axis'}) => {
  // If no data is provided, show empty state
  if (!data || data.length === 0) {
    return (
      <motion.div
        className='bg-gray-200 bg-opacity-50 backdrop-blur-md shadow-lg rounded-xl p-6 lg:col-span-2 border border-gray-700'
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className='text-lg font-medium mb-4 text-gray-900'>{title}</h2>
        <div className="flex items-center justify-center h-[300px]">
          <p>No app distribution data available</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className='bg-gray-100 bg-opacity-50 backdrop-blur-md shadow-lg rounded-xl p-6 lg:col-span-2 border border-gray-700'
      whileHover={{ y: -5, boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)" }}
      
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <h2 className='text-lg font-medium mb-4 text-gray-900'>{title}</h2>

      <div className='h-80'>
        <ResponsiveContainer>
          <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
          <XAxis 
            dataKey="name" 
            stroke="#9CA3AF"
            angle={-45} 
            textAnchor="end" 
            height={100} 
            tick={{
              fontSize: 13, // Adjust the size of the labels
              fill: "#4B5563" // Optional: Customize color
            }}
          />
          <YAxis 
            stroke="#9CA3AF" 
            // height={500}
            allowDecimals={false} 
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(31, 41, 55, 0.95)", 
              borderRadius: "8px",                        
              border: "1px solid #4B5563",               
              padding: "8px 12px",                      
            }}
            itemStyle={{ 
              color: "#ffffff",                          
              fontSize: "14px",
              padding: "4px 0"
            }}
            labelStyle={{
              color: "#ffffff",                          
              fontWeight: "500",
              marginBottom: "4px"
            }}
            formatter={(value) => [`${value} reviews`, "Count"]}
            cursor={{ fill: "rgba(0, 0, 0, 0.1)" }}     
          />
          <Legend 
            layout="horizontal"
            verticalAlign="bottom" // Moves the legend to the bottom
            align="center"         // Keeps the legend centered
            wrapperStyle={{
              paddingTop: "30px",  // Adds spacing between the chart and the legend
            }}
          />
          <Bar 
            dataKey="value" 
            fill="#8884d8" 
            name="Reviews"
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={COLORS[index % COLORS.length]} 
              />
            ))}
          </Bar>
        </BarChart>

        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

export default BarChartComponent;