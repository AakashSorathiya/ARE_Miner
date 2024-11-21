import { motion } from "framer-motion";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

// Color palette for general data visualization
const COLORS = [
  "#6366F1", // Indigo
  "#8B5CF6", // Purple
  "#EC4899", // Pink
  "#10B981", // Emerald
  "#F59E0B", // Amber
  "#EF4444", // Red
  "#3B82F6", // Blue
  "#14B8A6", // Teal
  "#F97316", // Orange
  "#8B5CF6"  // Purple (repeated for 10th item)
];

// Specific colors for sentiment analysis
const SENTIMENT_COLORS = {
  positive: "#4ade80",  // green
  negative: "#f87171",  // red
  neutral: "#94a3b8"    // gray
};

const PieChartComponent = ({ data, title = '', isSentiment = false }) => {
  // If no data is provided, show empty state
  if (!data || data.length === 0) {
    return (
      <motion.div
        className='bg-gray-200 bg-opacity-50 backdrop-blur-md shadow-lg rounded-xl p-6 border border-gray-700'
        whileHover={{ y: -5, boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)" }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className='text-xl font-semibold text-gray-900 mb-4'>{title}</h2>
        <div className="flex items-center justify-center h-[300px]">
          <p>No data available</p>
        </div>
      </motion.div>
    );
  }

  // Calculate total for percentage
  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <motion.div
      className='bg-gray-100 bg-opacity-50 backdrop-blur-md shadow-lg rounded-xl p-6 border border-gray-700'
      whileHover={{ y: -5, boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <h2 className='text-xl font-semibold text-gray-900 mb-4'>{title}</h2>

      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              cx='50%'
              cy='50%'
              outerRadius={80}
              fill='#8884d8'
              dataKey='value'
              label={({ name, value }) => 
                `${name} (${((value / total) * 100).toFixed(1)}%)`
              }
              labelLine={true}
            >
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={isSentiment ? 
                    SENTIMENT_COLORS[entry.name.toLowerCase()] : 
                    COLORS[index % COLORS.length]
                  } 
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(31, 41, 55, 0.8)",
                borderColor: "#4B5563",
              }}
              itemStyle={{ color: "#E5E7EB" }}
              formatter={(value, name) => [`Count: ${value}`, name]}
            />
            <Legend 
              formatter={(value) => value}
              wrapperStyle={{ fontSize: "12px" }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};

export default PieChartComponent;