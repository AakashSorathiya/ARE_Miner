import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const LineChartComponent = ({ data, title='' }) => {

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
              <p>No app data available</p>
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
            transition={{ delay: 0.2 }}
        >
            <h2 className='text-lg font-medium mb-4 text-gray-900'>{title}</h2>

            <div className='h-80'>
                <ResponsiveContainer width={"100%"} height={"100%"}>
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray='3 3' stroke='#4B5563' />
                        <XAxis 
                            dataKey={"name"}
                            stroke='#9ca3af'
                            // Optional: format the date to be more readable
                            tickFormatter={(value) => {
                                const [year, month] = value.split('-');
                                return `${month}/${year.slice(2)}`;
                            }}
                        />
                        <YAxis 
                            stroke='#9ca3af'
                            // Optional: ensure whole numbers only
                            allowDecimals={false}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "rgba(31, 41, 55, 0.95)",  // Darker, more opaque background
                                borderRadius: "8px",                        // Rounded corners
                                border: "1px solid #4B5563",               // Subtle border
                                padding: "8px 12px",                       // Consistent padding
                            }}
                            itemStyle={{ 
                                color: "#ffffff",                          // White text for items
                                fontSize: "14px",
                                padding: "4px 0"
                            }}
                            labelStyle={{
                                color: "#ffffff",                          // White text for label
                                fontWeight: "500",
                                marginBottom: "4px"
                            }}
                            labelFormatter={(label) => {
                                const [year, month] = label.split('-');
                                return `${month}/${year}`;
                            }}
                            formatter={(value) => [`${value} reviews`, "Count"]}
                            cursor={{ stroke: "rgba(255, 255, 255, 0.2)" }}  // Subtle vertical line
                        />
                        <Line
                            type='monotone'
                            dataKey='value' 
                            stroke='#6366F1'
                            strokeWidth={3}
                            dot={{ fill: "#6366F1", strokeWidth: 2, r: 6 }}
                            activeDot={{ r: 8, strokeWidth: 2 }}
                            name="Reviews"  // This will show in the tooltip
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </motion.div>
    );
}

export default LineChartComponent;