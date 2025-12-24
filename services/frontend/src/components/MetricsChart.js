import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Paper, Typography, Box } from '@mui/material';

const MetricsChart = ({ cmcData, title = 'CMC Curve' }) => {
  if (!cmcData || cmcData.length === 0) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Typography color="textSecondary">No data available</Typography>
      </Paper>
    );
  }

  // Transform CMC data for recharts
  // cmcData is an array of accuracy values, index is the rank (0-indexed)
  const chartData = cmcData.map((accuracy, index) => ({
    rank: index + 1,
    accuracy: (accuracy * 100).toFixed(2),
  }));

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <Box sx={{ width: '100%', height: 400, mt: 2 }}>
        <ResponsiveContainer>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="rank"
              label={{ value: 'Rank', position: 'insideBottom', offset: -5 }}
              stroke="rgba(255,255,255,0.7)"
            />
            <YAxis
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.7)"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(30, 30, 60, 0.95)',
                border: '1px solid rgba(99, 102, 241, 0.5)',
                borderRadius: '8px',
              }}
              formatter={(value) => [`${value}%`, 'Accuracy']}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#6366f1"
              strokeWidth={3}
              dot={{ r: 4, fill: '#6366f1' }}
              activeDot={{ r: 6, fill: '#8b5cf6' }}
              name="CMC Accuracy"
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};

export default MetricsChart;
