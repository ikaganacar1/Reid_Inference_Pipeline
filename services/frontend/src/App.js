import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';

import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import SingleCameraPipeline from './pages/SingleCameraPipeline';
import MultiCameraPipeline from './pages/MultiCameraPipeline';
import JobsList from './pages/JobsList';
import JobDetail from './pages/JobDetail';
import Configurations from './pages/Configurations';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Navbar />
          <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/single-camera" element={<SingleCameraPipeline />} />
              <Route path="/multi-camera" element={<MultiCameraPipeline />} />
              <Route path="/jobs" element={<JobsList />} />
              <Route path="/jobs/:jobId" element={<JobDetail />} />
              <Route path="/configs" element={<Configurations />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
