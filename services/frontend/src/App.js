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
import ResultsViewer from './pages/ResultsViewer';
import Configurations from './pages/Configurations';

// Premium theme with glassmorphism and gradients
const premiumTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6366f1', // Indigo
      light: '#818cf8',
      dark: '#4f46e5',
    },
    secondary: {
      main: '#ec4899', // Pink
      light: '#f472b6',
      dark: '#db2777',
    },
    success: {
      main: '#10b981', // Emerald
      light: '#34d399',
      dark: '#059669',
    },
    warning: {
      main: '#f59e0b', // Amber
      light: '#fbbf24',
      dark: '#d97706',
    },
    error: {
      main: '#ef4444', // Red
      light: '#f87171',
      dark: '#dc2626',
    },
    background: {
      default: '#0f0f23', // Deep dark blue
      paper: 'rgba(30, 30, 60, 0.7)', // Glassmorphic paper
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5e1',
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
    button: {
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%)',
          backgroundAttachment: 'fixed',
          '&::before': {
            content: '""',
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
            zIndex: 0,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(30, 30, 60, 0.5)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(30, 30, 60, 0.5)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 48px 0 rgba(99, 102, 241, 0.3)',
            border: '1px solid rgba(99, 102, 241, 0.5)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: '10px 24px',
          boxShadow: 'none',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        contained: {
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 24px rgba(99, 102, 241, 0.4)',
          },
        },
        outlined: {
          borderColor: 'rgba(99, 102, 241, 0.5)',
          '&:hover': {
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
          backdropFilter: 'blur(10px)',
        },
        filled: {
          backgroundColor: 'rgba(99, 102, 241, 0.2)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
        },
        bar: {
          borderRadius: 10,
          background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(15, 15, 35, 0.8)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 24px 0 rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(30, 30, 60, 0.95)',
          backdropFilter: 'blur(30px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={premiumTheme}>
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
              <Route path="/results/:jobId" element={<ResultsViewer />} />
              <Route path="/configs" element={<Configurations />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
