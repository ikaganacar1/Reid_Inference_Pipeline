import React from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';
import VideocamIcon from '@mui/icons-material/Videocam';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ListIcon from '@mui/icons-material/List';
import SettingsIcon from '@mui/icons-material/Settings';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import ViewModuleIcon from '@mui/icons-material/ViewModule';

// Styled navigation button with active state
const NavButton = styled(Button)(({ theme, active }) => ({
  color: active ? theme.palette.primary.main : theme.palette.text.secondary,
  borderRadius: 10,
  padding: '8px 16px',
  fontWeight: 600,
  position: 'relative',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  '&::before': {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: '50%',
    transform: 'translateX(-50%)',
    width: active ? '60%' : '0%',
    height: '2px',
    background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
    borderRadius: '2px 2px 0 0',
    transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  },
  '&:hover': {
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    color: theme.palette.primary.light,
    '&::before': {
      width: '60%',
    },
  },
}));

// Logo with gradient
const LogoText = styled(Typography)(({ theme }) => ({
  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  fontWeight: 800,
  fontSize: '1.5rem',
  letterSpacing: '-0.02em',
}));

function Navbar() {
  const location = useLocation();

  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar sx={{ py: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexGrow: 1 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 44,
              height: 44,
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              boxShadow: '0 4px 14px rgba(99, 102, 241, 0.4)',
            }}
          >
            <VideocamIcon sx={{ color: 'white', fontSize: 24 }} />
          </Box>
          <Box>
            <LogoText variant="h6" component="div">
              ReID Pipeline
            </LogoText>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
              Person Re-Identification System
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <NavButton
            component={RouterLink}
            to="/"
            startIcon={<DashboardIcon />}
            active={isActive('/') ? 1 : 0}
          >
            Dashboard
          </NavButton>
          <NavButton
            component={RouterLink}
            to="/single-camera"
            startIcon={<CameraAltIcon />}
            active={isActive('/single-camera') ? 1 : 0}
          >
            Single Camera
          </NavButton>
          <NavButton
            component={RouterLink}
            to="/multi-camera"
            startIcon={<ViewModuleIcon />}
            active={isActive('/multi-camera') ? 1 : 0}
          >
            Multi Camera
          </NavButton>
          <NavButton
            component={RouterLink}
            to="/jobs"
            startIcon={<ListIcon />}
            active={isActive('/jobs') ? 1 : 0}
          >
            Jobs
          </NavButton>
          <NavButton
            component={RouterLink}
            to="/configs"
            startIcon={<SettingsIcon />}
            active={isActive('/configs') ? 1 : 0}
          >
            Configs
          </NavButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
