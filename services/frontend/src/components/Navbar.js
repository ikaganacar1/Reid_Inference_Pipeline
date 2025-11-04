import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import VideocamIcon from '@mui/icons-material/Videocam';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ListIcon from '@mui/icons-material/List';
import SettingsIcon from '@mui/icons-material/Settings';

function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <VideocamIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          ReID Pipeline
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            startIcon={<DashboardIcon />}
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/single-camera"
          >
            Single Camera
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/multi-camera"
          >
            Multi Camera
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/jobs"
            startIcon={<ListIcon />}
          >
            Jobs
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/configs"
            startIcon={<SettingsIcon />}
          >
            Configs
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
