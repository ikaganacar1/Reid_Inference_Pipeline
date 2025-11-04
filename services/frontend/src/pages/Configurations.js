import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  Chip,
} from '@mui/material';
import { apiService } from '../services/api';

function Configurations() {
  const [configs, setConfigs] = useState([]);

  useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      const response = await apiService.listConfigs();
      setConfigs(response.data.configs);
    } catch (error) {
      console.error('Error loading configs:', error);
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Saved Configurations
      </Typography>

      <Paper sx={{ p: 3 }}>
        {configs.length === 0 ? (
          <Typography color="textSecondary">No saved configurations</Typography>
        ) : (
          <List>
            {configs.map((config) => (
              <ListItem key={config.config_id} divider>
                <ListItemText
                  primary={config.name}
                  secondary={`Last updated: ${new Date(config.updated_at).toLocaleString()}`}
                />
                <Chip label="Preset" color="primary" size="small" />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>
    </Container>
  );
}

export default Configurations;
