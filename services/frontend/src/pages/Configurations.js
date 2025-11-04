import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  Chip,
  Divider,
  Alert,
  Snackbar,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  FileCopy as FileCopyIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { apiService } from '../services/api';

// Default configuration template
const DEFAULT_CONFIG = {
  preset: 'development',
  device: 'cuda',
  yolo_model: 'yolo11n.pt',
  reid_model: 'reid_model.pth',
  detection_conf: 0.25,
  reid_threshold_match: 0.70,
  reid_threshold_new: 0.50,
  gallery_max_size: 300,
  reid_batch_size: 16,
  use_tensorrt: false,
  enable_display: true,
};

const PRESETS = ['development', 'xavier_nx', 'orin_nx', 'agx_orin'];

function Configurations() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [configForm, setConfigForm] = useState({ name: '', config: DEFAULT_CONFIG });
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [configToDelete, setConfigToDelete] = useState(null);

  useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      const response = await apiService.listConfigs();
      setConfigs(response.data.configs);
      setLoading(false);
    } catch (error) {
      console.error('Error loading configs:', error);
      showSnackbar('Failed to load configurations', 'error');
      setLoading(false);
    }
  };

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const handleOpenDialog = (config = null) => {
    if (config) {
      // Edit existing config
      setEditingConfig(config);
      setConfigForm({
        name: config.name,
        config: typeof config.config === 'string' ? JSON.parse(config.config) : config.config,
      });
    } else {
      // Create new config
      setEditingConfig(null);
      setConfigForm({ name: '', config: DEFAULT_CONFIG });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingConfig(null);
    setConfigForm({ name: '', config: DEFAULT_CONFIG });
  };

  const handleSaveConfig = async () => {
    if (!configForm.name.trim()) {
      showSnackbar('Please enter a configuration name', 'error');
      return;
    }

    try {
      if (editingConfig) {
        // Update existing config
        await apiService.updateConfig(editingConfig.config_id, configForm.name, configForm.config);
        showSnackbar('Configuration updated successfully');
      } else {
        // Save new config
        await apiService.saveConfig(configForm.name, configForm.config);
        showSnackbar('Configuration saved successfully');
      }
      loadConfigs();
      handleCloseDialog();
    } catch (error) {
      console.error('Error saving config:', error);
      showSnackbar('Failed to save configuration', 'error');
    }
  };

  const handleDeleteConfig = async () => {
    if (!configToDelete) return;

    try {
      await apiService.deleteConfig(configToDelete.config_id);
      showSnackbar('Configuration deleted successfully');
      loadConfigs();
      setDeleteDialogOpen(false);
      setConfigToDelete(null);
    } catch (error) {
      console.error('Error deleting config:', error);
      showSnackbar('Failed to delete configuration', 'error');
    }
  };

  const handleCloneConfig = (config) => {
    const parsedConfig = typeof config.config === 'string' ? JSON.parse(config.config) : config.config;
    setEditingConfig(null);
    setConfigForm({
      name: `${config.name} (Copy)`,
      config: parsedConfig,
    });
    setOpenDialog(true);
  };

  const handleExportConfig = (config) => {
    const parsedConfig = typeof config.config === 'string' ? JSON.parse(config.config) : config.config;
    const exportData = {
      name: config.name,
      config: parsedConfig,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${config.name.replace(/\s+/g, '_')}_config.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showSnackbar('Configuration exported successfully');
  };

  const handleImportConfig = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      try {
        const text = await file.text();
        const data = JSON.parse(text);
        setEditingConfig(null);
        setConfigForm({
          name: data.name || 'Imported Config',
          config: data.config || data,
        });
        setOpenDialog(true);
        showSnackbar('Configuration imported successfully');
      } catch (error) {
        console.error('Error importing config:', error);
        showSnackbar('Failed to import configuration', 'error');
      }
    };
    input.click();
  };

  const handleConfigFieldChange = (field, value) => {
    setConfigForm({
      ...configForm,
      config: {
        ...configForm.config,
        [field]: value,
      },
    });
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Pipeline Configurations
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Manage saved configurations for single and multi-camera pipelines
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            onClick={handleImportConfig}
          >
            Import
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => handleOpenDialog()}
          >
            New Configuration
          </Button>
        </Box>
      </Box>

      {/* Configuration Cards */}
      {loading ? (
        <Typography>Loading configurations...</Typography>
      ) : configs.length === 0 ? (
        <Paper sx={{ p: 8, textAlign: 'center' }}>
          <SettingsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No Configurations Saved
          </Typography>
          <Typography color="textSecondary" paragraph>
            Create your first configuration to get started
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => handleOpenDialog()}
          >
            Create Configuration
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {configs.map((config) => {
            const parsedConfig = typeof config.config === 'string' ? JSON.parse(config.config) : config.config;
            return (
              <Grid item xs={12} md={6} lg={4} key={config.config_id}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {config.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Last updated: {new Date(config.updated_at).toLocaleString()}
                    </Typography>
                    <Divider sx={{ my: 2 }} />
                    <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                      <Chip label={parsedConfig.preset || 'N/A'} size="small" color="primary" />
                      <Chip label={parsedConfig.device || 'N/A'} size="small" />
                      {parsedConfig.use_tensorrt && <Chip label="TensorRT" size="small" color="success" />}
                    </Box>
                    <Box sx={{ fontSize: '0.875rem', color: 'text.secondary' }}>
                      <Box display="flex" justifyContent="space-between" mb={0.5}>
                        <span>Detection Conf:</span>
                        <strong>{parsedConfig.detection_conf || 'N/A'}</strong>
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={0.5}>
                        <span>Match Threshold:</span>
                        <strong>{parsedConfig.reid_threshold_match || 'N/A'}</strong>
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={0.5}>
                        <span>Gallery Size:</span>
                        <strong>{parsedConfig.gallery_max_size || 'N/A'}</strong>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <span>Batch Size:</span>
                        <strong>{parsedConfig.reid_batch_size || 'N/A'}</strong>
                      </Box>
                    </Box>
                  </CardContent>
                  <CardActions sx={{ p: 2, pt: 0 }}>
                    <IconButton size="small" onClick={() => handleOpenDialog(config)} title="Edit">
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton size="small" onClick={() => handleCloneConfig(config)} title="Clone">
                      <FileCopyIcon fontSize="small" />
                    </IconButton>
                    <IconButton size="small" onClick={() => handleExportConfig(config)} title="Export">
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                    <Box flexGrow={1} />
                    <IconButton
                      size="small"
                      onClick={() => {
                        setConfigToDelete(config);
                        setDeleteDialogOpen(true);
                      }}
                      title="Delete"
                      color="error"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      )}

      {/* Edit/Create Configuration Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingConfig ? 'Edit Configuration' : 'New Configuration'}
        </DialogTitle>
        <DialogContent dividers>
          <TextField
            fullWidth
            label="Configuration Name"
            value={configForm.name}
            onChange={(e) => setConfigForm({ ...configForm, name: e.target.value })}
            margin="normal"
            required
          />

          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" fontWeight="bold">Basic Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Preset</InputLabel>
                    <Select
                      value={configForm.config.preset}
                      onChange={(e) => handleConfigFieldChange('preset', e.target.value)}
                      label="Preset"
                    >
                      {PRESETS.map((preset) => (
                        <MenuItem key={preset} value={preset}>{preset}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Device</InputLabel>
                    <Select
                      value={configForm.config.device}
                      onChange={(e) => handleConfigFieldChange('device', e.target.value)}
                      label="Device"
                    >
                      <MenuItem value="cuda">CUDA</MenuItem>
                      <MenuItem value="cpu">CPU</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="YOLO Model"
                    value={configForm.config.yolo_model}
                    onChange={(e) => handleConfigFieldChange('yolo_model', e.target.value)}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="ReID Model"
                    value={configForm.config.reid_model}
                    onChange={(e) => handleConfigFieldChange('reid_model', e.target.value)}
                    margin="normal"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" fontWeight="bold">Detection & ReID Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Detection Confidence"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={configForm.config.detection_conf}
                    onChange={(e) => handleConfigFieldChange('detection_conf', parseFloat(e.target.value))}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="ReID Batch Size"
                    type="number"
                    inputProps={{ min: 1, max: 64 }}
                    value={configForm.config.reid_batch_size}
                    onChange={(e) => handleConfigFieldChange('reid_batch_size', parseInt(e.target.value))}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Match Threshold"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={configForm.config.reid_threshold_match}
                    onChange={(e) => handleConfigFieldChange('reid_threshold_match', parseFloat(e.target.value))}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="New Person Threshold"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={configForm.config.reid_threshold_new}
                    onChange={(e) => handleConfigFieldChange('reid_threshold_new', parseFloat(e.target.value))}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Gallery Max Size"
                    type="number"
                    inputProps={{ min: 50, max: 2000 }}
                    value={configForm.config.gallery_max_size}
                    onChange={(e) => handleConfigFieldChange('gallery_max_size', parseInt(e.target.value))}
                    margin="normal"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" fontWeight="bold">Advanced Options</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <FormControlLabel
                control={
                  <Switch
                    checked={configForm.config.use_tensorrt}
                    onChange={(e) => handleConfigFieldChange('use_tensorrt', e.target.checked)}
                  />
                }
                label="Use TensorRT"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={configForm.config.enable_display}
                    onChange={(e) => handleConfigFieldChange('enable_display', e.target.checked)}
                  />
                }
                label="Enable Display"
              />
            </AccordionDetails>
          </Accordion>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveConfig}>
            {editingConfig ? 'Update' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Configuration</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the configuration "{configToDelete?.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleDeleteConfig}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default Configurations;
