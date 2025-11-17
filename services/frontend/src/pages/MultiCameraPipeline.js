import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Chip,
  Switch,
  FormControlLabel,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SettingsIcon from '@mui/icons-material/Settings';
import { apiService } from '../services/api';
import { useNavigate } from 'react-router-dom';

function MultiCameraPipeline() {
  const navigate = useNavigate();
  const [videos, setVideos] = useState([null, null, null, null]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [selectedConfigId, setSelectedConfigId] = useState('');
  const [yoloModels, setYoloModels] = useState([]);
  const [reidModels, setReidModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [config, setConfig] = useState({
    device: 'cuda',
    enable_display: false, // Disable by default for Docker/server
    yolo_model: 'yolo11n.pt',
    reid_model: '',
    detection_conf: 0.3,
    detection_use_tensorrt: false,
    reid_threshold_match: 0.50,
    reid_threshold_new: 0.70,
    reid_batch_size: 16,
    reid_use_tensorrt: false,
    tensorrt_precision: 'fp16',
    gallery_max_size: 1000,
    display_scale: 0.5,
  });

  useEffect(() => {
    loadSavedConfigs();
    loadAvailableModels();
  }, []);

  const loadSavedConfigs = async () => {
    try {
      const response = await apiService.listConfigs();
      setSavedConfigs(response.data.configs || []);
    } catch (err) {
      console.error('Error loading configs:', err);
    }
  };

  const loadAvailableModels = async () => {
    setLoadingModels(true);
    try {
      const response = await apiService.listModels();
      setYoloModels(response.data.yolo_models || []);
      setReidModels(response.data.reid_models || []);
    } catch (err) {
      console.error('Error loading models:', err);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleLoadConfig = (event) => {
    const configId = event.target.value;
    setSelectedConfigId(configId);

    if (!configId) {
      // Reset to default
      setConfig({
        device: 'cuda',
        enable_display: false,
        yolo_model: 'yolo11n.pt',
        reid_model: '',
        detection_conf: 0.3,
        detection_use_tensorrt: false,
        reid_threshold_match: 0.50,
        reid_threshold_new: 0.70,
        reid_batch_size: 16,
        reid_use_tensorrt: false,
        tensorrt_precision: 'fp16',
        gallery_max_size: 1000,
        display_scale: 0.5,
      });
      return;
    }

    const selectedConfig = savedConfigs.find((c) => c.config_id === configId);
    if (selectedConfig) {
      const parsedConfig = typeof selectedConfig.config === 'string'
        ? JSON.parse(selectedConfig.config)
        : selectedConfig.config;
      // Ensure display_scale is included for multi-camera
      setConfig({
        ...parsedConfig,
        display_scale: parsedConfig.display_scale || 0.5,
      });
    }
  };

  const uploadVideo = async (file, index) => {
    try {
      const response = await apiService.uploadVideo(file);
      const newVideos = [...videos];
      newVideos[index] = response.data;
      setVideos(newVideos);
    } catch (err) {
      setError('Failed to upload video: ' + (err.response?.data?.detail || err.message));
    }
  };

  const VideoUploadZone = ({ index }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop: (files) => {
        if (files.length > 0) {
          uploadVideo(files[0], index);
        }
      },
      accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv'] },
      maxFiles: 1,
    });

    return (
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: videos[index] ? 'success.main' : isDragActive ? 'primary.main' : 'grey.700',
          borderRadius: 2,
          p: 3,
          textAlign: 'center',
          cursor: 'pointer',
          bgcolor: videos[index] ? 'success.dark' : isDragActive ? 'action.hover' : 'transparent',
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 40, color: 'grey.500', mb: 1 }} />
        <Typography variant="body2">
          {videos[index] ? videos[index].filename : `Camera ${index + 1}`}
        </Typography>
      </Box>
    );
  };

  const handleStart = async () => {
    const allUploaded = videos.every((v) => v !== null);
    if (!allUploaded) {
      setError('Please upload all 4 videos');
      return;
    }

    setStarting(true);
    setError(null);

    try {
      const videoPaths = videos.map((v) => v.path);
      const response = await apiService.startMultiCameraPipeline(videoPaths, config);
      const jobId = response.data.job_id;

      navigate(`/jobs/${jobId}`);
    } catch (err) {
      setError('Failed to start pipeline: ' + (err.response?.data?.detail || err.message));
      setStarting(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" component="h1" gutterBottom>
        Multi-Camera Pipeline (4 Streams)
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          1. Upload 4 Video Streams
        </Typography>

        <Grid container spacing={2}>
          {[0, 1, 2, 3].map((index) => (
            <Grid item xs={6} key={index}>
              <VideoUploadZone index={index} />
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Configuration Selector */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          2. Configuration
        </Typography>

        <FormControl fullWidth>
          <InputLabel>Load Saved Configuration</InputLabel>
          <Select
            value={selectedConfigId}
            label="Load Saved Configuration"
            onChange={handleLoadConfig}
            startAdornment={<SettingsIcon sx={{ mr: 1, color: 'text.secondary' }} />}
          >
            <MenuItem value="">
              <em>Default Configuration</em>
            </MenuItem>
            {savedConfigs.map((savedConfig) => (
              <MenuItem key={savedConfig.config_id} value={savedConfig.config_id}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                  <Typography>{savedConfig.name}</Typography>
                  <Box sx={{ flexGrow: 1 }} />
                  <Chip
                    label={
                      typeof savedConfig.config === 'string'
                        ? JSON.parse(savedConfig.config).preset || 'N/A'
                        : savedConfig.config.preset || 'N/A'
                    }
                    size="small"
                    color="primary"
                  />
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {selectedConfigId && (
          <Alert severity="info" sx={{ mt: 2 }}>
            Configuration "{savedConfigs.find(c => c.config_id === selectedConfigId)?.name}" loaded.
          </Alert>
        )}

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ mt: 2 }}>
          Basic Settings
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Device</InputLabel>
              <Select
                value={config.device}
                label="Device"
                onChange={(e) => setConfig({ ...config, device: e.target.value })}
              >
                <MenuItem value="cuda">CUDA (GPU)</MenuItem>
                <MenuItem value="cpu">CPU</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Display Scale"
              type="number"
              inputProps={{ step: 0.1, min: 0.1, max: 1.0 }}
              value={config.display_scale}
              onChange={(e) => setConfig({ ...config, display_scale: parseFloat(e.target.value) })}
              helperText="Scale for 2x2 grid display (0.1-1.0)"
            />
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.enable_display}
                  onChange={(e) => setConfig({ ...config, enable_display: e.target.checked })}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Enable Display Window</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Shows live 2x2 grid preview. Disable for Docker/server deployments.
                  </Typography>
                </Box>
              }
            />
          </Grid>
        </Grid>

        <Divider sx={{ my: 3 }} />

        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Model Configuration
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>YOLO Model</InputLabel>
              <Select
                value={config.yolo_model}
                label="YOLO Model"
                onChange={(e) => setConfig({ ...config, yolo_model: e.target.value })}
                disabled={loadingModels}
              >
                {yoloModels.length === 0 && !loadingModels && (
                  <MenuItem value="" disabled>
                    <em>No YOLO models found in /app/models</em>
                  </MenuItem>
                )}
                {yoloModels.map((model) => (
                  <MenuItem key={model.filename} value={model.filename}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Typography>{model.filename}</Typography>
                      <Box sx={{ flexGrow: 1 }} />
                      <Chip
                        label={model.type.toUpperCase()}
                        size="small"
                        color={model.type === 'engine' ? 'success' : 'default'}
                        sx={{ fontSize: '0.7rem' }}
                      />
                    </Box>
                  </MenuItem>
                ))}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, ml: 1.5 }}>
                Person detection model (.pt or .engine)
              </Typography>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>ReID Model (optional)</InputLabel>
              <Select
                value={config.reid_model}
                label="ReID Model (optional)"
                onChange={(e) => setConfig({ ...config, reid_model: e.target.value })}
                disabled={loadingModels}
              >
                <MenuItem value="">
                  <em>None (use default)</em>
                </MenuItem>
                {reidModels.length === 0 && !loadingModels && (
                  <MenuItem value="" disabled>
                    <em>No ReID models found in /app/models</em>
                  </MenuItem>
                )}
                {reidModels.map((model) => (
                  <MenuItem key={model.filename} value={model.filename}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Typography>{model.filename}</Typography>
                      <Box sx={{ flexGrow: 1 }} />
                      <Chip
                        label={model.type.toUpperCase()}
                        size="small"
                        color={model.type === 'engine' ? 'success' : 'default'}
                        sx={{ fontSize: '0.7rem' }}
                      />
                    </Box>
                  </MenuItem>
                ))}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, ml: 1.5 }}>
                Feature extraction model (.pth, .pt, or .engine)
              </Typography>
            </FormControl>
          </Grid>
        </Grid>

        <Divider sx={{ my: 3 }} />

        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2">TensorRT Acceleration</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  TensorRT optimizes models for NVIDIA GPUs. Enable for Jetson devices or when using .engine model files.
                </Alert>
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.detection_use_tensorrt}
                      onChange={(e) => setConfig({ ...config, detection_use_tensorrt: e.target.checked })}
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2">YOLO Detection TensorRT</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Accelerate person detection model
                      </Typography>
                    </Box>
                  }
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.reid_use_tensorrt}
                      onChange={(e) => setConfig({ ...config, reid_use_tensorrt: e.target.checked })}
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2">ReID Feature TensorRT</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Accelerate feature extraction model
                      </Typography>
                    </Box>
                  }
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>TensorRT Precision</InputLabel>
                  <Select
                    value={config.tensorrt_precision}
                    label="TensorRT Precision"
                    onChange={(e) => setConfig({ ...config, tensorrt_precision: e.target.value })}
                  >
                    <MenuItem value="fp16">FP16 (Recommended)</MenuItem>
                    <MenuItem value="fp32">FP32 (Highest Accuracy)</MenuItem>
                    <MenuItem value="int8">INT8 (Fastest, Orin only)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Divider sx={{ my: 3 }} />

        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2">Detection & Tracking Settings</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Detection Confidence"
                  type="number"
                  inputProps={{ step: 0.05, min: 0, max: 1 }}
                  value={config.detection_conf}
                  onChange={(e) => setConfig({ ...config, detection_conf: parseFloat(e.target.value) })}
                  helperText="Min confidence for person detection (0.0-1.0)"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="ReID Match Threshold"
                  type="number"
                  inputProps={{ step: 0.05, min: 0, max: 1 }}
                  value={config.reid_threshold_match}
                  onChange={(e) => setConfig({ ...config, reid_threshold_match: parseFloat(e.target.value) })}
                  helperText="Min similarity to match existing person (0.0-1.0)"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="ReID New Person Threshold"
                  type="number"
                  inputProps={{ step: 0.05, min: 0, max: 1 }}
                  value={config.reid_threshold_new}
                  onChange={(e) => setConfig({ ...config, reid_threshold_new: parseFloat(e.target.value) })}
                  helperText="Max similarity to create new person (0.0-1.0)"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Gallery Max Size"
                  type="number"
                  inputProps={{ min: 1 }}
                  value={config.gallery_max_size}
                  onChange={(e) => setConfig({ ...config, gallery_max_size: parseInt(e.target.value) })}
                  helperText="Max number of unique persons to track"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="ReID Batch Size"
                  type="number"
                  inputProps={{ min: 1 }}
                  value={config.reid_batch_size}
                  onChange={(e) => setConfig({ ...config, reid_batch_size: parseInt(e.target.value) })}
                  helperText="Processing batch size (higher = faster, more memory)"
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </Paper>

      <Box display="flex" justifyContent="center">
        <Button
          variant="contained"
          size="large"
          startIcon={starting ? <CircularProgress size={20} /> : <PlayArrowIcon />}
          onClick={handleStart}
          disabled={starting || videos.some((v) => v === null)}
        >
          {starting ? 'Starting...' : 'Start Multi-Camera Pipeline'}
        </Button>
      </Box>
    </Container>
  );
}

export default MultiCameraPipeline;
