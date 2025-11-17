import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  LinearProgress,
  Divider,
  Chip,
  Switch,
  FormControlLabel,
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

function SingleCameraPipeline() {
  const navigate = useNavigate();
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [config, setConfig] = useState({
    preset: 'development',
    device: 'cuda',
    enable_display: false, // Disable by default for Docker/server deployments
    yolo_model: 'yolo11n.pt',
    reid_model: '',
    detection_conf: 0.3,
    detection_use_tensorrt: false,
    reid_threshold_match: 0.70,
    reid_threshold_new: 0.50,
    reid_batch_size: 16,
    reid_use_tensorrt: false,
    tensorrt_precision: 'fp16',
    gallery_max_size: 500,
  });
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [selectedConfigId, setSelectedConfigId] = useState('');
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [yoloModels, setYoloModels] = useState([]);
  const [reidModels, setReidModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    loadSavedConfigs();
    loadAvailableModels();
  }, []);

  const loadSavedConfigs = async () => {
    setLoadingConfigs(true);
    try {
      const response = await apiService.listConfigs();
      setSavedConfigs(response.data.configs || []);
    } catch (err) {
      console.error('Error loading configs:', err);
    } finally {
      setLoadingConfigs(false);
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
        preset: 'development',
        device: 'cuda',
        enable_display: false,
        yolo_model: 'yolo11n.pt',
        reid_model: '',
        detection_conf: 0.3,
        detection_use_tensorrt: false,
        reid_threshold_match: 0.70,
        reid_threshold_new: 0.50,
        reid_batch_size: 16,
        reid_use_tensorrt: false,
        tensorrt_precision: 'fp16',
        gallery_max_size: 500,
      });
      return;
    }

    const selectedConfig = savedConfigs.find((c) => c.config_id === configId);
    if (selectedConfig) {
      const parsedConfig = typeof selectedConfig.config === 'string'
        ? JSON.parse(selectedConfig.config)
        : selectedConfig.config;
      setConfig(parsedConfig);
    }
  };

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setError(null);

    try {
      const response = await apiService.uploadVideo(file, (progress) => {
        setUploadProgress(progress);
      });

      setUploadedVideo(response.data);
      setUploading(false);
      setUploadProgress(0);
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload video: ' + (err.response?.data?.detail || err.message));
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
    maxFiles: 1,
  });

  const handleConfigChange = (field) => (event) => {
    setConfig({
      ...config,
      [field]: event.target.value,
    });
  };

  const handleSwitchChange = (field) => (event) => {
    setConfig({
      ...config,
      [field]: event.target.checked,
    });
  };

  const handleStart = async () => {
    if (!uploadedVideo) {
      setError('Please upload a video first');
      return;
    }

    setStarting(true);
    setError(null);

    try {
      const response = await apiService.startPipeline(uploadedVideo.path, config);
      const jobId = response.data.job_id;

      // Navigate to job detail page
      navigate(`/jobs/${jobId}`);
    } catch (err) {
      console.error('Start pipeline error:', err);
      setError('Failed to start pipeline: ' + (err.response?.data?.detail || err.message));
      setStarting(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" component="h1" gutterBottom>
        Single Camera Pipeline
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Video Upload */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          1. Upload Video
        </Typography>

        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.700',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: isDragActive ? 'action.hover' : 'transparent',
            mb: 2,
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
          <Typography>
            {isDragActive
              ? 'Drop the video here'
              : 'Drag and drop a video file, or click to select'}
          </Typography>
          <Typography variant="caption" color="textSecondary">
            Supported formats: MP4, AVI, MOV, MKV
          </Typography>
        </Box>

        {uploading && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Uploading... {uploadProgress}%
            </Typography>
            <LinearProgress variant="determinate" value={uploadProgress} />
          </Box>
        )}

        {uploadedVideo && (
          <Alert severity="success">
            Video uploaded: {uploadedVideo.filename} (
            {(uploadedVideo.size / 1024 / 1024).toFixed(2)} MB)
          </Alert>
        )}
      </Paper>

      {/* Configuration */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          2. Configure Pipeline
        </Typography>

        {/* Load Saved Configuration */}
        <Box sx={{ mb: 3 }}>
          <FormControl fullWidth>
            <InputLabel>Load Saved Configuration</InputLabel>
            <Select
              value={selectedConfigId}
              label="Load Saved Configuration"
              onChange={handleLoadConfig}
              startAdornment={<SettingsIcon sx={{ mr: 1, color: 'text.secondary' }} />}
            >
              <MenuItem value="">
                <em>Manual Configuration</em>
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
              Configuration loaded. You can still modify the values below.
            </Alert>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Basic Settings
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Preset</InputLabel>
              <Select
                value={config.preset}
                label="Preset"
                onChange={handleConfigChange('preset')}
              >
                <MenuItem value="development">Development</MenuItem>
                <MenuItem value="xavier_nx">Xavier NX</MenuItem>
                <MenuItem value="orin_nx">Orin NX</MenuItem>
                <MenuItem value="agx_orin">AGX Orin</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Device</InputLabel>
              <Select value={config.device} label="Device" onChange={handleConfigChange('device')}>
                <MenuItem value="cuda">CUDA (GPU)</MenuItem>
                <MenuItem value="cpu">CPU</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.enable_display}
                  onChange={handleSwitchChange('enable_display')}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Enable Display Window</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Shows live video preview during processing. Disable for Docker/server deployments.
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
                onChange={handleConfigChange('yolo_model')}
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
                onChange={handleConfigChange('reid_model')}
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
                      onChange={handleSwitchChange('detection_use_tensorrt')}
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
                      onChange={handleSwitchChange('reid_use_tensorrt')}
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
                    onChange={handleConfigChange('tensorrt_precision')}
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
                  onChange={handleConfigChange('detection_conf')}
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
                  onChange={handleConfigChange('reid_threshold_match')}
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
                  onChange={handleConfigChange('reid_threshold_new')}
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
                  onChange={handleConfigChange('gallery_max_size')}
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
                  onChange={handleConfigChange('reid_batch_size')}
                  helperText="Processing batch size (higher = faster, more memory)"
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </Paper>

      {/* Start Button */}
      <Box display="flex" justifyContent="center">
        <Button
          variant="contained"
          size="large"
          startIcon={starting ? <CircularProgress size={20} /> : <PlayArrowIcon />}
          onClick={handleStart}
          disabled={!uploadedVideo || uploading || starting}
        >
          {starting ? 'Starting...' : 'Start Pipeline'}
        </Button>
      </Box>
    </Container>
  );
}

export default SingleCameraPipeline;
