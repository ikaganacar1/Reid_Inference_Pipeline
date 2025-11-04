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
} from '@mui/material';
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
  const [config, setConfig] = useState({
    device: 'cuda',
    yolo_model: 'yolo11n.pt',
    detection_conf: 0.3,
    reid_threshold_match: 0.50,
    reid_threshold_new: 0.70,
    gallery_max_size: 1000,
    reid_batch_size: 16,
    use_tensorrt: false,
    display_scale: 0.5,
  });

  useEffect(() => {
    loadSavedConfigs();
  }, []);

  const loadSavedConfigs = async () => {
    try {
      const response = await apiService.listConfigs();
      setSavedConfigs(response.data.configs || []);
    } catch (err) {
      console.error('Error loading configs:', err);
    }
  };

  const handleLoadConfig = (event) => {
    const configId = event.target.value;
    setSelectedConfigId(configId);

    if (!configId) {
      // Reset to default
      setConfig({
        device: 'cuda',
        yolo_model: 'yolo11n.pt',
        detection_conf: 0.3,
        reid_threshold_match: 0.50,
        reid_threshold_new: 0.70,
        gallery_max_size: 1000,
        reid_batch_size: 16,
        use_tensorrt: false,
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

        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Current Configuration
        </Typography>
        <Box sx={{ mt: 1, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Device:</Typography>
              <Typography variant="body2">{config.device}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Detection Conf:</Typography>
              <Typography variant="body2">{config.detection_conf}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Match Threshold:</Typography>
              <Typography variant="body2">{config.reid_threshold_match}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Gallery Size:</Typography>
              <Typography variant="body2">{config.gallery_max_size}</Typography>
            </Grid>
          </Grid>
        </Box>
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
