import React, { useState } from 'react';
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
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
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
    yolo_model: 'yolo11n.pt',
    reid_model: '',
    detection_conf: 0.3,
    reid_threshold_match: 0.70,
    reid_threshold_new: 0.50,
    gallery_max_size: 500,
    reid_batch_size: 16,
    use_tensorrt: false,
  });
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);

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
                <MenuItem value="cuda">CUDA</MenuItem>
                <MenuItem value="cpu">CPU</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="YOLO Model"
              value={config.yolo_model}
              onChange={handleConfigChange('yolo_model')}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="ReID Model (optional)"
              value={config.reid_model}
              onChange={handleConfigChange('reid_model')}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Detection Confidence"
              type="number"
              inputProps={{ step: 0.05, min: 0, max: 1 }}
              value={config.detection_conf}
              onChange={handleConfigChange('detection_conf')}
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
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Gallery Max Size"
              type="number"
              value={config.gallery_max_size}
              onChange={handleConfigChange('gallery_max_size')}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Batch Size"
              type="number"
              value={config.reid_batch_size}
              onChange={handleConfigChange('reid_batch_size')}
            />
          </Grid>
        </Grid>
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
