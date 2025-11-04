import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Grid,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { apiService } from '../services/api';
import { useNavigate } from 'react-router-dom';

function MultiCameraPipeline() {
  const navigate = useNavigate();
  const [videos, setVideos] = useState([null, null, null, null]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);

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
      const config = {
        device: 'cuda',
        yolo_model: 'yolo11n.pt',
        detection_conf: 0.3,
        reid_threshold_match: 0.50,
        reid_threshold_new: 0.70,
        gallery_max_size: 1000,
        reid_batch_size: 16,
        use_tensorrt: false,
        display_scale: 0.5,
      };

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
          Upload 4 Video Streams
        </Typography>

        <Grid container spacing={2}>
          {[0, 1, 2, 3].map((index) => (
            <Grid item xs={6} key={index}>
              <VideoUploadZone index={index} />
            </Grid>
          ))}
        </Grid>
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
