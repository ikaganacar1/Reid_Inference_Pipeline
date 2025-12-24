import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  MenuItem,
  Alert,
  CircularProgress,
  Divider,
} from '@mui/material';
import { PlayArrow as PlayIcon, List as ListIcon } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { apiService } from '../services/api';

function Evaluation() {
  const navigate = useNavigate();
  const location = useLocation();
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState({ reid_models: [], yolo_models: [] });
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Configuration state
  const [config, setConfig] = useState({
    dataset_id: location.state?.datasetId || '',
    reid_model: 'lttc_0.1.4.49.pth', // Default to PyTorch model
    reid_threshold_match: 0.70,
    reid_threshold_new: 0.50,
    gallery_max_size: 1500,
    reid_batch_size: 16,
    use_tensorrt: false,
    subset_size: null, // null = full dataset
  });

  useEffect(() => {
    loadDatasets();
    loadModels();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await apiService.listDatasets();
      setDatasets(response.data.datasets || []);
      setError(null);
    } catch (err) {
      console.error('Error loading datasets:', err);
      setError('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const loadModels = async () => {
    try {
      const response = await apiService.listModels();
      setModels({
        reid_models: response.data.reid_models || [],
        yolo_models: response.data.yolo_models || []
      });
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const handleConfigChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!config.dataset_id) {
      setError('Please select a dataset');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      // Convert string values to numbers where needed
      const submitConfig = {
        ...config,
        reid_threshold_match: parseFloat(config.reid_threshold_match),
        reid_threshold_new: parseFloat(config.reid_threshold_new),
        gallery_max_size: parseInt(config.gallery_max_size),
        reid_batch_size: parseInt(config.reid_batch_size),
        subset_size: config.subset_size ? parseInt(config.subset_size) : null,
      };

      const response = await apiService.startEvaluation(submitConfig);
      const jobId = response.data.eval_job_id;

      setSuccess('Evaluation started successfully!');
      setTimeout(() => {
        navigate(`/evaluation/results/${jobId}`);
      }, 1500);
    } catch (err) {
      console.error('Error starting evaluation:', err);
      setError(err.response?.data?.detail || 'Failed to start evaluation');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h1" fontWeight="bold">
            ReID Evaluation
          </Typography>
          <Button
            variant="outlined"
            startIcon={<ListIcon />}
            onClick={() => navigate('/evaluation/jobs')}
          >
            View Jobs
          </Button>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        <Paper sx={{ p: 4 }}>
          {loading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : (
            <form onSubmit={handleSubmit}>
              <Grid container spacing={3}>
                {/* Dataset Selection */}
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Dataset
                  </Typography>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    select
                    fullWidth
                    label="Select Dataset"
                    value={config.dataset_id}
                    onChange={(e) => handleConfigChange('dataset_id', e.target.value)}
                    required
                    disabled={submitting}
                  >
                    {datasets.length === 0 ? (
                      <MenuItem value="" disabled>
                        No datasets available
                      </MenuItem>
                    ) : (
                      datasets.map((dataset) => (
                        <MenuItem key={dataset.dataset_id} value={dataset.dataset_id}>
                          {dataset.name} ({dataset.num_queries} queries, {dataset.num_gallery} gallery)
                        </MenuItem>
                      ))
                    )}
                  </TextField>
                </Grid>

                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                </Grid>

                {/* Gallery Configuration */}
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Gallery Configuration
                  </Typography>
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="MATCH Threshold"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={config.reid_threshold_match}
                    onChange={(e) => handleConfigChange('reid_threshold_match', e.target.value)}
                    disabled={submitting}
                    helperText="Similarity ≥ threshold = MATCH (default: 0.70)"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="NEW Threshold"
                    type="number"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={config.reid_threshold_new}
                    onChange={(e) => handleConfigChange('reid_threshold_new', e.target.value)}
                    disabled={submitting}
                    helperText="Similarity < threshold = NEW (default: 0.50)"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Gallery Max Size"
                    type="number"
                    inputProps={{ min: 100, max: 5000, step: 100 }}
                    value={config.gallery_max_size}
                    onChange={(e) => handleConfigChange('gallery_max_size', e.target.value)}
                    disabled={submitting}
                    helperText="Maximum identities to track (default: 1500)"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                </Grid>

                {/* Model Configuration */}
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Model Configuration
                  </Typography>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    select
                    fullWidth
                    label="ReID Model"
                    value={config.reid_model || ''}
                    onChange={(e) => handleConfigChange('reid_model', e.target.value)}
                    disabled={submitting || models.reid_models.length === 0}
                    helperText="Select ReID model (.pth for PyTorch, .engine for TensorRT)"
                  >
                    {models.reid_models.length === 0 ? (
                      <MenuItem value="" disabled>
                        No models available
                      </MenuItem>
                    ) : (
                      models.reid_models.map((model) => (
                        <MenuItem key={model.filename} value={model.filename}>
                          {model.filename} ({model.type.toUpperCase()}, {(model.size / 1024 / 1024).toFixed(1)} MB)
                        </MenuItem>
                      ))
                    )}
                  </TextField>
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Batch Size"
                    type="number"
                    inputProps={{ min: 1, max: 64, step: 1 }}
                    value={config.reid_batch_size}
                    onChange={(e) => handleConfigChange('reid_batch_size', e.target.value)}
                    disabled={submitting}
                    helperText="ReID model batch size (default: 16)"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    select
                    fullWidth
                    label="Use TensorRT"
                    value={config.use_tensorrt}
                    onChange={(e) => handleConfigChange('use_tensorrt', e.target.value === 'true')}
                    disabled={submitting}
                    helperText="Use TensorRT engine for faster inference"
                  >
                    <MenuItem value={false}>No (PyTorch)</MenuItem>
                    <MenuItem value={true}>Yes (TensorRT)</MenuItem>
                  </TextField>
                </Grid>

                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                </Grid>

                {/* Advanced Options */}
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Advanced Options
                  </Typography>
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Subset Size (Optional)"
                    type="number"
                    inputProps={{ min: 10, max: 10000, step: 10 }}
                    value={config.subset_size || ''}
                    onChange={(e) => handleConfigChange('subset_size', e.target.value || null)}
                    disabled={submitting}
                    helperText="Limit queries for testing (leave empty for full dataset)"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Box display="flex" justifyContent="flex-end" gap={2} mt={3}>
                    <Button
                      variant="outlined"
                      onClick={() => navigate('/datasets')}
                      disabled={submitting}
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={submitting ? <CircularProgress size={20} /> : <PlayIcon />}
                      disabled={submitting || !config.dataset_id}
                      size="large"
                    >
                      {submitting ? 'Starting...' : 'Start Evaluation'}
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </form>
          )}
        </Paper>

        {/* Information Panel */}
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'rgba(99, 102, 241, 0.1)' }}>
          <Typography variant="h6" gutterBottom>
            About ReID Evaluation
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            This evaluation simulates the full pipeline (detection → ReID → gallery matching) on Market-1501 dataset images.
          </Typography>
          <Typography variant="body2" color="textSecondary">
            <strong>Standard Metrics:</strong> mAP (mean Average Precision), CMC curve, Rank-1/5/10 accuracy
          </Typography>
          <Typography variant="body2" color="textSecondary">
            <strong>Gallery Metrics:</strong> MATCH/UNCERTAIN/NEW decision statistics, gallery growth tracking
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            <strong>Performance Metrics:</strong> FPS, average inference times
          </Typography>
        </Paper>
      </Box>
    </Container>
  );
}

export default Evaluation;
