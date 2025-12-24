import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
  Alert,
  Chip,
  CircularProgress,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Assessment as AssessmentIcon,
  FolderOpen as FolderIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';

function DatasetManager() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [datasetName, setDatasetName] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    loadDatasets();
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

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadFile(file);
      // Auto-generate name from filename
      const name = file.name.replace(/\.(zip|tar\.gz|tar)$/i, '');
      setDatasetName(name);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile || !datasetName.trim()) {
      setError('Please select a file and enter a dataset name');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setUploadProgress(0);

      await apiService.uploadDataset(
        uploadFile,
        datasetName.trim(),
        (progress) => setUploadProgress(progress)
      );

      setSuccess('Dataset uploaded successfully!');
      setUploadDialogOpen(false);
      setUploadFile(null);
      setDatasetName('');
      setUploadProgress(0);
      loadDatasets();
    } catch (err) {
      console.error('Error uploading dataset:', err);
      setError(err.response?.data?.detail || 'Failed to upload dataset');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    try {
      await apiService.deleteDataset(datasetId);
      setSuccess('Dataset deleted successfully');
      loadDatasets();
    } catch (err) {
      console.error('Error deleting dataset:', err);
      setError(err.response?.data?.detail || 'Failed to delete dataset');
    }
  };

  const handleEvaluate = (datasetId) => {
    navigate('/evaluation', { state: { datasetId } });
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const mb = bytes / (1024 * 1024);
    return mb > 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(2)} MB`;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h1" fontWeight="bold">
            Dataset Manager
          </Typography>
          <Button
            variant="contained"
            startIcon={<UploadIcon />}
            onClick={() => setUploadDialogOpen(true)}
            size="large"
          >
            Upload Dataset
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

        <Paper sx={{ p: 0 }}>
          {loading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : datasets.length === 0 ? (
            <Box p={4} textAlign="center">
              <FolderIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="textSecondary" gutterBottom>
                No datasets uploaded yet
              </Typography>
              <Typography variant="body2" color="textSecondary" mb={3}>
                Upload a Market-1501 format dataset to get started
              </Typography>
              <Button
                variant="contained"
                startIcon={<UploadIcon />}
                onClick={() => setUploadDialogOpen(true)}
              >
                Upload Dataset
              </Button>
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Format</TableCell>
                    <TableCell>Queries</TableCell>
                    <TableCell>Gallery</TableCell>
                    <TableCell>Train</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Uploaded</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {datasets.map((dataset) => (
                    <TableRow key={dataset.dataset_id} hover>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {dataset.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={dataset.format || 'Market-1501'} size="small" />
                      </TableCell>
                      <TableCell>{dataset.num_queries || 'N/A'}</TableCell>
                      <TableCell>{dataset.num_gallery || 'N/A'}</TableCell>
                      <TableCell>{dataset.num_train || 'N/A'}</TableCell>
                      <TableCell>{formatFileSize(dataset.size)}</TableCell>
                      <TableCell>{formatDate(dataset.created_at)}</TableCell>
                      <TableCell align="right">
                        <IconButton
                          color="primary"
                          onClick={() => handleEvaluate(dataset.dataset_id)}
                          title="Evaluate"
                        >
                          <AssessmentIcon />
                        </IconButton>
                        <IconButton
                          color="error"
                          onClick={() => handleDelete(dataset.dataset_id)}
                          title="Delete"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Box>

      {/* Upload Dialog */}
      <Dialog
        open={uploadDialogOpen}
        onClose={() => !uploading && setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Upload Dataset</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Dataset Name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={uploading}
              sx={{ mb: 3 }}
            />

            <Button
              variant="outlined"
              component="label"
              fullWidth
              disabled={uploading}
              sx={{ mb: 2 }}
            >
              {uploadFile ? uploadFile.name : 'Select ZIP File'}
              <input
                type="file"
                hidden
                accept=".zip"
                onChange={handleFileSelect}
              />
            </Button>

            {uploadFile && (
              <Alert severity="info" sx={{ mb: 2 }}>
                File size: {formatFileSize(uploadFile.size)}
              </Alert>
            )}

            {uploading && (
              <Box sx={{ width: '100%', mb: 2 }}>
                <LinearProgress variant="determinate" value={uploadProgress} />
                <Typography variant="caption" color="textSecondary" align="center" display="block" mt={1}>
                  Uploading: {uploadProgress}%
                </Typography>
              </Box>
            )}

            <Alert severity="info">
              Upload a Market-1501 format dataset as a ZIP file containing:
              <ul>
                <li>query/</li>
                <li>bounding_box_test/</li>
                <li>bounding_box_train/</li>
              </ul>
            </Alert>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)} disabled={uploading}>
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!uploadFile || !datasetName.trim() || uploading}
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default DatasetManager;
