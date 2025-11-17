import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  LinearProgress,
  Tooltip,
  Checkbox,
  Button,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Toolbar,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  VideoLibrary as VideoLibraryIcon,
  DeleteSweep as DeleteSweepIcon,
  GetApp as GetAppIcon,
  Sort as SortIcon,
} from '@mui/icons-material';
import { apiService, wsService } from '../services/api';

function JobsList() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedJobs, setSelectedJobs] = useState([]);
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  useEffect(() => {
    loadJobs();

    // Connect to WebSocket for updates
    wsService.connect(handleWebSocketMessage);
    wsService.addListener('jobsList', handleWebSocketMessage);

    return () => {
      wsService.removeListener('jobsList');
    };
  }, [sortBy, sortOrder]);

  const loadJobs = async () => {
    try {
      const response = await apiService.listJobs(50, 0, sortBy, sortOrder);
      setJobs(response.data.jobs);
      setLoading(false);
    } catch (error) {
      console.error('Error loading jobs:', error);
      setLoading(false);
      showSnackbar('Failed to load jobs', 'error');
    }
  };

  const handleWebSocketMessage = (data) => {
    if (data.type === 'job_updates') {
      loadJobs();
    }
  };

  const handleSelectAll = (event) => {
    if (event.target.checked) {
      setSelectedJobs(jobs.map(job => job.job_id));
    } else {
      setSelectedJobs([]);
    }
  };

  const handleSelectJob = (jobId) => {
    setSelectedJobs(prev => {
      if (prev.includes(jobId)) {
        return prev.filter(id => id !== jobId);
      } else {
        return [...prev, jobId];
      }
    });
  };

  const handleViewJob = (jobId) => {
    navigate(`/jobs/${jobId}`);
  };

  const handleCancelJob = async (jobId) => {
    try {
      await apiService.cancelJob(jobId);
      loadJobs();
      showSnackbar('Job cancelled successfully', 'success');
    } catch (error) {
      console.error('Error cancelling job:', error);
      showSnackbar('Failed to cancel job', 'error');
    }
  };

  const handleBulkDelete = async () => {
    if (selectedJobs.length === 0) {
      showSnackbar('No jobs selected', 'warning');
      return;
    }

    if (!window.confirm(`Are you sure you want to delete ${selectedJobs.length} job(s)? This will also delete output files.`)) {
      return;
    }

    try {
      const response = await apiService.bulkDeleteJobs(selectedJobs);
      showSnackbar(
        `Deleted ${response.data.deleted_count} of ${response.data.total_requested} jobs`,
        response.data.errors.length > 0 ? 'warning' : 'success'
      );
      setSelectedJobs([]);
      loadJobs();
    } catch (error) {
      console.error('Error deleting jobs:', error);
      showSnackbar('Failed to delete jobs', 'error');
    }
  };

  const handleBulkDownload = async () => {
    if (selectedJobs.length === 0) {
      showSnackbar('No jobs selected', 'warning');
      return;
    }

    // Filter for completed jobs only
    const completedJobs = jobs.filter(job =>
      selectedJobs.includes(job.job_id) && job.status === 'completed' && job.output_video
    );

    if (completedJobs.length === 0) {
      showSnackbar('No completed jobs with output files selected', 'warning');
      return;
    }

    // Download each job sequentially
    for (const job of completedJobs) {
      try {
        await handleDownload(job.job_id);
      } catch (error) {
        console.error(`Error downloading job ${job.job_id}:`, error);
      }
    }

    showSnackbar(`Downloaded ${completedJobs.length} file(s)`, 'success');
  };

  const handleDownload = async (jobId) => {
    try {
      const response = await apiService.downloadOutput(jobId);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${jobId}_output.mp4`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading output:', error);
      throw error;
    }
  };

  const handleSortChange = (field) => {
    if (sortBy === field) {
      // Toggle sort order
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'warning';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'default';
      default:
        return 'info';
    }
  };

  if (loading) {
    return (
      <Container>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <LinearProgress sx={{ width: '50%' }} />
        </Box>
      </Container>
    );
  }

  const allSelected = jobs.length > 0 && selectedJobs.length === jobs.length;
  const someSelected = selectedJobs.length > 0 && selectedJobs.length < jobs.length;

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Pipeline Jobs
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              label="Sort By"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="created_at">Date Created</MenuItem>
              <MenuItem value="status">Status</MenuItem>
              <MenuItem value="progress">Progress</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Order</InputLabel>
            <Select
              value={sortOrder}
              label="Order"
              onChange={(e) => setSortOrder(e.target.value)}
            >
              <MenuItem value="desc">Descending</MenuItem>
              <MenuItem value="asc">Ascending</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {selectedJobs.length > 0 && (
        <Toolbar
          sx={{
            pl: { sm: 2 },
            pr: { xs: 1, sm: 1 },
            mb: 2,
            bgcolor: 'primary.light',
            borderRadius: 1,
          }}
        >
          <Typography sx={{ flex: '1 1 100%' }} color="inherit" variant="subtitle1">
            {selectedJobs.length} selected
          </Typography>
          <Tooltip title="Bulk Download">
            <Button
              variant="contained"
              color="primary"
              startIcon={<GetAppIcon />}
              onClick={handleBulkDownload}
              sx={{ mr: 1 }}
            >
              Download
            </Button>
          </Tooltip>
          <Tooltip title="Bulk Delete">
            <Button
              variant="contained"
              color="error"
              startIcon={<DeleteSweepIcon />}
              onClick={handleBulkDelete}
            >
              Delete
            </Button>
          </Tooltip>
        </Toolbar>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={someSelected}
                  checked={allSelected}
                  onChange={handleSelectAll}
                />
              </TableCell>
              <TableCell>Job ID</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Progress</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography color="textSecondary">No jobs found</Typography>
                </TableCell>
              </TableRow>
            ) : (
              jobs.map((job) => {
                const isSelected = selectedJobs.includes(job.job_id);
                return (
                  <TableRow
                    key={job.job_id}
                    hover
                    selected={isSelected}
                    onClick={() => handleSelectJob(job.job_id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                      <Checkbox
                        checked={isSelected}
                        onChange={() => handleSelectJob(job.job_id)}
                      />
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {job.job_id.slice(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Chip label={job.status} color={getStatusColor(job.status)} size="small" />
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <LinearProgress
                          variant="determinate"
                          value={job.progress || 0}
                          sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                        />
                        <Typography variant="caption">{(job.progress || 0).toFixed(0)}%</Typography>
                      </Box>
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Typography variant="body2">
                        {new Date(job.created_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      {job.completed_at && job.started_at ? (
                        <Typography variant="body2">
                          {Math.round(
                            (new Date(job.completed_at) - new Date(job.started_at)) / 1000
                          )}s
                        </Typography>
                      ) : (
                        <Typography variant="body2" color="textSecondary">
                          -
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                      <Tooltip title="View Details">
                        <IconButton size="small" onClick={() => handleViewJob(job.job_id)}>
                          <VisibilityIcon />
                        </IconButton>
                      </Tooltip>
                      {job.status === 'completed' && job.output_video && (
                        <>
                          <Tooltip title="View Results">
                            <IconButton
                              size="small"
                              onClick={() => navigate(`/results/${job.job_id}`)}
                              color="primary"
                            >
                              <VideoLibraryIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Download Output">
                            <IconButton size="small" onClick={() => handleDownload(job.job_id)}>
                              <DownloadIcon />
                            </IconButton>
                          </Tooltip>
                        </>
                      )}
                      {job.status === 'running' && (
                        <Tooltip title="Cancel Job">
                          <IconButton size="small" onClick={() => handleCancelJob(job.job_id)}>
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default JobsList;
