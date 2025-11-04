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
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  VideoLibrary as VideoLibraryIcon,
} from '@mui/icons-material';
import { apiService, wsService } from '../services/api';

function JobsList() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadJobs();

    // Connect to WebSocket for updates
    wsService.connect(handleWebSocketMessage);
    wsService.addListener('jobsList', handleWebSocketMessage);

    return () => {
      wsService.removeListener('jobsList');
    };
  }, []);

  const loadJobs = async () => {
    try {
      const response = await apiService.listJobs(50, 0);
      setJobs(response.data.jobs);
      setLoading(false);
    } catch (error) {
      console.error('Error loading jobs:', error);
      setLoading(false);
    }
  };

  const handleWebSocketMessage = (data) => {
    if (data.type === 'job_updates') {
      loadJobs();
    }
  };

  const handleViewJob = (jobId) => {
    navigate(`/jobs/${jobId}`);
  };

  const handleCancelJob = async (jobId) => {
    try {
      await apiService.cancelJob(jobId);
      loadJobs();
    } catch (error) {
      console.error('Error cancelling job:', error);
    }
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
    } catch (error) {
      console.error('Error downloading output:', error);
    }
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

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Pipeline Jobs
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
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
                <TableCell colSpan={6} align="center">
                  <Typography color="textSecondary">No jobs found</Typography>
                </TableCell>
              </TableRow>
            ) : (
              jobs.map((job) => (
                <TableRow key={job.job_id} hover>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {job.job_id.slice(0, 8)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip label={job.status} color={getStatusColor(job.status)} size="small" />
                  </TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={1}>
                      <LinearProgress
                        variant="determinate"
                        value={job.progress || 0}
                        sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="caption">{(job.progress || 0).toFixed(0)}%</Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {new Date(job.created_at).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {job.completed_at && job.started_at ? (
                      <Typography variant="body2">
                        {Math.round(
                          (new Date(job.completed_at) - new Date(job.started_at)) / 1000
                        )}
                        s
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="textSecondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">
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
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  );
}

export default JobsList;
