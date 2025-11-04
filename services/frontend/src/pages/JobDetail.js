import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Button,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { apiService, wsService } from '../services/api';

function JobDetail() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadJob();

    // Connect to WebSocket for real-time updates
    wsService.connect(handleWebSocketMessage);
    wsService.addListener('jobDetail', handleWebSocketMessage);

    // Poll for updates every 10 seconds as fallback (WebSocket handles real-time updates)
    const interval = setInterval(loadJob, 10000);

    return () => {
      clearInterval(interval);
      wsService.removeListener('jobDetail');
    };
  }, [jobId]);

  const loadJob = async () => {
    try {
      const response = await apiService.getJob(jobId);
      setJob(response.data.job || response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading job:', error);
      setLoading(false);
    }
  };

  const handleWebSocketMessage = (data) => {
    if (data.type === 'job_updates') {
      const update = data.data.find((j) => j.job_id === jobId);
      if (update) {
        loadJob();
      }
    }
  };

  const handleDownload = async () => {
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

  if (loading || !job) {
    return (
      <Container>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <LinearProgress sx={{ width: '50%' }} />
        </Box>
      </Container>
    );
  }

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

  return (
    <Container maxWidth="lg">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Job Details
        </Typography>
        {job.status === 'completed' && job.output_video && (
          <Box display="flex" gap={2}>
            <Button
              variant="contained"
              startIcon={<VisibilityIcon />}
              onClick={() => navigate(`/results/${jobId}`)}
            >
              View Results
            </Button>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={handleDownload}
            >
              Download Output
            </Button>
          </Box>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Job Information */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Job Information
            </Typography>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell><strong>Job ID</strong></TableCell>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{job.job_id}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Status</strong></TableCell>
                  <TableCell>
                    <Chip label={job.status} color={getStatusColor(job.status)} size="small" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Progress</strong></TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={1}>
                      <LinearProgress
                        variant="determinate"
                        value={job.progress || 0}
                        sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="caption">
                        {(job.progress || 0).toFixed(1)}%
                      </Typography>
                    </Box>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Created</strong></TableCell>
                  <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                </TableRow>
                {job.started_at && (
                  <TableRow>
                    <TableCell><strong>Started</strong></TableCell>
                    <TableCell>{new Date(job.started_at).toLocaleString()}</TableCell>
                  </TableRow>
                )}
                {job.completed_at && (
                  <TableRow>
                    <TableCell><strong>Completed</strong></TableCell>
                    <TableCell>{new Date(job.completed_at).toLocaleString()}</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </Paper>
        </Grid>

        {/* Statistics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Statistics
            </Typography>
            {job.stats ? (
              <Table size="small">
                <TableBody>
                  {Object.entries(job.stats).map(([key, value]) => (
                    <TableRow key={key}>
                      <TableCell><strong>{key.replace(/_/g, ' ')}</strong></TableCell>
                      <TableCell>{typeof value === 'number' ? value.toLocaleString() : value}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <Typography color="textSecondary">No statistics available</Typography>
            )}
          </Paper>
        </Grid>

        {/* Configuration */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'background.default',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '0.875rem',
              }}
            >
              {JSON.stringify(job.config, null, 2)}
            </Box>
          </Paper>
        </Grid>

        {/* Error Message */}
        {job.error_message && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3, bgcolor: 'error.dark' }}>
              <Typography variant="h6" gutterBottom>
                Error
              </Typography>
              <Typography variant="body2">{job.error_message}</Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  );
}

export default JobDetail;
