import React, { useState, useEffect } from 'react';
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
  Button,
  CircularProgress,
  Alert,
  IconButton,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';

function EvaluationJobs() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      setLoading(true);
      const response = await apiService.listEvaluationJobs(50, 0);
      setJobs(response.data.jobs || []);
      setError(null);
    } catch (err) {
      console.error('Error loading evaluation jobs:', err);
      setError('Failed to load evaluation jobs');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'running':
        return 'primary';
      default:
        return 'warning';
    }
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
            Evaluation Jobs
          </Typography>
          <Box display="flex" gap={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadJobs}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => navigate('/evaluation')}
            >
              New Evaluation
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper sx={{ p: 0 }}>
          {loading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : jobs.length === 0 ? (
            <Box p={4} textAlign="center">
              <Typography variant="h6" color="textSecondary" gutterBottom>
                No evaluation jobs found
              </Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => navigate('/evaluation')}
                sx={{ mt: 2 }}
              >
                Start First Evaluation
              </Button>
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Job ID</TableCell>
                    <TableCell>Dataset</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>mAP</TableCell>
                    <TableCell>Rank-1</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Completed</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {jobs.map((job) => (
                    <TableRow key={job.eval_job_id} hover>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {job.eval_job_id.substring(0, 8)}...
                        </Typography>
                      </TableCell>
                      <TableCell>{job.dataset_id}</TableCell>
                      <TableCell>
                        <Chip
                          label={job.status}
                          color={getStatusColor(job.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {job.map_score
                          ? `${(job.map_score * 100).toFixed(2)}%`
                          : '-'}
                      </TableCell>
                      <TableCell>
                        {job.rank1_accuracy
                          ? `${(job.rank1_accuracy * 100).toFixed(2)}%`
                          : '-'}
                      </TableCell>
                      <TableCell>{formatDate(job.created_at)}</TableCell>
                      <TableCell>{formatDate(job.completed_at)}</TableCell>
                      <TableCell align="right">
                        <IconButton
                          color="primary"
                          onClick={() =>
                            navigate(`/evaluation/results/${job.eval_job_id}`)
                          }
                          title="View Results"
                        >
                          <ViewIcon />
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
    </Container>
  );
}

export default EvaluationJobs;
