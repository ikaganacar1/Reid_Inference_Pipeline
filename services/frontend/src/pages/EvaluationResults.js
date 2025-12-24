import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Button,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import MetricsChart from '../components/MetricsChart';

function EvaluationResults() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadJob();
  }, [jobId]);

  useEffect(() => {
    if (!autoRefresh || !job || job.status === 'completed' || job.status === 'failed') {
      return;
    }

    const interval = setInterval(loadJob, 3000); // Refresh every 3 seconds
    return () => clearInterval(interval);
  }, [autoRefresh, job]);

  const loadJob = async () => {
    try {
      const response = await apiService.getEvaluationJob(jobId);
      setJob(response.data);
      setError(null);

      // Stop auto-refresh if job is completed or failed
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        setAutoRefresh(false);
      }
    } catch (err) {
      console.error('Error loading evaluation job:', err);
      setError(err.response?.data?.detail || 'Failed to load evaluation job');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'running':
        return <CircularProgress size={24} />;
      default:
        return <PendingIcon color="warning" />;
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

  const MetricCard = ({ title, value, subtitle, icon, color = 'primary' }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="overline" color="textSecondary">
            {title}
          </Typography>
          <Box sx={{ color: `${color}.main` }}>{icon}</Box>
        </Box>
        <Typography variant="h4" fontWeight="bold" mb={0.5}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="textSecondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Container>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
        <Button onClick={() => navigate('/evaluation/jobs')} sx={{ mt: 2 }}>
          Back to Jobs
        </Button>
      </Container>
    );
  }

  if (!job) {
    return (
      <Container>
        <Alert severity="warning" sx={{ mt: 3 }}>
          Job not found
        </Alert>
      </Container>
    );
  }

  const standardMetrics = job.standard_metrics || {};
  const galleryMetrics = job.gallery_stats || {};
  const performanceMetrics = job.performance_stats || {};

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Box>
            <Typography variant="h4" component="h1" fontWeight="bold" gutterBottom>
              Evaluation Results
            </Typography>
            <Typography variant="subtitle1" color="textSecondary">
              Job ID: {jobId}
            </Typography>
          </Box>
          <Box display="flex" gap={2} alignItems="center">
            <Chip
              icon={getStatusIcon(job.status)}
              label={job.status.toUpperCase()}
              color={getStatusColor(job.status)}
              size="large"
            />
            {job.status === 'running' && (
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={loadJob}
              >
                Refresh
              </Button>
            )}
          </Box>
        </Box>

        {/* Progress Bar for Running Jobs */}
        {job.status === 'running' && job.progress !== undefined && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="body2" gutterBottom>
              Progress: {job.progress}%
            </Typography>
            <LinearProgress variant="determinate" value={job.progress} />
          </Paper>
        )}

        {/* Error Message */}
        {job.status === 'failed' && job.error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {job.error}
          </Alert>
        )}

        {/* Results (only show for completed jobs) */}
        {job.status === 'completed' && (
          <>
            {/* Standard ReID Metrics */}
            <Typography variant="h5" fontWeight="bold" mb={2}>
              Standard ReID Metrics
            </Typography>
            <Grid container spacing={3} mb={4}>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="mAP"
                  value={`${(standardMetrics.map * 100).toFixed(2)}%`}
                  subtitle="Mean Average Precision"
                  icon={<TrendingUpIcon />}
                  color="success"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Rank-1"
                  value={`${(standardMetrics.rank1 * 100).toFixed(2)}%`}
                  subtitle="Single-shot accuracy"
                  icon={<TrendingUpIcon />}
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Rank-5"
                  value={`${(standardMetrics.rank5 * 100).toFixed(2)}%`}
                  subtitle="Top-5 accuracy"
                  icon={<TrendingUpIcon />}
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Rank-10"
                  value={`${(standardMetrics.rank10 * 100).toFixed(2)}%`}
                  subtitle="Top-10 accuracy"
                  icon={<TrendingUpIcon />}
                  color="primary"
                />
              </Grid>
            </Grid>

            {/* CMC Curve */}
            {standardMetrics.cmc_curve && (
              <Box mb={4}>
                <MetricsChart
                  cmcData={standardMetrics.cmc_curve}
                  title="Cumulative Matching Characteristics (CMC)"
                />
              </Box>
            )}

            {/* Gallery Simulation Statistics */}
            <Typography variant="h5" fontWeight="bold" mb={2}>
              Gallery Simulation Statistics
            </Typography>
            <Grid container spacing={3} mb={4}>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="MATCH Decisions"
                  value={galleryMetrics.total_match || 0}
                  subtitle={`${((galleryMetrics.match_rate || 0) * 100).toFixed(1)}% of queries`}
                  icon={<CheckCircleIcon />}
                  color="success"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="UNCERTAIN Decisions"
                  value={galleryMetrics.total_uncertain || 0}
                  subtitle={`${((galleryMetrics.uncertain_rate || 0) * 100).toFixed(1)}% of queries`}
                  icon={<PendingIcon />}
                  color="warning"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="NEW Decisions"
                  value={galleryMetrics.total_new || 0}
                  subtitle={`${((galleryMetrics.new_rate || 0) * 100).toFixed(1)}% of queries`}
                  icon={<StorageIcon />}
                  color="info"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Final Gallery Size"
                  value={galleryMetrics.final_gallery_size || 0}
                  subtitle="Tracked identities"
                  icon={<StorageIcon />}
                  color="primary"
                />
              </Grid>
            </Grid>

            {/* Performance Metrics */}
            <Typography variant="h5" fontWeight="bold" mb={2}>
              Performance Metrics
            </Typography>
            <Grid container spacing={3} mb={4}>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Processing FPS"
                  value={performanceMetrics.fps?.toFixed(2) || 'N/A'}
                  subtitle="Images per second"
                  icon={<SpeedIcon />}
                  color="success"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Avg ReID Time"
                  value={`${performanceMetrics.avg_reid_time?.toFixed(2) || 'N/A'} ms`}
                  subtitle="Per image"
                  icon={<SpeedIcon />}
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Avg Gallery Time"
                  value={`${performanceMetrics.avg_gallery_time?.toFixed(2) || 'N/A'} ms`}
                  subtitle="Per image"
                  icon={<SpeedIcon />}
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Total Duration"
                  value={`${performanceMetrics.total_duration?.toFixed(2) || 'N/A'} s`}
                  subtitle="Evaluation time"
                  icon={<SpeedIcon />}
                  color="info"
                />
              </Grid>
            </Grid>

            {/* Configuration Details */}
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Dataset ID</TableCell>
                      <TableCell>{job.dataset_id}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>MATCH Threshold</TableCell>
                      <TableCell>{job.config?.reid_threshold_match}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>NEW Threshold</TableCell>
                      <TableCell>{job.config?.reid_threshold_new}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Gallery Max Size</TableCell>
                      <TableCell>{job.config?.gallery_max_size}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Batch Size</TableCell>
                      <TableCell>{job.config?.reid_batch_size}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Use TensorRT</TableCell>
                      <TableCell>{job.config?.use_tensorrt ? 'Yes' : 'No'}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Created At</TableCell>
                      <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                    </TableRow>
                    {job.completed_at && (
                      <TableRow>
                        <TableCell>Completed At</TableCell>
                        <TableCell>{new Date(job.completed_at).toLocaleString()}</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </>
        )}
      </Box>
    </Container>
  );
}

export default EvaluationResults;
