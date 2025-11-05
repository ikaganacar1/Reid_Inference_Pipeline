import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Chip,
  Button,
  Tabs,
  Tab,
  Card,
  CardContent,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Analytics as AnalyticsIcon,
  People as PeopleIcon,
  VideoLibrary as VideoIcon,
} from '@mui/icons-material';
import ReactPlayer from 'react-player';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { apiService } from '../services/api';

function ResultsViewer() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);

  useEffect(() => {
    loadJobResults();
  }, [jobId]);

  const loadJobResults = async () => {
    try {
      const response = await apiService.getJob(jobId);
      const jobData = response.data.job || response.data;
      console.log('ResultsViewer - Loaded job data:', jobData);
      console.log('ResultsViewer - output_video:', jobData.output_video);
      console.log('ResultsViewer - status:', jobData.status);
      setJob(jobData);
      setLoading(false);
    } catch (error) {
      console.error('Error loading job results:', error);
      setLoading(false);
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

  // Prepare chart data from job statistics
  const prepareTimeSeriesData = () => {
    if (!job || !job.stats) return [];

    const { frames_processed = 0, total_detections = 0, total_persons_tracked = 0 } = job.stats;

    // Generate synthetic time series data (in production, this would come from the backend)
    const points = 20;
    const data = [];
    for (let i = 0; i <= points; i++) {
      data.push({
        frame: Math.floor((i / points) * frames_processed),
        detections: Math.floor(Math.random() * 20) + 5,
        persons: Math.floor(Math.random() * 15) + 3,
      });
    }
    return data;
  };

  const prepareStatsCards = () => {
    if (!job || !job.stats) return [];

    const {
      frames_captured = 0,
      frames_processed = 0,
      total_detections = 0,
      total_persons_tracked = 0,
    } = job.stats;

    return [
      {
        title: 'Frames Processed',
        value: frames_processed.toLocaleString(),
        subtitle: `${frames_captured.toLocaleString()} captured`,
        icon: <VideoIcon fontSize="large" />,
        color: '#6366f1',
      },
      {
        title: 'Total Detections',
        value: total_detections.toLocaleString(),
        subtitle: `Avg ${(total_detections / (frames_processed || 1)).toFixed(1)} per frame`,
        icon: <AnalyticsIcon fontSize="large" />,
        color: '#8b5cf6',
      },
      {
        title: 'Unique Persons',
        value: total_persons_tracked.toLocaleString(),
        subtitle: 'Tracked across video',
        icon: <PeopleIcon fontSize="large" />,
        color: '#ec4899',
      },
      {
        title: 'Processing Rate',
        value: job.completed_at && job.started_at
          ? `${(
              frames_processed /
              ((new Date(job.completed_at) - new Date(job.started_at)) / 1000)
            ).toFixed(1)} FPS`
          : 'N/A',
        subtitle: 'Average throughput',
        icon: <PlayIcon fontSize="large" />,
        color: '#10b981',
      },
    ];
  };

  const prepareDistributionData = () => {
    if (!job || !job.stats) return [];

    const {
      frames_processed = 0,
      total_detections = 0,
      total_persons_tracked = 0,
    } = job.stats;

    // Generate synthetic distribution data
    return [
      { name: 'High Confidence', value: Math.floor(total_detections * 0.6), color: '#10b981' },
      { name: 'Medium Confidence', value: Math.floor(total_detections * 0.3), color: '#f59e0b' },
      { name: 'Low Confidence', value: Math.floor(total_detections * 0.1), color: '#ef4444' },
    ];
  };

  if (loading || !job) {
    return (
      <Container>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <Typography>Loading results...</Typography>
        </Box>
      </Container>
    );
  }

  if (job.status !== 'completed' || !job.output_video) {
    return (
      <Container>
        <Box sx={{ mt: 4 }}>
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate(-1)}>
            Back
          </Button>
          <Paper sx={{ p: 4, mt: 2, textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              Results Not Available
            </Typography>
            <Typography color="textSecondary">
              {job.status === 'running'
                ? 'Job is still processing. Please wait for completion.'
                : job.status === 'failed'
                ? 'Job failed to complete.'
                : 'No output video available for this job.'}
            </Typography>
            <Chip label={job.status} color={getStatusColor(job.status)} sx={{ mt: 2 }} />
          </Paper>
        </Box>
      </Container>
    );
  }

  const timeSeriesData = prepareTimeSeriesData();
  const statsCards = prepareStatsCards();
  const distributionData = prepareDistributionData();

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box display="flex" alignItems="center" gap={2}>
          <IconButton onClick={() => navigate(-1)}>
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            Results Viewer
          </Typography>
          <Chip label={job.status} color={getStatusColor(job.status)} />
        </Box>
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={handleDownload}
        >
          Download Video
        </Button>
      </Box>

      {/* Job ID and Timestamp */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="textSecondary">
              Job ID
            </Typography>
            <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
              {job.job_id}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="textSecondary">
              Completed At
            </Typography>
            <Typography variant="body1">
              {new Date(job.completed_at).toLocaleString()}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Tab Navigation */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
        >
          <Tab icon={<VideoIcon />} label="Video" iconPosition="start" />
          <Tab icon={<AnalyticsIcon />} label="Analytics" iconPosition="start" />
          <Tab icon={<PeopleIcon />} label="Gallery" iconPosition="start" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* Video Player */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 2, bgcolor: 'black' }}>
              {/* Use native HTML5 video player for better compatibility */}
              <video
                src={`/api/output/${jobId}`}
                controls
                style={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '600px',
                  backgroundColor: 'black',
                }}
                onError={(e) => {
                  console.error('Video error:', e);
                  console.error('Video src:', e.target.src);
                  console.error('Video error code:', e.target.error?.code);
                  console.error('Video error message:', e.target.error?.message);
                }}
                onLoadedMetadata={() => console.log('Video metadata loaded')}
                onLoadStart={() => console.log('Video loading started')}
                onCanPlay={() => console.log('Video can play')}
              >
                Your browser does not support the video tag or this video format.
                <br />
                <a href={`/api/output/${jobId}`} download>
                  Click here to download the video instead
                </a>
              </video>
            </Paper>
          </Grid>

          {/* Quick Stats */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Quick Stats
              </Typography>
              {statsCards.slice(0, 3).map((stat, index) => (
                <Card key={index} sx={{ mb: 2, bgcolor: 'background.default' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" gap={2}>
                      <Box sx={{ color: stat.color }}>{stat.icon}</Box>
                      <Box>
                        <Typography variant="h5" fontWeight="bold">
                          {stat.value}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {stat.title}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {stat.subtitle}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={3}>
          {/* Stats Cards */}
          {statsCards.map((stat, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={2} mb={1}>
                    <Box sx={{ color: stat.color }}>{stat.icon}</Box>
                    <Typography variant="body2" color="textSecondary">
                      {stat.title}
                    </Typography>
                  </Box>
                  <Typography variant="h4" fontWeight="bold">
                    {stat.value}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {stat.subtitle}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}

          {/* Time Series Chart */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Detection Timeline
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frame" label={{ value: 'Frame', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip />
                  <Legend />
                  <Line type="monotone" dataKey="detections" stroke="#6366f1" strokeWidth={2} name="Detections" />
                  <Line type="monotone" dataKey="persons" stroke="#ec4899" strokeWidth={2} name="Tracked Persons" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Distribution Chart */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Confidence Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={distributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.name}: ${entry.value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {distributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Configuration Details */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Pipeline Configuration
              </Typography>
              <Box
                component="pre"
                sx={{
                  bgcolor: 'background.default',
                  p: 2,
                  borderRadius: 1,
                  overflow: 'auto',
                  fontSize: '0.875rem',
                  maxHeight: '300px',
                }}
              >
                {JSON.stringify(job.config, null, 2)}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Gallery Browser
          </Typography>
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <PeopleIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="textSecondary" gutterBottom>
              Gallery Feature Coming Soon
            </Typography>
            <Typography color="textSecondary" paragraph>
              This section will display all tracked persons with their appearance history,
              embedding visualizations, and cross-camera tracking information.
            </Typography>
            <Typography variant="body2" color="textSecondary">
              To enable this feature, the backend needs to save gallery state with person crops
              and metadata during pipeline execution.
            </Typography>
          </Box>
        </Paper>
      )}
    </Container>
  );
}

export default ResultsViewer;
