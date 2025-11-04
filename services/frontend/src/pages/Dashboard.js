import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';
import { apiService, wsService } from '../services/api';

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    running: 0,
    completed: 0,
    failed: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();

    // Connect to WebSocket for real-time updates
    wsService.connect(handleWebSocketMessage);

    return () => {
      wsService.disconnect();
    };
  }, []);

  const loadData = async () => {
    try {
      // Load health status
      const healthRes = await apiService.healthCheck();
      setHealth(healthRes.data);

      // Load recent jobs
      const jobsRes = await apiService.listJobs(10, 0);
      const jobsList = jobsRes.data.jobs;
      setJobs(jobsList);

      // Calculate stats
      const statsData = {
        total: jobsList.length,
        running: jobsList.filter((j) => j.status === 'running').length,
        completed: jobsList.filter((j) => j.status === 'completed').length,
        failed: jobsList.filter((j) => j.status === 'failed').length,
      };
      setStats(statsData);

      setLoading(false);
    } catch (error) {
      console.error('Error loading data:', error);
      setLoading(false);
    }
  };

  const handleWebSocketMessage = (data) => {
    if (data.type === 'job_updates') {
      // Refresh job list when updates arrive
      loadData();
    }
  };

  const StatCard = ({ title, value, icon, color }) => (
    <Card sx={{ height: '100%', bgcolor: color + '20' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography variant="h3" component="div">
              {value}
            </Typography>
          </Box>
          <Box sx={{ color: color, fontSize: 48 }}>{icon}</Box>
        </Box>
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

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard
        </Typography>

        {/* Health Status */}
        {health && (
          <Alert
            severity={
              health.redis === 'connected' && health.database === 'connected'
                ? 'success'
                : 'warning'
            }
            sx={{ mb: 3 }}
          >
            System Status: Redis {health.redis}, Database {health.database}
          </Alert>
        )}

        {/* Statistics Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Jobs"
              value={stats.total}
              icon={<PlayArrowIcon />}
              color="#90caf9"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Running"
              value={stats.running}
              icon={<PendingIcon />}
              color="#ffa726"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Completed"
              value={stats.completed}
              icon={<CheckCircleIcon />}
              color="#66bb6a"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Failed"
              value={stats.failed}
              icon={<ErrorIcon />}
              color="#ef5350"
            />
          </Grid>
        </Grid>

        {/* Recent Jobs */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recent Jobs
          </Typography>
          {jobs.length === 0 ? (
            <Typography color="textSecondary">No jobs yet</Typography>
          ) : (
            <Box>
              {jobs.slice(0, 5).map((job) => (
                <Box
                  key={job.job_id}
                  sx={{
                    py: 2,
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                    '&:last-child': { borderBottom: 'none' },
                  }}
                >
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="body1">Job {job.job_id.slice(0, 8)}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {new Date(job.created_at).toLocaleString()}
                      </Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <Typography
                        variant="body2"
                        sx={{
                          px: 2,
                          py: 0.5,
                          borderRadius: 1,
                          bgcolor:
                            job.status === 'completed'
                              ? 'success.main'
                              : job.status === 'running'
                              ? 'warning.main'
                              : job.status === 'failed'
                              ? 'error.main'
                              : 'grey.700',
                        }}
                      >
                        {job.status}
                      </Typography>
                      {job.progress !== undefined && (
                        <Typography variant="body2">{job.progress.toFixed(1)}%</Typography>
                      )}
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default Dashboard;
