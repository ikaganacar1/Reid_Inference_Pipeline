import axios from 'axios';

// Use relative URLs in production (goes through nginx proxy)
// In development, you can set REACT_APP_API_URL=http://localhost:8000
const API_BASE_URL = process.env.REACT_APP_API_URL || '';
// Automatically use wss:// for HTTPS and ws:// for HTTP
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || `${WS_PROTOCOL}//${window.location.host}`;

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API methods
export const apiService = {
  // Health check
  healthCheck: () => api.get('/api/health'),

  // Video upload
  uploadVideo: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    return api.post('/api/upload/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });
  },

  // Pipeline jobs
  startPipeline: (videoPath, config) =>
    api.post('/api/pipeline/start', config, {
      params: { video_path: videoPath },
    }),

  startMultiCameraPipeline: (videoPaths, config) =>
    api.post('/api/pipeline/multi-camera/start', config, {
      params: { video_paths: videoPaths },
    }),

  listJobs: (limit = 20, offset = 0, sortBy = 'created_at', sortOrder = 'desc') =>
    api.get('/api/jobs', { params: { limit, offset, sort_by: sortBy, sort_order: sortOrder } }),

  getJob: (jobId) => api.get(`/api/jobs/${jobId}`),

  cancelJob: (jobId) => api.delete(`/api/jobs/${jobId}`),

  bulkDeleteJobs: (jobIds) => api.post('/api/jobs/bulk-delete', jobIds),

  downloadOutput: (jobId) => api.get(`/api/output/${jobId}`, {
    responseType: 'blob',
  }),

  // Configurations
  listConfigs: () => api.get('/api/configs'),

  getConfig: (configId) => api.get(`/api/configs/${configId}`),

  saveConfig: (name, config) =>
    api.post('/api/configs', config, { params: { name } }),

  updateConfig: (configId, name, config) =>
    api.put(`/api/configs/${configId}`, config, { params: { name } }),

  deleteConfig: (configId) => api.delete(`/api/configs/${configId}`),

  // Models
  listModels: () => api.get('/api/models'),

  getBaseURL: () => API_BASE_URL || window.location.origin,
};

// WebSocket connection
export class WebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = new Map();
  }

  connect(onMessage) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.ws = new WebSocket(`${WS_BASE_URL}/ws`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      // Send ping every 30 seconds to keep connection alive
      this.pingInterval = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send('ping');
        }
      }, 30000);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessage) {
          onMessage(data);
        }
        // Notify all listeners
        this.listeners.forEach((callback) => callback(data));
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (this.pingInterval) {
        clearInterval(this.pingInterval);
      }
      // Attempt to reconnect after 5 seconds
      setTimeout(() => this.connect(onMessage), 5000);
    };
  }

  addListener(id, callback) {
    this.listeners.set(id, callback);
  }

  removeListener(id) {
    this.listeners.delete(id);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
    }
  }
}

export const wsService = new WebSocketService();

export default api;
