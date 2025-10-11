import LogViewer from '../LogViewer';

export default function LogViewerExample() {
  const handleExport = () => {
    console.log('Exporting logs...');
  };

  // //todo: remove mock functionality
  const mockLogs = [
    {
      id: "1",
      timestamp: "14:32:15",
      level: "info" as const,
      source: "VGPU-Core",
      message: "Virtual GPU VGPU-Compute-01 initialized successfully with 8 cores"
    },
    {
      id: "2", 
      timestamp: "14:32:22",
      level: "info" as const,
      source: "JobScheduler",
      message: "Compute job 'Matrix Multiplication Task' queued for execution"
    },
    {
      id: "3",
      timestamp: "14:32:28",
      level: "warn" as const,
      source: "ResourceMgr",
      message: "Memory utilization approaching 85% threshold on VGPU-ML-02"
    },
    {
      id: "4",
      timestamp: "14:32:35",
      level: "error" as const,
      source: "VGPU-Core",
      message: "Failed to allocate resources for VGPU-Render-03: insufficient memory"
    },
    {
      id: "5",
      timestamp: "14:32:41",
      level: "debug" as const,
      source: "PythonBridge",
      message: "CUDNT accelerator function call completed in 0.023ms"
    },
    {
      id: "6",
      timestamp: "14:32:47",
      level: "info" as const,
      source: "JobScheduler",
      message: "Job 'Neural Network Training' started on VGPU-ML-02"
    }
  ];

  return (
    <div className="p-4">
      <LogViewer
        logs={mockLogs}
        onExport={handleExport}
      />
    </div>
  );
}