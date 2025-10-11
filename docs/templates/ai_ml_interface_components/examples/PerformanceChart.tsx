import PerformanceChart from '../PerformanceChart';

export default function PerformanceChartExample() {
  // //todo: remove mock functionality
  const mockData = [
    { time: "14:00", cpuUtilization: 45, memoryUsage: 62, throughput: 1200, activeJobs: 8 },
    { time: "14:05", cpuUtilization: 52, memoryUsage: 64, throughput: 1350, activeJobs: 10 },
    { time: "14:10", cpuUtilization: 48, memoryUsage: 67, throughput: 1180, activeJobs: 9 },
    { time: "14:15", cpuUtilization: 61, memoryUsage: 71, throughput: 1420, activeJobs: 12 },
    { time: "14:20", cpuUtilization: 58, memoryUsage: 69, throughput: 1380, activeJobs: 11 },
    { time: "14:25", cpuUtilization: 73, memoryUsage: 75, throughput: 1680, activeJobs: 15 },
    { time: "14:30", cpuUtilization: 67, memoryUsage: 72, throughput: 1550, activeJobs: 13 },
    { time: "14:35", cpuUtilization: 71, memoryUsage: 78, throughput: 1620, activeJobs: 14 },
    { time: "14:40", cpuUtilization: 69, memoryUsage: 76, throughput: 1590, activeJobs: 13 },
    { time: "14:45", cpuUtilization: 74, memoryUsage: 81, throughput: 1720, activeJobs: 16 },
  ];

  return (
    <div className="p-4">
      <PerformanceChart data={mockData} />
    </div>
  );
}