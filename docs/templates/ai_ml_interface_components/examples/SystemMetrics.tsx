import SystemMetrics from '../SystemMetrics';

export default function SystemMetricsExample() {
  return (
    <div className="p-4">
      <SystemMetrics
        totalCores={64}
        activeCores={47}
        totalMemory={128}
        usedMemory={89}
        activeVGPUs={5}
        totalVGPUs={8}
        runningJobs={12}
        queuedJobs={6}
        throughput={1847}
        systemLoad={73}
      />
    </div>
  );
}