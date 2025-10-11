import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Cpu, HardDrive, Zap } from "lucide-react";

interface SystemMetricsProps {
  totalCores: number;
  activeCores: number;
  totalMemory: number;
  usedMemory: number;
  activeVGPUs: number;
  totalVGPUs: number;
  runningJobs: number;
  queuedJobs: number;
  throughput: number;
  systemLoad: number;
}

export default function SystemMetrics({
  totalCores,
  activeCores,
  totalMemory,
  usedMemory,
  activeVGPUs,
  totalVGPUs,
  runningJobs,
  queuedJobs,
  throughput,
  systemLoad,
}: SystemMetricsProps) {
  const coreUtilization = Math.round((activeCores / totalCores) * 100);
  const memoryUtilization = Math.round((usedMemory / totalMemory) * 100);
  const vgpuUtilization = Math.round((activeVGPUs / totalVGPUs) * 100);

  const getLoadStatus = (load: number) => {
    if (load < 50) return { color: "bg-chart-1", text: "Optimal" };
    if (load < 80) return { color: "bg-chart-2", text: "Moderate" };
    return { color: "bg-destructive", text: "High" };
  };

  const loadStatus = getLoadStatus(systemLoad);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card data-testid="card-cpu-metrics">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">CPU Resources</CardTitle>
          <Cpu className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="text-2xl font-bold font-mono">{coreUtilization}%</div>
            <p className="text-xs text-muted-foreground">
              {activeCores} of {totalCores} cores active
            </p>
            <div className="flex items-center gap-2">
              <div className="text-xs text-muted-foreground">Load:</div>
              <Badge className={`${loadStatus.color} border-0 text-white text-xs`}>
                {loadStatus.text}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-memory-metrics">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
          <HardDrive className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="text-2xl font-bold font-mono">{memoryUtilization}%</div>
            <p className="text-xs text-muted-foreground">
              {usedMemory}GB of {totalMemory}GB used
            </p>
            <div className="text-xs text-muted-foreground">
              {totalMemory - usedMemory}GB available
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-vgpu-metrics">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Virtual GPUs</CardTitle>
          <Zap className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="text-2xl font-bold font-mono">{activeVGPUs}/{totalVGPUs}</div>
            <p className="text-xs text-muted-foreground">
              {vgpuUtilization}% utilization
            </p>
            <div className="text-xs text-muted-foreground">
              {totalVGPUs - activeVGPUs} available
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-job-metrics">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Compute Jobs</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="text-2xl font-bold font-mono">{runningJobs}</div>
            <p className="text-xs text-muted-foreground">
              {queuedJobs} queued
            </p>
            <div className="text-xs text-muted-foreground">
              {throughput} ops/sec
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}