import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp } from "lucide-react";
import { useState } from "react";

interface PerformanceDataPoint {
  time: string;
  cpuUtilization: number;
  memoryUsage: number;
  throughput: number;
  activeJobs: number;
}

interface PerformanceChartProps {
  data: PerformanceDataPoint[];
}

export default function PerformanceChart({ data }: PerformanceChartProps) {
  const [metric, setMetric] = useState("cpuUtilization");

  const getMetricConfig = (metricType: string) => {
    switch (metricType) {
      case "cpuUtilization":
        return {
          title: "CPU Utilization",
          dataKey: "cpuUtilization",
          color: "hsl(var(--chart-1))",
          unit: "%",
          max: 100
        };
      case "memoryUsage":
        return {
          title: "Memory Usage",
          dataKey: "memoryUsage", 
          color: "hsl(var(--chart-2))",
          unit: "%",
          max: 100
        };
      case "throughput":
        return {
          title: "Throughput",
          dataKey: "throughput",
          color: "hsl(var(--chart-3))",
          unit: " ops/sec",
          max: undefined
        };
      case "activeJobs":
        return {
          title: "Active Jobs",
          dataKey: "activeJobs",
          color: "hsl(var(--chart-4))",
          unit: " jobs",
          max: undefined
        };
      default:
        return {
          title: "CPU Utilization",
          dataKey: "cpuUtilization",
          color: "hsl(var(--chart-1))",
          unit: "%",
          max: 100
        };
    }
  };

  const config = getMetricConfig(metric);

  return (
    <Card data-testid="card-performance-chart">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Performance Metrics
          </CardTitle>
          <Select value={metric} onValueChange={setMetric}>
            <SelectTrigger className="w-48" data-testid="select-metric">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cpuUtilization">CPU Utilization</SelectItem>
              <SelectItem value="memoryUsage">Memory Usage</SelectItem>
              <SelectItem value="throughput">Throughput</SelectItem>
              <SelectItem value="activeJobs">Active Jobs</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={config.color} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={config.color} stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="time" 
                className="text-xs"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                className="text-xs"
                tick={{ fontSize: 12 }}
                domain={config.max ? [0, config.max] : ['auto', 'auto']}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
                formatter={(value: number) => [
                  `${value}${config.unit}`,
                  config.title
                ]}
              />
              <Area
                type="monotone"
                dataKey={config.dataKey}
                stroke={config.color}
                strokeWidth={2}
                fill="url(#colorGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}