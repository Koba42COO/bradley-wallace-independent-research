import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Cpu, Activity, Power, Settings } from "lucide-react";

interface VirtualGPUCardProps {
  id: string;
  name: string;
  cores: number;
  utilization: number;
  status: "active" | "idle" | "error" | "disabled";
  memory: {
    used: number;
    total: number;
  };
  onToggle: (id: string) => void;
  onConfigure: (id: string) => void;
}

export default function VirtualGPUCard({
  id,
  name,
  cores,
  utilization,
  status,
  memory,
  onToggle,
  onConfigure,
}: VirtualGPUCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-chart-1";
      case "idle": return "bg-muted";
      case "error": return "bg-destructive";
      case "disabled": return "bg-muted-foreground";
      default: return "bg-muted";
    }
  };

  const memoryPercent = memory.total > 0 ? (memory.used / memory.total) * 100 : 0;

  return (
    <Card className="hover-elevate" data-testid={`card-virtual-gpu-${id}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          {name}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge 
            variant="outline" 
            className={`${getStatusColor(status)} border-0 text-white`}
            data-testid={`status-${id}`}
          >
            {status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">CPU Cores</div>
              <div className="font-mono font-medium">{cores}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Utilization</div>
              <div className="font-mono font-medium">{utilization}%</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">CPU Utilization</span>
              <span className="font-mono">{utilization}%</span>
            </div>
            <Progress value={utilization} className="h-2" data-testid={`progress-utilization-${id}`} />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Memory</span>
              <span className="font-mono">{memory.used}GB / {memory.total}GB</span>
            </div>
            <Progress value={memoryPercent} className="h-2" data-testid={`progress-memory-${id}`} />
          </div>

          <div className="flex justify-between gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onToggle(id)}
              data-testid={`button-toggle-${id}`}
            >
              <Power className="h-3 w-3 mr-1" />
              {status === "active" ? "Stop" : "Start"}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onConfigure(id)}
              data-testid={`button-configure-${id}`}
            >
              <Settings className="h-3 w-3 mr-1" />
              Configure
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}