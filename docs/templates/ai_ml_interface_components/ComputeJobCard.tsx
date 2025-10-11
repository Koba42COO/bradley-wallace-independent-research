import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Timer, Zap, X, Play, Pause } from "lucide-react";

interface ComputeJobCardProps {
  id: string;
  name: string;
  status: "running" | "queued" | "completed" | "failed" | "paused";
  progress: number;
  duration: string;
  vgpuId: string;
  priority: "high" | "medium" | "low";
  onAction: (id: string, action: "start" | "pause" | "cancel") => void;
}

export default function ComputeJobCard({
  id,
  name,
  status,
  progress,
  duration,
  vgpuId,
  priority,
  onAction,
}: ComputeJobCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "bg-chart-1";
      case "queued": return "bg-chart-2";
      case "completed": return "bg-chart-1";
      case "failed": return "bg-destructive";
      case "paused": return "bg-muted";
      default: return "bg-muted";
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high": return "bg-destructive";
      case "medium": return "bg-chart-2";
      case "low": return "bg-muted";
      default: return "bg-muted";
    }
  };

  return (
    <Card className="hover-elevate" data-testid={`card-compute-job-${id}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Zap className="h-4 w-4" />
          {name}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge 
            variant="outline" 
            className={`${getPriorityColor(priority)} border-0 text-white text-xs`}
            data-testid={`priority-${id}`}
          >
            {priority}
          </Badge>
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
              <div className="text-muted-foreground">Virtual GPU</div>
              <div className="font-mono font-medium">{vgpuId}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Duration</div>
              <div className="font-mono font-medium flex items-center gap-1">
                <Timer className="h-3 w-3" />
                {duration}
              </div>
            </div>
          </div>
          
          {status === "running" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-mono">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" data-testid={`progress-${id}`} />
            </div>
          )}

          <div className="flex justify-between gap-2">
            {status === "running" ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onAction(id, "pause")}
                data-testid={`button-pause-${id}`}
              >
                <Pause className="h-3 w-3 mr-1" />
                Pause
              </Button>
            ) : status === "paused" || status === "queued" ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onAction(id, "start")}
                data-testid={`button-start-${id}`}
              >
                <Play className="h-3 w-3 mr-1" />
                Start
              </Button>
            ) : null}
            
            {(status === "running" || status === "queued" || status === "paused") && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onAction(id, "cancel")}
                data-testid={`button-cancel-${id}`}
              >
                <X className="h-3 w-3 mr-1" />
                Cancel
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}