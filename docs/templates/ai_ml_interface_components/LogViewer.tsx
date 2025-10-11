import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileText, Download, Filter } from "lucide-react";
import { useState } from "react";

interface LogEntry {
  id: string;
  timestamp: string;
  level: "info" | "warn" | "error" | "debug";
  source: string;
  message: string;
}

interface LogViewerProps {
  logs: LogEntry[];
  onExport: () => void;
}

export default function LogViewer({ logs, onExport }: LogViewerProps) {
  const [filterLevel, setFilterLevel] = useState<string>("all");
  const [filterSource, setFilterSource] = useState<string>("all");

  const sources = Array.from(new Set(logs.map(log => log.source)));

  const filteredLogs = logs.filter(log => {
    const levelMatch = filterLevel === "all" || log.level === filterLevel;
    const sourceMatch = filterSource === "all" || log.source === filterSource;
    return levelMatch && sourceMatch;
  });

  const getLevelColor = (level: string) => {
    switch (level) {
      case "error": return "bg-destructive";
      case "warn": return "bg-chart-2";
      case "info": return "bg-chart-1";
      case "debug": return "bg-muted";
      default: return "bg-muted";
    }
  };

  return (
    <Card data-testid="card-log-viewer">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            System Logs
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={onExport}
            data-testid="button-export-logs"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
        
        <div className="flex gap-2">
          <Select value={filterLevel} onValueChange={setFilterLevel}>
            <SelectTrigger className="w-32" data-testid="select-filter-level">
              <SelectValue placeholder="Level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="error">Error</SelectItem>
              <SelectItem value="warn">Warning</SelectItem>
              <SelectItem value="info">Info</SelectItem>
              <SelectItem value="debug">Debug</SelectItem>
            </SelectContent>
          </Select>

          <Select value={filterSource} onValueChange={setFilterSource}>
            <SelectTrigger className="w-40" data-testid="select-filter-source">
              <SelectValue placeholder="Source" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources</SelectItem>
              {sources.map(source => (
                <SelectItem key={source} value={source}>{source}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent>
        <ScrollArea className="h-80" data-testid="scroll-logs">
          <div className="space-y-2">
            {filteredLogs.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No logs found matching the current filters
              </div>
            ) : (
              filteredLogs.map((log) => (
                <div
                  key={log.id}
                  className="flex items-start gap-3 p-3 rounded border bg-card text-sm font-mono"
                  data-testid={`log-entry-${log.id}`}
                >
                  <Badge
                    className={`${getLevelColor(log.level)} border-0 text-white text-xs shrink-0`}
                  >
                    {log.level.toUpperCase()}
                  </Badge>
                  <div className="shrink-0 text-muted-foreground w-20">
                    {log.timestamp}
                  </div>
                  <div className="shrink-0 text-muted-foreground w-24">
                    {log.source}
                  </div>
                  <div className="flex-1 break-words">
                    {log.message}
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}