import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Plus, Settings2 } from "lucide-react";
import { useState } from "react";

interface ResourceAllocationProps {
  availableCores: number;
  availableMemory: number;
  onCreateVGPU: (config: VGPUConfig) => void;
}

interface VGPUConfig {
  name: string;
  cores: number;
  memory: number;
  priority: "high" | "medium" | "low";
}

export default function ResourceAllocation({
  availableCores,
  availableMemory,
  onCreateVGPU,
}: ResourceAllocationProps) {
  const [name, setName] = useState("VGPU-New");
  const [cores, setCores] = useState([4]);
  const [memory, setMemory] = useState([8]);
  const [priority, setPriority] = useState<"high" | "medium" | "low">("medium");

  const handleCreate = () => {
    const config: VGPUConfig = {
      name,
      cores: cores[0],
      memory: memory[0],
      priority,
    };
    onCreateVGPU(config);
    // Reset form //todo: remove mock functionality
    setName("VGPU-New");
    setCores([4]);
    setMemory([8]);
    setPriority("medium");
  };

  const canCreate = cores[0] <= availableCores && memory[0] <= availableMemory;

  return (
    <Card data-testid="card-resource-allocation">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="h-5 w-5" />
          Create Virtual GPU
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Badge variant="outline" className="mb-2">Available Resources</Badge>
            <div className="text-sm space-y-1">
              <div>CPU Cores: <span className="font-mono">{availableCores}</span></div>
              <div>Memory: <span className="font-mono">{availableMemory}GB</span></div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <Label htmlFor="vgpu-name">Virtual GPU Name</Label>
            <Input
              id="vgpu-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter VGPU name"
              data-testid="input-vgpu-name"
            />
          </div>

          <div>
            <Label>CPU Cores: {cores[0]}</Label>
            <div className="mt-2">
              <Slider
                value={cores}
                onValueChange={setCores}
                max={availableCores}
                min={1}
                step={1}
                data-testid="slider-cores"
              />
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {cores[0]} of {availableCores} available cores
            </div>
          </div>

          <div>
            <Label>Memory: {memory[0]}GB</Label>
            <div className="mt-2">
              <Slider
                value={memory}
                onValueChange={setMemory}
                max={availableMemory}
                min={1}
                step={1}
                data-testid="slider-memory"
              />
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {memory[0]}GB of {availableMemory}GB available
            </div>
          </div>

          <div>
            <Label>Priority Level</Label>
            <Select value={priority} onValueChange={(value: "high" | "medium" | "low") => setPriority(value)}>
              <SelectTrigger data-testid="select-priority">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button
            onClick={handleCreate}
            disabled={!canCreate || !name.trim()}
            className="w-full"
            data-testid="button-create-vgpu"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Virtual GPU
          </Button>

          {!canCreate && (
            <div className="text-sm text-destructive">
              Insufficient resources available for this configuration.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}