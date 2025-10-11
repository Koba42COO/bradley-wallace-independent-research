import VirtualGPUCard from '../VirtualGPUCard';

export default function VirtualGPUCardExample() {
  const handleToggle = (id: string) => {
    console.log(`Toggle virtual GPU ${id}`);
  };

  const handleConfigure = (id: string) => {
    console.log(`Configure virtual GPU ${id}`);
  };

  return (
    <div className="p-4 space-y-4">
      <VirtualGPUCard
        id="vgpu-001"
        name="VGPU-Compute-01"
        cores={8}
        utilization={78}
        status="active"
        memory={{ used: 12, total: 16 }}
        onToggle={handleToggle}
        onConfigure={handleConfigure}
      />
      <VirtualGPUCard
        id="vgpu-002"
        name="VGPU-ML-02"
        cores={16}
        utilization={34}
        status="idle"
        memory={{ used: 6, total: 32 }}
        onToggle={handleToggle}
        onConfigure={handleConfigure}
      />
      <VirtualGPUCard
        id="vgpu-003"
        name="VGPU-Render-03"
        cores={4}
        utilization={0}
        status="error"
        memory={{ used: 0, total: 8 }}
        onToggle={handleToggle}
        onConfigure={handleConfigure}
      />
    </div>
  );
}