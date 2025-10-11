import ResourceAllocation from '../ResourceAllocation';

export default function ResourceAllocationExample() {
  const handleCreateVGPU = (config: any) => {
    console.log('Creating VGPU with config:', config);
  };

  return (
    <div className="p-4 max-w-md">
      <ResourceAllocation
        availableCores={24}
        availableMemory={64}
        onCreateVGPU={handleCreateVGPU}
      />
    </div>
  );
}