import ComputeJobCard from '../ComputeJobCard';

export default function ComputeJobCardExample() {
  const handleAction = (id: string, action: "start" | "pause" | "cancel") => {
    console.log(`${action} job ${id}`);
  };

  return (
    <div className="p-4 space-y-4">
      <ComputeJobCard
        id="job-001"
        name="Matrix Multiplication Task"
        status="running"
        progress={67}
        duration="05:23"
        vgpuId="VGPU-Compute-01"
        priority="high"
        onAction={handleAction}
      />
      <ComputeJobCard
        id="job-002"
        name="Neural Network Training"
        status="queued"
        progress={0}
        duration="00:00"
        vgpuId="VGPU-ML-02"
        priority="medium"
        onAction={handleAction}
      />
      <ComputeJobCard
        id="job-003"
        name="Vector Operations Batch"
        status="completed"
        progress={100}
        duration="12:45"
        vgpuId="VGPU-Compute-01"
        priority="low"
        onAction={handleAction}
      />
      <ComputeJobCard
        id="job-004"
        name="Parallel Sort Algorithm"
        status="failed"
        progress={34}
        duration="02:11"
        vgpuId="VGPU-Render-03"
        priority="medium"
        onAction={handleAction}
      />
    </div>
  );
}