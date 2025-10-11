#!/bin/bash
#SBATCH --job-name=wallace_10billion
#SBATCH --nodes=1000
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024GB  # 1TB per node
#SBATCH --time=219:00:00
#SBATCH --partition=compute
#SBATCH --output=wallace_10billion_%j.out

echo "🚀 Starting 10^10 Prime Analysis"
echo "Target: 10,000,000,000 primes"
echo "Cluster: 1000 nodes, 64,000 CPUs"
echo "Memory: 3.3 TB required"
echo "Estimated time: 9.2 days"

# Load modules
module load python/3.9
module load openmpi/4.1

# Run distributed analysis
srun --mpi=pmix python3 distributed_prime_analysis.py --scale 10billion

echo "✅ Analysis complete"
