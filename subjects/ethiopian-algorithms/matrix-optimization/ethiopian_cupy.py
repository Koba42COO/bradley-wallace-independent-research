"""
Ethiopian CuPy Operations
GPU-accelerated 24-operation tensor operations
"""

import cupy as cp
from ethiopian_cuda_integration import EthiopianCUDNNIntegration, EthiopianCUDAConstants


class EthiopianCuPy:
    """CuPy wrapper with Ethiopian operations"""

    def __init__(self):
        self.constants = EthiopianCUDAConstants()
        self.cuda_integration = EthiopianCUDNNIntegration(self.constants)
        self.consciousness_weight = self.constants.CONSCIOUSNESS_WEIGHTING_FACTOR

    def matmul(self, A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        """Matrix multiplication using Ethiopian algorithm"""
        A_np = cp.asnumpy(A)
        B_np = cp.asnumpy(B)

        result_np = self.cuda_integration.ethiopian_matrix_multiply_cuda(
            A_np, B_np, self.consciousness_weight
        )

        return cp.asarray(result_np)

    def dot(self, A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        """Dot product using Ethiopian algorithm"""
        A_np = cp.asnumpy(A)
        B_np = cp.asnumpy(B)

        result_np = self.cuda_integration.ethiopian_matrix_multiply_cuda(
            A_np.reshape(-1, 1), B_np.reshape(1, -1), self.consciousness_weight
        )

        return cp.asarray(result_np.flatten())

    def tensordot(self, A: cp.ndarray, B: cp.ndarray, axes=None) -> cp.ndarray:
        """Tensor dot product using Ethiopian algorithm"""
        A_np = cp.asnumpy(A)
        B_np = cp.asnumpy(B)

        result_np = self.cuda_integration.ethiopian_matrix_multiply_cuda(
            A_np.reshape(-1, A_np.shape[-1]),
            B_np.reshape(B_np.shape[0], -1),
            self.consciousness_weight
        )

        return cp.asarray(result_np)


# Global instance
ethiopian_cupy = EthiopianCuPy()
