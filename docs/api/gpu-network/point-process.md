# Network point processes

`GPUNetworkKernelDensity` estimates network intensity from supplied shortest-path distances and edge-length quadrature. `GPUNetworkInhomogeneousK` computes a distinct intensity-reweighted K statistic with optional Ang shell correction. See [the point-process contracts](../gpu-spatial-analysis/point-process.md) for formulas, bounds, and status semantics.

These executable scenarios use a three-node chain with two endpoint events. Each result is a GPU readback. The intensity values and shortest-path tables are explicit inputs.

## `GPUNetworkKernelDensity`

{/* lens:GPUNetworkKernelDensity */}

::lens{scenario=gpu-network.network-kernel-density rung=result caption="Inspect the three-node network result"}

::lens{scenario=gpu-network.network-kernel-density rung=graph caption="Inspect the executed network computation"}

## `GPUNetworkInhomogeneousK`

{/* lens:GPUNetworkInhomogeneousK */}

::lens{scenario=gpu-network.network-inhomogeneous-k rung=result caption="Inspect the three-node network result"}

::lens{scenario=gpu-network.network-inhomogeneous-k rung=graph caption="Inspect the executed network computation"}
