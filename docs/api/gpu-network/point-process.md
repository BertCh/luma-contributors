# Network point processes

Import from `@luma-contributors/analysis/gpu-network`.

`GPUNetworkKernelDensity` estimates network intensity from supplied shortest-path distances and edge-length quadrature. `GPUNetworkInhomogeneousK` computes a distinct intensity-reweighted K statistic with optional Ang shell correction. See [the point-process contracts](../gpu-spatial-analysis/point-process.md) for formulas, bounds, and status semantics.

These executable scenarios use a three-node chain with two endpoint events. Each result is a GPU readback. The intensity values and shortest-path tables are explicit inputs.

## `GPUNetworkKernelDensity`

Provide one `uint32` word per query in `output.validity`. The contributor writes 1 for a
usable intensity and 0 when invalid data, an unsupported event, or incomplete matrix evidence
prevents the estimate. Request the one-row `uint32` destination `output.status.invalidCount`
to count zero-validity rows on the GPU. Omit `output.status` when neither diagnostic is requested.

A `GPUNetworkCostMatrix` stopped at `maximumIterations`, or built with a `costLimit` below
`bandwidth`, leaves `+Infinity` inside the kernel support, which reads as disconnected. Pass the
matrix `status.converged` as `costsComplete` and its `costLimit` view as `costLimit`.
Without a certificate destination, failed supplied evidence writes NaN to intensity and mass
and sets every query's `output.validity` to 0. Request the one-row `uint32` destination
`output.status.complete` to retain incumbent intensity and mass. The certificate is 1 only when
the matrix is complete and `bandwidth`
does not exceed its supplied horizon. Inspect that certificate before consuming incumbents as
complete estimates. A NaN horizon fails the comparison; an omitted horizon makes the caller
responsible for supplying distances over the entire kernel support.

{/* lens:GPUNetworkKernelDensity */}

::lens{scenario=gpu-network.network-kernel-density rung=result caption="Inspect the three-node network result"}

::lens{scenario=gpu-network.network-kernel-density rung=graph caption="Inspect the executed network computation"}

## `GPUNetworkInhomogeneousK`

This is the intensity-reweighted network K statistic, with arc-length distance inputs,
per-radius invalidity bits and explicit pair work limits. See
[Inputs, outputs and limits](../gpu-spatial-analysis/point-process.md#inhomogeneous-network-k).
It accepts at most 1,024 events and 256 radii. The separate homogeneous `GPUNetworkKFunction`
accepts at most 92,682 events (`GPU_NETWORK_K_FUNCTION_MAXIMUM_EVENT_COUNT`) so the possible
unordered-pair count fits uint32, and 2–1,024 bands. These are construction caps, not throughput
claims. Density estimates and the two K statistics have distinct normalization contracts.

{/* lens:GPUNetworkInhomogeneousK */}

::lens{scenario=gpu-network.network-inhomogeneous-k rung=result caption="Inspect the three-node network result"}

::lens{scenario=gpu-network.network-inhomogeneous-k rung=graph caption="Inspect the executed network computation"}
