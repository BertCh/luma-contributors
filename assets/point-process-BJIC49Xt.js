var e={slug:`api/gpu-network/point-process`,title:`Network point processes`,html:`<h1 id="network-point-processes">Network point processes</h1>
<p><code>GPUNetworkKernelDensity</code> estimates network intensity from supplied shortest-path distances and edge-length quadrature. <code>GPUNetworkInhomogeneousK</code> computes a distinct intensity-reweighted K statistic with optional Ang shell correction. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/point-process">the point-process contracts</a> for formulas, bounds, and status semantics.</p>
<p>These executable scenarios use a three-node chain with two endpoint events. Each result is a GPU readback. The intensity values and shortest-path tables are explicit inputs.</p>
<h2 id="gpunetworkkerneldensity"><code>GPUNetworkKernelDensity</code></h2>
<p>{/* lens:GPUNetworkKernelDensity */}</p>
<p>::lens{scenario=gpu-network.network-kernel-density rung=result caption="Inspect the three-node network result"}</p>
<p>::lens{scenario=gpu-network.network-kernel-density rung=graph caption="Inspect the executed network computation"}</p>
<h2 id="gpunetworkinhomogeneousk"><code>GPUNetworkInhomogeneousK</code></h2>
<p>{/* lens:GPUNetworkInhomogeneousK */}</p>
<p>::lens{scenario=gpu-network.network-inhomogeneous-k rung=result caption="Inspect the three-node network result"}</p>
<p>::lens{scenario=gpu-network.network-inhomogeneous-k rung=graph caption="Inspect the executed network computation"}</p>`};export{e as default};