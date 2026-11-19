var e={slug:`api/gpu-network/network-movement-benchmarks`,title:`Network and movement measurements`,html:`<h1 id="network-and-movement-measurements">Network and movement measurements</h1>
<p>One native Chromium WebGPU run on Apple Metal3. Times below are median wall-clock milliseconds for three warm encode/submit/queue-completion runs. Upload/preparation, compilation, cold execution and readback are recorded separately in the <a href="../../../modules/analysis/test/oracles/fixtures/network-movement-benchmark.json">JSON fixture</a>.</p>
<table>
<thead>
<tr>
<th>Operation</th>
<th align="right">Workload size</th>
<th align="right">Warm ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>flow</td>
<td align="right">8</td>
<td align="right">0.80</td>
</tr>
<tr>
<td>flow</td>
<td align="right">32</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>flow</td>
<td align="right">128</td>
<td align="right">1.30</td>
</tr>
<tr>
<td>louvain</td>
<td align="right">4</td>
<td align="right">0.60</td>
</tr>
<tr>
<td>louvain</td>
<td align="right">16</td>
<td align="right">5.00</td>
</tr>
<tr>
<td>louvain</td>
<td align="right">32</td>
<td align="right">28.90</td>
</tr>
<tr>
<td>time-routing</td>
<td align="right">8</td>
<td align="right">0.30</td>
</tr>
<tr>
<td>time-routing</td>
<td align="right">64</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>time-routing</td>
<td align="right">256</td>
<td align="right">0.40</td>
</tr>
<tr>
<td>point-routing</td>
<td align="right">8</td>
<td align="right">0.30</td>
</tr>
<tr>
<td>point-routing</td>
<td align="right">64</td>
<td align="right">0.60</td>
</tr>
<tr>
<td>point-routing</td>
<td align="right">128</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>facility</td>
<td align="right">4</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>facility</td>
<td align="right">8</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>facility</td>
<td align="right">16</td>
<td align="right">3.20</td>
</tr>
<tr>
<td>raam</td>
<td align="right">4</td>
<td align="right">1.20</td>
</tr>
<tr>
<td>raam</td>
<td align="right">16</td>
<td align="right">0.80</td>
</tr>
<tr>
<td>raam</td>
<td align="right">64</td>
<td align="right">2.50</td>
</tr>
<tr>
<td>kalman</td>
<td align="right">32</td>
<td align="right">0.60</td>
</tr>
<tr>
<td>kalman</td>
<td align="right">512</td>
<td align="right">0.30</td>
</tr>
<tr>
<td>kalman</td>
<td align="right">4096</td>
<td align="right">1.00</td>
</tr>
<tr>
<td>restriction</td>
<td align="right">32</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>restriction</td>
<td align="right">512</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>restriction</td>
<td align="right">4096</td>
<td align="right">0.70</td>
</tr>
<tr>
<td>similarity</td>
<td align="right">64</td>
<td align="right">0.80</td>
</tr>
<tr>
<td>similarity (95% pruned)</td>
<td align="right">64</td>
<td align="right">0.80</td>
</tr>
<tr>
<td>similarity</td>
<td align="right">1024</td>
<td align="right">2.00</td>
</tr>
<tr>
<td>similarity (95% pruned)</td>
<td align="right">1024</td>
<td align="right">0.80</td>
</tr>
<tr>
<td>similarity</td>
<td align="right">4096</td>
<td align="right">5.20</td>
</tr>
<tr>
<td>similarity (95% pruned)</td>
<td align="right">4096</td>
<td align="right">1.10</td>
</tr>
</tbody>
</table>
<p>Run <code>yarn test-benchmarks modules/analysis/test/gpu-network/network-optimization/network-movement-benchmark.spec.ts</code> after selecting the repository Node version. The launcher enables benchmark flags and forwards the selected file.</p>
<p>These fixtures use chain flow/routes, disconnected dense community blocks, p-dispersion over a distance grid, balanced accessibility, and 16-observation tracks. They establish a measured bounded envelope, not worst-case complexity, end-to-end production scale or CPU acceleration. Louvain objective reevaluation and exhaustive facility enumeration remain expensive. The similarity comparison changes survivor selectivity; it does not compare equivalent workloads to a CPU baseline.</p>
<p>The source records uploaded/output bytes; transient graph allocation is not included in those counts. Similarity dynamic-program allocation is bounded explicitly by <code>survivorCapacity</code>. Device/driver breadth, longer single-track latency and large sparse graph solvers remain separate performance work.</p>`};export{e as default};