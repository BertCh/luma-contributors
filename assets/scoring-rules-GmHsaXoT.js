var e={slug:`api/gpu-spatial-analysis/scoring-rules`,title:`Proper scoring rules`,html:`<h1 id="proper-scoring-rules">Proper scoring rules</h1>
<p>Scores of probabilistic forecasts against observed values: the continuous ranked probability score
(CRPS), the energy score and the variogram score, for ensembles such as conditional Gaussian
simulations, and the closed-form Gaussian CRPS for kriging predictions. Lower is better for all of
them, and each is a proper score: in expectation it is smallest for the true distribution.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUEnsembleScores, GPUGaussianPredictiveScores} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span></code></pre>
<p>The classes are in the directory <code>src/gpu-spatial-analysis/scoring-rules/</code>. The re-export line
<code>export * from './scoring-rules/index';</code> in <code>src/gpu-spatial-analysis/index.ts</code> is what makes the
import above resolve; until it is added, import from <code>./scoring-rules/index</code>.</p>
<h2 id="definitions">Definitions</h2>
<p>Definitions are those of scoringRules 1.1.3 (Jordan, Krueger and Lerch, Journal of Statistical
Software 90(12), 2019). With members <code>x_1 ... x_m</code> and observation <code>y</code>:</p>
<table>
<thead>
<tr>
<th>Score</th>
<th>scoringRules function</th>
<th>Formula</th>
</tr>
</thead>
<tbody>
<tr>
<td>CRPS</td>
<td><code>crps_sample(method = "edf")</code></td>
<td><code>(1/m) sum_i abs(x_i - y) - 1/(2 m^2) sum_i sum_j abs(x_i - x_j)</code></td>
</tr>
<tr>
<td>Energy score</td>
<td><code>es_sample</code> (Gneiting et al. 2008)</td>
<td>the same with Euclidean norms of the <code>d</code>-vectors</td>
</tr>
<tr>
<td>Variogram score of order <code>p</code></td>
<td><code>vs_sample(p = ...)</code> (Scheuerer and Hamill 2015)</td>
<td><code>sum_(i,j) w_ij (abs(y_i - y_j)^p - (1/m) sum_k abs(x_ki - x_kj)^p)^2</code></td>
</tr>
<tr>
<td>Gaussian CRPS</td>
<td><code>crps_norm</code> (Gneiting et al. 2005)</td>
<td><code>sd (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))</code>, <code>z = (y - mean) / sd</code></td>
</tr>
</tbody>
</table>
<h3 id="fair-and-non-fair-ensemble-variants">Fair and non-fair ensemble variants</h3>
<p>scoringRules 1.1.3 defines only the non-fair forms above, the empirical-distribution scores whose
member-pair term is divided by <code>2 m^2</code>. They are the default (<code>fair</code> false). They are biased
upward for a finite ensemble: even a calibrated ensemble of <code>m</code> draws scores worse, by
<code>E|X - X'| / (2m)</code>, than its distribution would.</p>
<p>With <code>fair: true</code> the pair term of the CRPS and the energy score is divided by <code>2 m (m - 1)</code>
instead (Ferro 2014; Zamo and Naveau 2018). That removes the bias and lets scores of ensembles with
different member counts be compared. The fair variant is an extension of this package: scoringRules
1.1.3 does not define it, so no scoringRules value exists for it. It needs <code>memberCount >= 2</code>. The
variogram score has no fair variant here.</p>
<h2 id="gpuensemblescores">GPUEnsembleScores</h2>
<p>Variogram pair count is at most 2,147,483,647 to prevent flattened-index overflow; all score inputs
and outputs remain device-binding bounded.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> scores</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUEnsembleScores</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ensemble,        </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, memberCount * dimension rows, member-major</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  observations,    </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, dimension rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  memberCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dimension: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4096</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  fair: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variogramOrder: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variogramPairs,  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32x2'>: ordered site pairs [i, j]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variogramWeights, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional GraphDataView&#x3C;'float32'>, one weight per pair</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {crps, energyScore, variogramScore}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(scores);</span></span></code></pre>
<ul>
<li><code>ensemble</code> is member-major, row <code>member * dimension + site</code>. That is the layout of
<code>GPUConditionalGaussianSimulation</code> <code>output.values</code>, so a simulation batch is scored in place.</li>
<li><code>output.crps</code> has one row per site. <code>output.energyScore</code> and <code>output.variogramScore</code> have one row
each and score the whole field. At least one output is required.</li>
<li><code>variogramPairs</code> lists ordered pairs; every listed pair contributes one term. Listing every
<code>[i, j]</code> with <code>i != j</code> reproduces <code>vs_sample</code> with its default unit weights, which sums the full
symmetric matrix. A list of neighbouring pairs restricts the score to them: the all-pairs form
costs <code>O(d^2 m)</code>, so for large fields use neighbour pairs, a documented choice that matches the
<code>w_vs</code> weights argument of <code>vs_sample</code>.</li>
<li>Members, observations or pairs that are NaN give NaN scores; a pair index at or beyond <code>dimension</code>
gives a NaN variogram score.</li>
</ul>
<h3 id="how-it-is-computed">How it is computed</h3>
<p>The CRPS runs one 64-lane workgroup per site. Lane <code>l</code> adds the terms of members <code>l</code>, <code>l + 64</code>, ...
in order, then a fixed binary tree adds the lanes, so the f32 sum does not depend on scheduling.
The energy and variogram scores write one term per member or per pair and the published
<code>GPUReduction</code> sums them. There are no float atomics. The result is repeatable on one device; the
published reduction may use subgroup collectives where the device offers them, and their order is
the device's.</p>
<p>Cost: <code>O(m^2 d)</code> for the CRPS and energy score, <code>O(P m)</code> for the variogram score over <code>P</code> pairs.
<code>memberCount</code> is at most 65535, <code>dimension</code> at most 67108863 (<code>floor((2^32 - 1) / 64)</code>), and the
ensemble must fit one storage binding. Tile larger fields by calling the class on subsets of sites
for the CRPS; the energy and variogram scores are scores of the whole field.</p>
<p>Accuracy: f32. The CRPS and variogram scores agree with a binary64 evaluation to about
<code>2e-5</code> relative on the test fixtures; the CRPS is a difference of two means of similar size when
the ensemble is wide relative to the error, so its absolute error scales with the spread.</p>
<p>{/* lens:GPUEnsembleScores */}</p>
<h2 id="gpugaussianpredictivescores">GPUGaussianPredictiveScores</h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> gaussianScores</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGaussianPredictiveScores</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  means: prediction,      </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// kriging prediction</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variances: variance,    </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// kriging variance, squared value units</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  observations,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {crps}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>The closed-form CRPS of <code>N(mean, variance)</code> per site, for <code>GPUKriging</code>, <code>GPULocalKriging</code> and
<code>GPUConditionalGaussianSimulation</code> <code>output.moments</code> (split into means and variances). Use the
leave-one-out prediction and variance of <code>GPUKriging</code> with the sample values to score kriging by
cross-validation. A zero variance is a point forecast and scores the absolute error; a negative or
NaN variance scores NaN.</p>
<p><code>2 Phi(z) - 1 = erf(z / sqrt(2))</code> has no WGSL built-in. The kernel sums the all-positive series
<code>erf(x) = 2/sqrt(pi) exp(-x^2) sum_n 2^n x^(2n+1) / (2n + 1)!!</code> for <code>x &#x3C;= 4</code> (no cancellation in
f32) and takes 1 beyond, which truncates <code>erf</code> by less than <code>1.6e-8</code> (<code>|z|</code> above 5.7). The
agreement with <code>math.erf</code> on the fixture is within <code>2e-6</code>.
The leading term is evaluated as <code>|error| * erf(...)</code> and only the bounded correction is scaled
by the deviation. For <code>|error| > 4 sqrt(2) * deviation</code>, the kernel uses
<code>|error| - deviation / sqrt(pi)</code> before dividing, so an overflowing standardized error does not
turn a finite representable score into infinity.</p>
<p>A Gaussian ensemble scored with <code>GPUEnsembleScores</code> converges to this score as <code>m</code> grows: the test
suite checks a 200-member quantile ensemble of <code>N(0, 1)</code> against <code>crps_norm(1; 0, 1)</code> within
<code>2e-3</code>.</p>
<p>{/* lens:GPUGaussianPredictiveScores */}</p>
<h2 id="verification">Verification</h2>
<ul>
<li>Oracle <code>test/oracles/scoring-rules.py</code> writes <code>fixtures/scoring-rules.json</code>: binary64
evaluations of the definitions above, with the CRPS cross-checked against the exact integral of
<code>(F_m - 1{y &#x3C;= x})^2</code> and the Gaussian CRPS against numerical integration.</li>
<li>Oracle <code>test/oracles/scoring-rules-fair.py</code> (NumPy and SciPy) writes
<code>fixtures/scoring-rules-fair.json</code> for the fair variants, which scoringRules does not define: the
fair CRPS by order statistics, the fair energy score as a mean over distinct pairs
(<code>scipy.spatial.distance.pdist</code>), and the closed-form CRPS and energy score of the standard normal
distribution, which the mean of the fair scores over many small ensembles must equal (and the
non-fair mean must exceed). Spec <code>gpu-ensemble-scores-fair.spec.ts</code>.</li>
<li><strong>The external scoringRules fixture is pending.</strong> scoringRules is an R package and R was not
installed where the fixture was made. A sibling script that calls <code>crps_sample</code>, <code>es_sample</code>,
<code>vs_sample</code> and <code>crps_norm</code> on the same inputs should replace the self-computed values.</li>
<li>Specs in <code>test/gpu-spatial-analysis/scoring-rules/</code>: GPU values against the fixture, hand values
(members 0, 1, 2 with <code>y = 0</code> score <code>5/9</code>, fair <code>1/3</code>), bitwise repeatability, propriety of the
CRPS on simulated data, kernel limits and prop validation.</li>
</ul>`};export{e as default};