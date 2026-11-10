// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Three contributors in one command graph: each one's output views are the next one's inputs, and
// the CPU line is crossed only by your buffers going in and the result coming out.

const STEPS = [
  {name: 'GPUContiguityWeights', detail: "criterion: 'queen'"},
  {name: 'GPUSpatialWeightsTransform', detail: "operation: 'row'"},
  {name: 'GPULocalMoran', detail: 'local I and p-values'}
] as const;

/** What each step hands to the next (the views passed in local-moran.example.ts:61-68). */
const FLOWS = [
  ['queen.outputWeights', 'CSR weights'],
  ['rowWeights.outputWeights', 'row weights']
] as const;

const BOX = {width: 218, height: 62, top: 178};
const BOX_LEFT = [28, 326, 624];
const WIDTH = 870;

export function OneGraphFigure() {
  return (
    <div className="meta-scroll">
      <svg
        className="workflows-graph"
        viewBox={`0 0 ${WIDTH} 310`}
        style={{minWidth: 800}}
        role="img"
        aria-label="Your buffers enter one command graph; contiguity weights, the row transform and local Moran pass views to each other on the GPU; only the result is copied back to the CPU."
      >
        <defs>
          <marker
            id="workflows-arrow-accent"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" className="workflows-graph__head-accent" />
          </marker>
          <marker
            id="workflows-arrow-muted"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" className="workflows-graph__head-muted" />
          </marker>
          <marker
            id="workflows-arrow-signal"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" className="workflows-graph__head-signal" />
          </marker>
        </defs>

        {/* CPU side */}
        <text x="16" y="22" className="workflows-graph__lane">
          CPU
        </text>
        <rect x="28" y="34" width="216" height="48" rx="8" className="workflows-graph__cpu" />
        <text x="44" y="54" className="workflows-graph__title">
          Your buffers
        </text>
        <text x="44" y="71" className="workflows-graph__detail">
          tracts, poverty %, parameters
        </text>
        <rect x="625" y="34" width="216" height="48" rx="8" className="workflows-graph__readback" />
        <text x="641" y="54" className="workflows-graph__title">
          Read back
        </text>
        <text x="641" y="71" className="workflows-graph__detail">
          localI, adjustedPValues
        </text>

        <line x1="12" y1="104" x2={WIDTH - 12} y2="104" className="workflows-graph__boundary" />
        <text x={WIDTH / 2} y="98" textAnchor="middle" className="workflows-graph__detail">
          nothing crosses here between steps
        </text>
        <text x="16" y="124" className="workflows-graph__lane">
          GPU
        </text>

        {/* The graph */}
        <rect
          x="16"
          y="132"
          width={WIDTH - 32}
          height="164"
          rx="12"
          className="workflows-graph__frame"
        />
        <text x="32" y="156" className="workflows-graph__frame-title">
          one GPUCommandGraph
        </text>
        {STEPS.map((step, index) => (
          <g key={step.name}>
            <rect
              x={BOX_LEFT[index]}
              y={BOX.top}
              width={BOX.width}
              height={BOX.height}
              rx="8"
              className="workflows-graph__kernel"
            />
            <text
              x={BOX_LEFT[index] + 12}
              y={BOX.top + 26}
              className="workflows-graph__title workflows-graph__code"
            >
              {step.name}
            </text>
            <text
              x={BOX_LEFT[index] + 12}
              y={BOX.top + 45}
              className="workflows-graph__detail workflows-graph__code"
            >
              {step.detail}
            </text>
          </g>
        ))}
        {FLOWS.map(([view, label], index) => {
          const from = BOX_LEFT[index] + BOX.width;
          const to = BOX_LEFT[index + 1];
          const middle = (from + to) / 2;
          return (
            <g key={view}>
              <line
                x1={from + 4}
                y1={BOX.top + BOX.height / 2}
                x2={to - 4}
                y2={BOX.top + BOX.height / 2}
                className="workflows-graph__flow"
                markerEnd="url(#workflows-arrow-accent)"
              />
              <text
                x={middle}
                y={BOX.top + BOX.height / 2 - 10}
                textAnchor="middle"
                className="workflows-graph__flow-label"
              >
                {label}
              </text>
              <text
                x={middle}
                y={BOX.top + BOX.height / 2 + 22}
                textAnchor="middle"
                className="workflows-graph__detail"
              >
                view
              </text>
            </g>
          );
        })}
        <text x={WIDTH / 2} y="278" textAnchor="middle" className="workflows-graph__detail">
          compileAsync once, then encode and submit; each arrow is a view passed as a prop
        </text>

        {/* Crossings */}
        <line
          x1="137"
          y1="84"
          x2="137"
          y2={BOX.top - 4}
          className="workflows-graph__in"
          markerEnd="url(#workflows-arrow-muted)"
        />
        <text x="147" y="124" className="workflows-graph__detail">
          imported once
        </text>
        <line
          x1="733"
          y1={BOX.top - 2}
          x2="733"
          y2="86"
          className="workflows-graph__out"
          markerEnd="url(#workflows-arrow-signal)"
        />
        <text x="723" y="124" textAnchor="end" className="workflows-graph__detail">
          copied out at the end
        </text>
      </svg>
      <p className="meta-scroll-hint">Scroll sideways for the whole graph.</p>
    </div>
  );
}
