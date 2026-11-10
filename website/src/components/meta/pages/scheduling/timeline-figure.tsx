// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Once, then every frame: construct and compile happen once on the CPU; each frame encodes the fixed
// order and the GPU runs the same kernels; a parameter write between frames changes the output
// without compiling again.

import type {LensViewModel} from '../../../../lens/types';

export type TimelineStep = 'construct' | 'compile' | 'encode' | 'write';

const WIDTH = 820;
const HEIGHT = 222;
const LANE_X = 78;
const CPU_Y = 46;
const GPU_Y = 112;
const OUTPUT_Y = 182;
const BOX_HEIGHT = 34;

type Frame = {x: number; label: string; output: 'first' | 'second'};

const FRAMES: readonly Frame[] = [
  {x: 318, label: 'frame 1', output: 'first'},
  {x: 426, label: 'frame 2', output: 'first'},
  {x: 616, label: 'frame 3', output: 'second'}
];
const FRAME_WIDTH = 100;
const WRITE_X = 536;
const WRITE_WIDTH = 70;

export function TimelineFigure({
  model,
  active
}: {
  /** The recorded run whose compute kernels each frame dispatches. */
  model: LensViewModel;
  active: TimelineStep;
}) {
  const kernelCount = model.nodes.filter(node => node.kind === 'compute').length;
  const parameterBytes = model.bufferBytes?.['kde-parameters'];
  const isActive = (step: TimelineStep) => (active === step ? 'true' : undefined);
  const tickWidth = (FRAME_WIDTH - 12) / Math.max(1, kernelCount);
  return (
    <div className="meta-panel">
      <div className="meta-scroll">
        <svg
          className="scheduling-timeline"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          style={{minWidth: 760}}
          role="img"
          aria-label={`Construct and compile run once. Each of three frames encodes the same ${kernelCount} kernels and submits them. A parameter write between frame 2 and frame 3 changes the output with no compile.`}
        >
          <text className="scheduling-timeline__lane" x={8} y={CPU_Y + 21}>
            CPU
          </text>
          <text className="scheduling-timeline__lane" x={8} y={GPU_Y + 21}>
            GPU
          </text>
          <text className="scheduling-timeline__lane" x={8} y={OUTPUT_Y + 16}>
            output
          </text>
          <line
            className="scheduling-timeline__axis"
            x1={LANE_X}
            x2={WIDTH - 10}
            y1={GPU_Y - 14}
            y2={GPU_Y - 14}
          />
          <line
            className="scheduling-timeline__axis"
            x1={LANE_X}
            x2={WIDTH - 10}
            y1={OUTPUT_Y - 12}
            y2={OUTPUT_Y - 12}
          />

          <g className="scheduling-timeline__once">
            <text className="scheduling-timeline__bracket" x={LANE_X} y={20}>
              once
            </text>
            <path className="scheduling-timeline__bracket-line" d={`M${LANE_X} 28 H${306}`} />
            <text className="scheduling-timeline__bracket" x={318} y={20}>
              every frame
            </text>
            <path className="scheduling-timeline__bracket-line" d={`M318 28 H${WIDTH - 10}`} />
          </g>

          <g
            className="scheduling-timeline__box"
            data-side="cpu"
            data-active={isActive('construct')}
          >
            <rect x={LANE_X} y={CPU_Y} width={106} height={BOX_HEIGHT} rx={6} />
            <text x={LANE_X + 10} y={CPU_Y + 15}>
              construct
            </text>
            <text className="scheduling-timeline__meta" x={LANE_X + 10} y={CPU_Y + 28}>
              validate props
            </text>
          </g>
          <g className="scheduling-timeline__box" data-side="cpu" data-active={isActive('compile')}>
            <rect x={192} y={CPU_Y} width={116} height={BOX_HEIGHT} rx={6} />
            <text x={202} y={CPU_Y + 15}>
              compile
            </text>
            <text className="scheduling-timeline__meta" x={202} y={CPU_Y + 28}>
              order, pipelines
            </text>
          </g>

          {FRAMES.map(frame => (
            <g key={frame.label}>
              <text className="scheduling-timeline__frame" x={frame.x} y={CPU_Y - 6}>
                {frame.label}
              </text>
              <g
                className="scheduling-timeline__box"
                data-side="cpu"
                data-active={isActive('encode')}
              >
                <rect x={frame.x} y={CPU_Y} width={FRAME_WIDTH - 8} height={BOX_HEIGHT} rx={6} />
                <text x={frame.x + 8} y={CPU_Y + 15}>
                  encode
                </text>
                <text className="scheduling-timeline__meta" x={frame.x + 8} y={CPU_Y + 28}>
                  + submit
                </text>
              </g>
              <g className="scheduling-timeline__kernels" data-active={isActive('encode')}>
                {Array.from({length: kernelCount}, (_, index) => (
                  <rect
                    key={index}
                    x={frame.x + 6 + index * tickWidth}
                    y={GPU_Y}
                    width={Math.max(1, tickWidth - 1.5)}
                    height={BOX_HEIGHT}
                    rx={1.5}
                  />
                ))}
              </g>
              <g
                className="scheduling-timeline__output"
                data-output={frame.output}
                data-active={frame.output === 'second' ? isActive('write') : undefined}
              >
                <rect x={frame.x} y={OUTPUT_Y} width={FRAME_WIDTH - 8} height={24} rx={4} />
                <text x={frame.x + 8} y={OUTPUT_Y + 16}>
                  {frame.output === 'first' ? 'old result' : 'new result'}
                </text>
              </g>
            </g>
          ))}
          <text
            className="scheduling-timeline__meta"
            x={FRAMES[0].x + 6}
            y={GPU_Y + BOX_HEIGHT + 14}
          >
            {kernelCount} kernels, same order
          </text>

          <g className="scheduling-timeline__write" data-active={isActive('write')}>
            <rect x={WRITE_X} y={CPU_Y} width={WRITE_WIDTH} height={BOX_HEIGHT} rx={6} />
            <text x={WRITE_X + 8} y={CPU_Y + 15}>
              write
            </text>
            <text className="scheduling-timeline__meta" x={WRITE_X + 8} y={CPU_Y + 28}>
              {parameterBytes ? `${parameterBytes} bytes` : 'parameters'}
            </text>
            <path
              className="scheduling-timeline__write-arrow"
              d={`M${WRITE_X + WRITE_WIDTH / 2} ${CPU_Y + BOX_HEIGHT} V${GPU_Y + 4}`}
            />
            <text
              className="scheduling-timeline__meta"
              x={WRITE_X + WRITE_WIDTH / 2}
              y={GPU_Y + 22}
              textAnchor="middle"
            >
              no compile
            </text>
          </g>
          <text
            className="scheduling-timeline__meta"
            x={WIDTH - 20}
            y={GPU_Y + 22}
            textAnchor="end"
          >
            …
          </text>
        </svg>
      </div>
      <p className="meta-scroll-hint">Scroll sideways to follow the frames.</p>
    </div>
  );
}
