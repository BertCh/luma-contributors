// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The interpreter only inventories storage declarations. Preserve the uniform words and layout,
// changing their declaration for interpretation alone (production WGSL is unchanged).
export function adaptMultipartNearestUniforms(nodes: readonly unknown[]): unknown[] {
  type Node = {
    compile(context: unknown): {encode(context: Record<string, unknown>): void; destroy?(): void};
  };
  type Pipeline = {source: string; constants: Record<string, number>};
  type Pass = {setPipeline(pipeline: Pipeline): void};
  return nodes.map(value => {
    const node = value as Node;
    return {
      ...node,
      compile(context: unknown) {
        // The shared executor's NullBuffer shim retains later writes, but not createBuffer(data).
        // Preserve initial uniform uploads here, without changing any computed shader values.
        type BufferDevice = {
          createBuffer(props: {data?: ArrayBufferView}): {write(data: ArrayBufferView): void};
        };
        const device = (context as {device: BufferDevice}).device;
        const createBuffer = device.createBuffer;
        device.createBuffer = function (props) {
          const buffer = createBuffer.call(this, props);
          if (props.data) buffer.write(props.data);
          return buffer;
        };
        let compiled: ReturnType<Node['compile']>;
        try {
          compiled = node.compile(context);
        } finally {
          device.createBuffer = createBuffer;
        }
        return {
          ...compiled,
          encode(encoding: Record<string, unknown>) {
            const pass = encoding['computePass'] as Pass;
            const proxy = new Proxy(pass, {
              get(target, key) {
                if (key === 'setPipeline')
                  return (pipeline: Pipeline) =>
                    target.setPipeline({
                      ...pipeline,
                      source: pipeline.source
                        .replace(
                          'var<uniform> graphChunkGroupParameters: GraphChunkGroupParameters',
                          'var<storage, read> graphChunkGroupParameters: array<u32>'
                        )
                        .replace(
                          'graphChunkGroupParameters.words[word / 4u][word % 4u]',
                          'graphChunkGroupParameters[word]'
                        )
                        .replace(
                          /var<uniform> selectionSettings: array<vec4<u32>, \d+>/,
                          'var<storage, read> selectionSettings: array<u32>'
                        )
                        .replace(
                          /selectionSettings\[(\d+)\]\.([xyzw])/g,
                          (_, row, component) =>
                            `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
                        )
                        .replace(
                          /var<uniform> chunkSettings: array<vec4<u32>, \d+>/,
                          'var<storage, read> chunkSettings: array<u32>'
                        )
                        .replace(
                          /chunkSettings\[(\d+)\]\[(\d+)\]/g,
                          (_, row, component) =>
                            `chunkSettings[${Number(row) * 4 + Number(component)}]`
                        )
                    });
                if (key === 'setBindings')
                  return (bindings: Record<string, unknown>) => {
                    bindings = Object.fromEntries(
                      Object.entries(bindings).map(([name, value]) => {
                        const buffer = value as {backing?: ArrayBuffer; byteLength: number};
                        return [
                          name,
                          buffer.backing ? {buffer, offset: 0, size: buffer.byteLength} : value
                        ];
                      })
                    );
                    return (Reflect.get(target, key) as (value: unknown) => void)(bindings);
                  };
                return Reflect.get(target, key);
              }
            });
            compiled.encode({...encoding, computePass: proxy});
          }
        };
      }
    };
  });
}
