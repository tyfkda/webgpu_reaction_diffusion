const WORKGROUP_SIZE = 8
const GRID_SIZE = 256

const kSimulationShaderCode = `
  struct CellState {
    a: f32,
    b: f32,
  };
  struct Uniform {
    grid: vec2f,
    dA: f32,
    dB: f32,
    feed: f32,
    kill: f32,
  };

  @group(0) @binding(0) var<uniform> param: Uniform;
  @group(0) @binding(1) var<storage> cellStateIn: array<CellState>;
  @group(0) @binding(2) var<storage, read_write> cellStateOut: array<CellState>;

  fn cellIndex(x: u32, y: u32) -> u32 {
    return (y % u32(param.grid.y)) * u32(param.grid.x) + (x % u32(param.grid.x));
  }

  fn laplaceA(x: u32, y: u32) -> f32 {
    let h = u32(param.grid.y);
    let w = u32(param.grid.x);
    let x0 = (x - 1 + w) % w;
    let y0 = (y - 1 + h) % h;
    let x1 = x;
    let y1 = y;
    let x2 = (x + 1) % w;
    let y2 = (y + 1) % h;
    return (cellStateIn[cellIndex(x0, y0)].a * 0.05 + cellStateIn[cellIndex(x1, y0)].a * 0.2 + cellStateIn[cellIndex(x2, y0)].a * 0.05 +
            cellStateIn[cellIndex(x0, y1)].a * 0.2  + cellStateIn[cellIndex(x1, y1)].a * -1  + cellStateIn[cellIndex(x2, y1)].a * 0.2 +
            cellStateIn[cellIndex(x0, y2)].a * 0.05 + cellStateIn[cellIndex(x1, y2)].a * 0.2 + cellStateIn[cellIndex(x2, y2)].a * 0.05);
  }

  fn laplaceB(x: u32, y: u32) -> f32 {
    let h = u32(param.grid.y);
    let w = u32(param.grid.x);
    let x0 = (x - 1 + w) % w;
    let y0 = (y - 1 + h) % h;
    let x1 = x;
    let y1 = y;
    let x2 = (x + 1) % w;
    let y2 = (y + 1) % h;
    return (cellStateIn[cellIndex(x0, y0)].b * 0.05 + cellStateIn[cellIndex(x1, y0)].b * 0.2 + cellStateIn[cellIndex(x2, y0)].b * 0.05 +
            cellStateIn[cellIndex(x0, y1)].b * 0.2  + cellStateIn[cellIndex(x1, y1)].b * -1  + cellStateIn[cellIndex(x2, y1)].b * 0.2 +
            cellStateIn[cellIndex(x0, y2)].b * 0.05 + cellStateIn[cellIndex(x1, y2)].b * 0.2 + cellStateIn[cellIndex(x2, y2)].b * 0.05);
  }

  @compute
  @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
  fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
    let dA = param.dA;
    let dB = param.dB;
    let feed = param.feed;
    let kill = param.kill;

    let i = cellIndex(cell.x, cell.y);
    let a = cellStateIn[i].a;
    let b = cellStateIn[i].b;
    cellStateOut[i].a = a + (dA * laplaceA(cell.x, cell.y)) - (a * b * b) + feed * (1 - a);
    cellStateOut[i].b = b + (dB * laplaceB(cell.x, cell.y)) + (a * b * b) - (kill + feed) * b;
  }
`

const kCellShaderCode = `
  struct VertexInput {
    @location(0) pos: vec2f,
    @builtin(instance_index) instance: u32,
  };

  struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) cell: vec2f,
  };

  struct CellState {
    a: f32,
    b: f32,
  };

  struct Uniform {
    grid: vec2f,
    dA: f32,
    dB: f32,
    feed: f32,
    k: f32,
  };

  @group(0) @binding(0) var<uniform> param: Uniform;
  @group(0) @binding(1) var<storage> cellState: array<CellState>;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    let i = f32(input.instance);

    let cell = vec2f(i % param.grid.x, floor(i / param.grid.x));
    let state = cellState[input.instance].b * 2.0;  // サイズ拡大

    let cellOffset = cell / param.grid * 2;
    let gridPos = (input.pos * state + 1) / param.grid - 1 + cellOffset;

    var output: VertexOutput;
    output.pos = vec4f(gridPos, 0, 1);
    output.cell = cell;
    return output;
  }

  struct FragInput {
    @location(0) cell: vec2f,
  };

  @fragment
  fn fragmentMain(input: FragInput) -> @location(0) vec4f {
    let c = input.cell / param.grid;
    return vec4f(c, 1 - c.x, 1);
  }
`

class WgslFramework {
  async setUpWgsl() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported on this browser.')
    }

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      throw new Error('No appropriate GPUAdapter found.')
    }
    const device = await adapter.requestDevice()

    const canvas = document.querySelector('canvas')
    const context = canvas.getContext('webgpu')
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat()
    context.configure({
      device: device,
      format: canvasFormat,
    })

    this.device = device
    this.context = context
    this.canvasFormat = canvasFormat
  }
}

function frandom(min, max) {
  return Math.random() * (max - min) + min
}

function irandom(min, max) {
  return Math.floor(frandom(min, max))
}

class MyApp extends WgslFramework {
  page = 0

  setUpComputeData() {
    const BUFFER_SIZE = GRID_SIZE * GRID_SIZE * 2 * 4

    const cellStateStorage = [
      this.device.createBuffer({
        label: 'Cell State A',
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      this.device.createBuffer({
        label: 'Cell State B',
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    ]

    const simulationShaderModule = this.device.createShaderModule({
      label: 'Game of Life simulation shader',
      code: kSimulationShaderCode,
    })

    this.cellStateStorage = cellStateStorage
    this.simulationShaderModule = simulationShaderModule

    this.randomizeCellState()
  }

  randomizeCellState() {
    const cellStateArray = new Float32Array(GRID_SIZE * GRID_SIZE * 2)

    for (let i = 0; i < cellStateArray.length; i += 2) {
      cellStateArray[i + 0] = 1.0
      cellStateArray[i + 1] = 0.0
    }
    const n = Math.floor(irandom(1, 16))
    for (let k = 0; k < n; ++k) {
      const size = Math.floor(irandom(2, 5))
      const x = Math.floor(irandom(size, GRID_SIZE - size))
      const y = Math.floor(irandom(size, GRID_SIZE - size))
      for (let i = -size; i <= size; ++i) {
        for (let j = -size; j <= size; ++j) {
          const p = (((y + i) * GRID_SIZE) + (x + j)) * 2
          cellStateArray[p + 0] = 0.0
          cellStateArray[p + 1] = 0.5
        }
      }
    }

    this.device.queue.writeBuffer(this.cellStateStorage[0], 0, cellStateArray)
  }

  setUpRenderingData() {
    const vertices = new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
       1.0,  1.0,

      -1.0, -1.0,
       1.0,  1.0,
      -1.0,  1.0,
    ])

    const vertexBuffer = this.device.createBuffer({
      label: 'Cell vertices',
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(vertexBuffer, 0, vertices)

    const vertexBufferLayout = {
      arrayStride: 8,
      attributes: [
        {
          format: 'float32x2',
          offset: 0,
          shaderLocation: 0,
        },
      ],
    }

    const cellShaderModule = this.device.createShaderModule({
      label: 'Cell shader',
      code: kCellShaderCode,
    })

    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.055, 0.062])
    // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.026, 0.061])
    // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.057])
    // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.065])  // バクテリア
    // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.060, 0.062])  // Coral pattern
    // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 0.9, 0.61, 0.023, 0.052])  // 動き続ける

    const uniformBuffer = this.device.createBuffer({
      label: 'Uniform parameter',
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray)

    this.vertices = vertices
    this.vertexBuffer = vertexBuffer
    this.vertexBufferLayout = vertexBufferLayout
    this.cellShaderModule = cellShaderModule
    this.uniformBuffer = uniformBuffer
  }

  setUpPipelineData() {
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'Cell Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    })

    const bindGroups = [
      this.device.createBindGroup({
        label: 'Cell renderer bind group A',
        layout: bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: { buffer: this.uniformBuffer },
          },
          {
            binding: 1,
            resource: { buffer: this.cellStateStorage[0] },
          },
          {
            binding: 2,
            resource: { buffer: this.cellStateStorage[1] },
          },
        ],
      }),
      this.device.createBindGroup({
        label: 'Cell renderer bind group B',
        layout: bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: { buffer: this.uniformBuffer },
          },
          {
            binding: 1,
            resource: { buffer: this.cellStateStorage[1] },
          },
          {
            binding: 2,
            resource: { buffer: this.cellStateStorage[0] },
          },
        ],
      }),
    ]

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Cell Pipeline Layout',
      bindGroupLayouts: [ bindGroupLayout ],
    })

    const simulationPipeline = this.device.createComputePipeline({
      label: 'Simulation pipeline',
      layout: pipelineLayout,
      compute: {
        module: this.simulationShaderModule,
        entryPoint: 'computeMain',
      },
    })

    const cellPipeline = this.device.createRenderPipeline({
      label: 'Cell pipeline',
      layout: pipelineLayout,
      vertex: {
        module: this.cellShaderModule,
        entryPoint: 'vertexMain',
        buffers: [this.vertexBufferLayout],
      },
      fragment: {
        module: this.cellShaderModule,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.canvasFormat,
        }],
      },
    })

    this.simulationPipeline = simulationPipeline
    this.cellPipeline = cellPipeline
    this.bindGroups = bindGroups
  }

  draw() {
    const encoder = this.device.createCommandEncoder()

    for (let i = 0; i < 8; ++i) {
      const computePass = encoder.beginComputePass()
      computePass.setPipeline(this.simulationPipeline)
      computePass.setBindGroup(0, this.bindGroups[this.page])
      const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
      computePass.dispatchWorkgroups(workgroupCount, workgroupCount)
      computePass.end()

      this.page = 1 - this.page
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0.2, a: 1.0 },
        storeOp: 'store',
      }]
    })

    pass.setPipeline(this.cellPipeline)
    pass.setBindGroup(0, this.bindGroups[this.page])
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.draw(this.vertices.length / 2, GRID_SIZE * GRID_SIZE)
    pass.end()

    this.device.queue.submit([encoder.finish()])
  }
}

async function main() {
  const myapp = new MyApp()
  await myapp.setUpWgsl()

  myapp.setUpComputeData()
  myapp.setUpRenderingData()
  myapp.setUpPipelineData()

  const UPDATE_INTERVAL = 20
  setInterval(myapp.draw.bind(myapp), UPDATE_INTERVAL)

  document.getElementById('reset-btn').addEventListener('click', () => {
    myapp.randomizeCellState()
  })
}

await main()
