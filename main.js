const WORKGROUP_SIZE = 8
const GRID_SIZE = 32

const kSimulationShaderCode = `
  @group(0) @binding(0) var<uniform> grid: vec2f;
  @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
  @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

  fn cellIndex(cell: vec2u) -> u32 {
    return (cell.y % u32(grid.y)) * u32(grid.x) +
           (cell.x % u32(grid.x));
  }

  fn cellActive(x: u32, y: u32) -> u32 {
    return cellStateIn[cellIndex(vec2(x, y))];
  }

  @compute
  @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
  fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
    let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                          cellActive(cell.x+1, cell.y) +
                          cellActive(cell.x+1, cell.y-1) +
                          cellActive(cell.x, cell.y-1) +
                          cellActive(cell.x-1, cell.y-1) +
                          cellActive(cell.x-1, cell.y) +
                          cellActive(cell.x-1, cell.y+1) +
                          cellActive(cell.x, cell.y+1);

    let i = cellIndex(cell.xy);

    // Conway's game of life rules:
    switch activeNeighbors {
      case 2: { // Active cells with 2 neighbors stay active.
        cellStateOut[i] = cellStateIn[i];
      }
      case 3: { // Cells with 3 neighbors become or stay active.
        cellStateOut[i] = 1;
      }
      default: { // Cells with < 2 or > 3 neighbors become inactive.
        cellStateOut[i] = 0;
      }
    }
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

  @group(0) @binding(0) var<uniform> grid: vec2f;
  @group(0) @binding(1) var<storage> cellState: array<u32>;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    let i = f32(input.instance);

    let cell = vec2f(i % grid.x, floor(i / grid.x));
    let state = f32(cellState[input.instance]);

    let cellOffset = cell / grid * 2;
    let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

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
    let c = input.cell / grid;
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

class MyApp extends WgslFramework {
  page = 0

  setUpComputeData() {
    const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE)

    const cellStateStorage = [
      this.device.createBuffer({
        label: 'Cell State A',
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      this.device.createBuffer({
        label: 'Cell State B',
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    ]

    for (let i = 0; i < cellStateArray.length; ++i) {
      cellStateArray[i] = Math.random() > 0.6 ? 1 : 0
    }
    this.device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray)
    this.device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray)  // Dummy.

    const simulationShaderModule = this.device.createShaderModule({
      label: 'Game of Life simulation shader',
      code: kSimulationShaderCode,
    })

    this.cellStateStorage = cellStateStorage
    this.simulationShaderModule = simulationShaderModule
  }

  setUpRenderingData() {
    const vertices = new Float32Array([
      -0.8, -0.8,
       0.8, -0.8,
       0.8,  0.8,

      -0.8, -0.8,
       0.8,  0.8,
      -0.8,  0.8,
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

    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE])
    const uniformBuffer = this.device.createBuffer({
      label: 'Grid Uniforms',
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

    const computePass = encoder.beginComputePass()
    computePass.setPipeline(this.simulationPipeline)
    computePass.setBindGroup(0, this.bindGroups[this.page])
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount)
    computePass.end()

    this.page = 1 - this.page

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

  const UPDATE_INTERVAL = 100
  setInterval(myapp.draw.bind(myapp), UPDATE_INTERVAL)
}

await main()
