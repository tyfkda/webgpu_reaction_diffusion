import {mat4} from 'https://wgpu-matrix.org/dist/2.x/wgpu-matrix.module.js'
const eulerZXY = (rx, ry, rz) => mat4.rotateY(mat4.rotateX(mat4.rotateZ(mat4.identity(), rz), rx), ry)

const WORKGROUP_SIZE = 8
const GRID_SIZE = 256

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
        this.canvas = canvas
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

async function fetchTextFile(path) {
    const response = await fetch(path)
    const text = await response.text()
    return text
}

class MyApp extends WgslFramework {
    page = 0
    erasePos = []
    drawing = false

    async loadShaderCode() {
        const shaderCodes = await Promise.all([
            fetchTextFile('reaction_diffusion_compute.wgsl'),
            fetchTextFile('reaction_diffusion_render.wgsl'),
        ])
        this.shaderCodes = shaderCodes
    }

    setUpComputeData() {
        const BUFFER_SIZE = GRID_SIZE * GRID_SIZE * 4 * 4

        this.stagingBuffer = this.device.createBuffer({
            label: 'Staging buffer',
            size: BUFFER_SIZE,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })

        const cellStateStorage = [
            this.device.createBuffer({
                label: 'Cell State A',
                size: BUFFER_SIZE,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            }),
            this.device.createBuffer({
                label: 'Cell State B',
                size: BUFFER_SIZE,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            }),
        ]

        const simulationShaderModule = this.device.createShaderModule({
            label: 'Game of Life simulation shader',
            code: this.shaderCodes[0],
        })

        this.cellStateStorage = cellStateStorage
        this.simulationShaderModule = simulationShaderModule

        this.randomizeCellState()
    }

    randomizeCellState() {
        const cellStateArray = new Float32Array(GRID_SIZE * GRID_SIZE * 4)

        for (let i = 0; i < cellStateArray.length; i += 4) {
            cellStateArray[i + 0] = 1.0
            cellStateArray[i + 1] = 0.0
            cellStateArray[i + 2] = 0.0
            cellStateArray[i + 3] = 1.0
        }
        const n = Math.floor(irandom(1, 32))
        for (let k = 0; k < n; ++k) {
            const size = Math.floor(irandom(2, 5))
            const x = Math.floor(irandom(size, GRID_SIZE - size))
            const y = Math.floor(irandom(size, GRID_SIZE - size))
            for (let i = -size; i <= size; ++i) {
                for (let j = -size; j <= size; ++j) {
                    const p = (((y + i) * GRID_SIZE) + (x + j)) * 4
                    cellStateArray[p + 0] = 0.5
                    cellStateArray[p + 1] = 0.5
                }
            }
        }

        this.device.queue.writeBuffer(this.cellStateStorage[0], 0, cellStateArray)
    }

    setUpRenderingData() {
        // const vertices = new Float32Array([
        //     -1.0, -1.0,  0.0, 0.0,
        //     1.0, -1.0,   1.0, 0.0,
        //     -1.0, 1.0,   0.0, 1.0,
        //     1.0, 1.0,    1.0, 1.0,
        // ])
        const vertices = new Float32Array([
            // float4 position, float4 UV

            // X-
            -1,  1,  1, 1,  0, 0,
            -1,  1, -1, 1,  1, 0,
            -1, -1,  1, 1,  0, 1,
            -1, -1, -1, 1,  1, 1,
            -1, -1,  1, 1,  0, 1,
            -1,  1, -1, 1,  1, 0,

            // Z-
            -1,  1, -1, 1,  0, 0,
             1,  1, -1, 1,  1, 0,
            -1, -1, -1, 1,  0, 1,
             1, -1, -1, 1,  1, 1,
            -1, -1, -1, 1,  0, 1,
             1,  1, -1, 1,  1, 0,

            // X+
             1,  1, -1, 1,   0, 0,
             1,  1,  1, 1,   1, 0,
             1, -1, -1, 1,   0, 1,
             1, -1,  1, 1,   1, 1,
             1, -1, -1, 1,   0, 1,
             1,  1,  1, 1,   1, 0,

            // Z+
             1,  1,  1, 1,   0, 0,
            -1,  1,  1, 1,   1, 0,
             1, -1,  1, 1,   0, 1,
            -1, -1,  1, 1,   1, 1,
             1, -1,  1, 1,   0, 1,
            -1,  1,  1, 1,   1, 0,

            // Y-
            -1, -1, -1, 1,  0, 0,
             1, -1, -1, 1,  1, 0,
            -1, -1,  1, 1,  0, 1,
             1, -1,  1, 1,  1, 1,
            -1, -1,  1, 1,  0, 1,
             1, -1, -1, 1,  1, 0,

            // Y+
            -1,  1,  1, 1,   0, 0,
             1,  1,  1, 1,   1, 0,
            -1,  1, -1, 1,   0, 1,
             1,  1, -1, 1,   1, 1,
            -1,  1, -1, 1,   0, 1,
             1,  1,  1, 1,   1, 0,
       ])

        const vertexBuffer = this.device.createBuffer({
            label: 'Cell vertices',
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(vertexBuffer, 0, vertices)

        const vertexBufferLayout = {
            arrayStride: 6 * 4,
            attributes: [
                {  // Pos
                    format: 'float32x4',
                    offset: 0,
                    shaderLocation: 0,
                },
                {  // UV
                    format: 'float32x2',
                    offset: 4 * 4,
                    shaderLocation: 1,
                },
            ],
        }

        const cellShaderModule = this.device.createShaderModule({
            label: 'Cell shader',
            code: this.shaderCodes[1],
        })

        this.texture = this.device.createTexture({
            label: 'Cell texture',
            size: [GRID_SIZE, GRID_SIZE],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })

        const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.055, 0.062])
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.026, 0.061])  // 点々
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.057])  // あみあみ
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.065])  // バクテリア
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.060, 0.062])  // Coral pattern
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 0.9, 0.61, 0.023, 0.052])  // 動き続ける

        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 0.2*4, 0.1*4, 0.082, 0.060])  // なにか
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 0.2*4.5, 0.1*4.5, 0.092, 0.057])
        // const simulationUniform = new Float32Array([GRID_SIZE, GRID_SIZE, 0.92, 0.5, 0.088, 0.057])

        const simulationUniformBuffer = this.device.createBuffer({
            label: 'Uniform parameter',
            size: simulationUniform.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(simulationUniformBuffer, 0, simulationUniform)

        const cellUniform = new Float32Array(4 * 4 * 4 * 3)  // f32 4x4 matrix x 3 (projection, view, world)
        const cellUniformBuffer = this.device.createBuffer({
            label: 'Uniform parameter',
            size: cellUniform.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(cellUniformBuffer, 0, cellUniform)

        this.vertices = vertices
        this.vertexBuffer = vertexBuffer
        this.vertexBufferLayout = vertexBufferLayout
        this.cellShaderModule = cellShaderModule
        this.simulationUniformBuffer = simulationUniformBuffer
        this.cellUniform = cellUniform
        this.cellUniformBuffer = cellUniformBuffer
    }

    setUpSimulationPipelineData() {
        const simulationPipeline = this.device.createComputePipeline({
            label: 'Simulation pipeline',
            layout: 'auto',
            compute: {
                module: this.simulationShaderModule,
                entryPoint: 'computeMain',
            },
        })

        const simulationBindGroupLayout = simulationPipeline.getBindGroupLayout(0)

        const simulationBindGroups = [
            this.device.createBindGroup({
                label: 'Simulation bind group A',
                layout: simulationBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.simulationUniformBuffer },
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
                label: 'Simulation bind group B',
                layout: simulationBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.simulationUniformBuffer },
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

        this.simulationPipeline = simulationPipeline
        this.simulationBindGroups = simulationBindGroups
    }

    setUpCellPipelineData() {
        const sampler = this.device.createSampler({
            addressModeU: 'repeat',
            addressModeV: 'repeat',
            // magFilter: 'linear',
            // minFilter: 'linear',
        })

        const cellBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Cell Bind Group Layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'uniform' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'non-filtering' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
            ],
        })

        const cellPipelineLayout = this.device.createPipelineLayout({
            label: 'Cell Pipeline Layout',
            bindGroupLayouts: [cellBindGroupLayout],
        })

        const cellPipeline = this.device.createRenderPipeline({
            label: 'Cell pipeline',
            layout: cellPipelineLayout,
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
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        })

        const cellBindGroup = this.device.createBindGroup({
            label: 'Cell renderer bind group',
            layout: cellBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.cellUniformBuffer },
                },
                {
                    binding: 1,
                    resource: sampler,
                },
                {
                    binding: 2,
                    resource: this.texture.createView(),
                },
            ],
        })

        this.cellPipeline = cellPipeline
        this.cellBindGroup = cellBindGroup
    }

    draw() {
        if (this.drawing)
            return
        this.drawing = true

        this.getTransformationMatrix(this.cellUniformBuffer)

        const encoder = this.device.createCommandEncoder()

        for (let i = 0; i < 8; ++i) {
            const computePass = encoder.beginComputePass()
            computePass.setPipeline(this.simulationPipeline)
            computePass.setBindGroup(0, this.simulationBindGroups[this.page])
            const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
            computePass.dispatchWorkgroups(workgroupCount, workgroupCount)
            computePass.end()

            this.page = 1 - this.page
        }

        encoder.copyBufferToTexture(
            {
                buffer: this.cellStateStorage[this.page],
                bytesPerRow: GRID_SIZE * 4 * 4,
            },
            {texture: this.texture},
            {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depthOrArrayLayers: 1,
            },
        )

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0.2, a: 1.0 },
                // clearValue: { r: 0xe1/255.0, g: 0xdd/255.0, b: 0xd3/255.0, a: 1.0 },
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        })

        pass.setPipeline(this.cellPipeline)
        pass.setBindGroup(0, this.cellBindGroup)
        pass.setVertexBuffer(0, this.vertexBuffer)
        pass.draw(this.vertices.length / 6)
        pass.end()

        if (this.erasePos.length > 0) {
            const output = this.cellStateStorage[this.page]
            encoder.copyBufferToBuffer(
                output,
                0, // Source offset
                this.stagingBuffer,
                0, // Destination offset
                this.stagingBuffer.size,
            )
        }

        this.device.queue.submit([encoder.finish()])

        if (this.erasePos.length <= 0) {
            this.drawing = false
        } else {
            this.runErase()
                .then(() => this.drawing = false)
        }
    }

    getTransformationMatrix(uniformBuffer) {
        const fovy = 45 * Math.PI / 180
        const aspect = this.canvas.width / this.canvas.height
        const near = 0.1
        const far = 1000
        const projectionMatrix = mat4.perspective(fovy, aspect, near, far)
        this.device.queue.writeBuffer(
            uniformBuffer, 4 * 16 * 0,
            projectionMatrix.buffer, projectionMatrix.byteOffset, projectionMatrix.byteLength)

        const eye = [0, 0, -5]
        const target = [0, 0, 0]
        const up = [0, 1, 0]
        const viewMatrix = mat4.lookAt(eye, target, up)
        this.device.queue.writeBuffer(
            uniformBuffer, 4 * 16 * 1,
            viewMatrix.buffer, viewMatrix.byteOffset, viewMatrix.byteLength)

        const t = Date.now() / 6000
        const rx = t * 2
        const ry = t * 3
        const rz = t * 5

        const worldMatrix = eulerZXY(rx, ry, rz)
        this.device.queue.writeBuffer(
            uniformBuffer, 4 * 16 * 2,
            worldMatrix.buffer, worldMatrix.byteOffset, worldMatrix.byteLength)
      }


    async runErase() {
        const BUFFER_SIZE = this.stagingBuffer.size
        await this.stagingBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            BUFFER_SIZE // Length
        )

        const copyArrayBuffer = this.stagingBuffer.getMappedRange(0, BUFFER_SIZE)
        const data = copyArrayBuffer.slice()
        this.stagingBuffer.unmap()
        const cellStateArray = new Float32Array(data)

        const width = GRID_SIZE
        const height = GRID_SIZE

        for (let i = 0; i < this.erasePos.length; i += 2) {
            const cx = this.erasePos[i]
            const cy = this.erasePos[i + 1]

            const radius = (GRID_SIZE / 32) | 0, radius2 = radius * radius
            const x = cx | 0, y = cy | 0
            const dx0 = Math.max(-radius, -x), dy0 = Math.max(-radius, -y)
            const dx1 = Math.min(radius, width - 1 - x), dy1 = Math.min(radius, height - 1 - y)
            for (let dy = dy0; dy <= dy1; ++dy) {
                for (let dx = dx0; dx <= dx1; ++dx) {
                    if (dx * dx + dy * dy >= radius2)
                        continue
                    const i = ((y + dy) * GRID_SIZE + (x + dx)) * 4
                    cellStateArray[i + 0] = 1.0  // a
                    cellStateArray[i + 1] = 0.0  // b
                }
            }
        }
        this.erasePos.length = 0

        this.device.queue.writeBuffer(this.cellStateStorage[this.page], 0, cellStateArray)
    }

    async pushErase(cx, cy) {
        this.erasePos.push(cx, cy)
    }
}

async function main() {
    const myapp = new MyApp()
    await myapp.loadShaderCode()
    await myapp.setUpWgsl()

    myapp.setUpComputeData()
    myapp.setUpRenderingData()
    myapp.setUpSimulationPipelineData()
    myapp.setUpCellPipelineData()

    const interval = () => {
        myapp.draw()
        requestAnimationFrame(interval)
    }
    requestAnimationFrame(interval)

    document.getElementById('reset-btn').addEventListener('click', () => {
        myapp.randomizeCellState()
    })

    const canvas = document.querySelector('canvas')
    canvas.addEventListener('mousedown', (event) => {
        const clientRect = canvas.getBoundingClientRect()
        const erase = (ev) => {
            const cx = ev.clientX - clientRect.x
            const cy = ev.clientY - clientRect.y
            // console.log(`${cx}, ${cy}`)
            if (0 <= cx && cx < canvas.width && 0 <= cy && cy < canvas.height) {
                myapp.pushErase(cx * GRID_SIZE / clientRect.width, (clientRect.height - 1 - cy) * GRID_SIZE / clientRect.height)
            }
        }
        erase(event)

        const mousemove = (event) => {
            erase(event)
        }
        const mouseup = (_event) => {
            document.removeEventListener('mousemove', mousemove)
            document.removeEventListener('mouseup', mouseup)
        }
        document.addEventListener('mousemove', mousemove)
        document.addEventListener('mouseup', mouseup)
    })
}

await main()
