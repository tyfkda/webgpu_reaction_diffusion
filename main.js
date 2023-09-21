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
        const BUFFER_SIZE = GRID_SIZE * GRID_SIZE * 2 * 4

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
        const cellStateArray = new Float32Array(GRID_SIZE * GRID_SIZE * 2)

        for (let i = 0; i < cellStateArray.length; i += 2) {
            cellStateArray[i + 0] = 1.0
            cellStateArray[i + 1] = 0.0
        }
        const n = Math.floor(irandom(1, 32))
        for (let k = 0; k < n; ++k) {
            const size = Math.floor(irandom(2, 5))
            const x = Math.floor(irandom(size, GRID_SIZE - size))
            const y = Math.floor(irandom(size, GRID_SIZE - size))
            for (let i = -size; i <= size; ++i) {
                for (let j = -size; j <= size; ++j) {
                    const p = (((y + i) * GRID_SIZE) + (x + j)) * 2
                    cellStateArray[p + 0] = 0.5
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
            -1.0, 1.0,
            1.0, 1.0,
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
            code: this.shaderCodes[1],
        })

        const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.055, 0.062])
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.026, 0.061])  // 点々
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.057])  // あみあみ
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.035, 0.065])  // バクテリア
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 1.0, 0.5, 0.060, 0.062])  // Coral pattern
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 0.9, 0.61, 0.023, 0.052])  // 動き続ける

        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 0.2*4, 0.1*4, 0.082, 0.060])  // なにか
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 0.2*4.5, 0.1*4.5, 0.092, 0.057])
        // const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE, 0.92, 0.5, 0.088, 0.057])

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
            bindGroupLayouts: [bindGroupLayout],
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
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        })

        this.simulationPipeline = simulationPipeline
        this.cellPipeline = cellPipeline
        this.bindGroups = bindGroups
    }

    draw() {
        if (this.drawing)
            return
        this.drawing = true

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
                // clearValue: { r: 0xe1/255.0, g: 0xdd/255.0, b: 0xd3/255.0, a: 1.0 },
                storeOp: 'store',
            }]
        })

        pass.setPipeline(this.cellPipeline)
        pass.setBindGroup(0, this.bindGroups[this.page])
        pass.setVertexBuffer(0, this.vertexBuffer)
        pass.draw(this.vertices.length / 2, GRID_SIZE * GRID_SIZE)
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
                .then(() => {
                    this.drawing = false
                })
        }
    }

    async runErase() {
        const BUFFER_SIZE = this.stagingBuffer.size
        this.stagingBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            BUFFER_SIZE // Length
        ).then(() => {
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
                        const i = ((y + dy) * GRID_SIZE + (x + dx)) * 2
                        cellStateArray[i + 0] = 1.0  // a
                        cellStateArray[i + 1] = 0.0  // b
                    }
                }
            }
            this.erasePos.length = 0

            this.device.queue.writeBuffer(this.cellStateStorage[this.page], 0, cellStateArray)
        })
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
    myapp.setUpPipelineData()

    const UPDATE_INTERVAL = 20
    setInterval(myapp.draw.bind(myapp), UPDATE_INTERVAL)

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
