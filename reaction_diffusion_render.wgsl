struct VertexInput {
    @location(0) pos: vec2f,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) cell: vec2f,
    @location(1) state: vec2f,
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
    let state = cellState[input.instance];

    let cellOffset = cell / param.grid * 2;
    let gridPos = (input.pos + 1) / param.grid - 1 + cellOffset;

    var output: VertexOutput;
    output.pos = vec4f(gridPos, 0, 1);
    output.cell = cell;
    output.state = vec2f(state.a, state.b);
    return output;
}

struct FragInput {
    @location(0) cell: vec2f,
    @location(1) state: vec2f,
};

@fragment
fn fragmentMain(input: FragInput) -> @location(0) vec4f {
    let c = input.cell / param.grid;
    let a = min(input.state.y * 3.0, 1.0);
    return vec4f(c, 1 - c.x, a);
    // return vec4f(0x78/255.0, 0x45/255.0, 0x2a/255.0, a);
}
