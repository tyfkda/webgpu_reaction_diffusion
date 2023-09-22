struct VertexInput {
    @location(0) pos: vec2f,
    @location(1) uv: vec2f,
};

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var cellSampler: sampler;
@group(0) @binding(1) var cellTexture: texture_2d<f32>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.pos = vec4f(input.pos, 0, 1);
    output.uv = input.uv;
    return output;
}

struct FragInput {
    @location(0) uv: vec2f,
};

@fragment
fn fragmentMain(input: FragInput) -> @location(0) vec4f {
    let col = textureSample(cellTexture, cellSampler, input.uv);
    let b = col.y * 3;
    return vec4f(b, b, b, 1);
}
