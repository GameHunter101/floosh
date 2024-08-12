struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(2) @binding(0)
var<uniform> grid: array<array<vec4<f32>, 32>, 128>;

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    // return textureSample(texture, texture_sampler, in.tex_coords) * coefficient;

    let x_coord = u32(floor(in.tex_coords.x * 128.0));
    let y_coord = u32(floor(in.tex_coords.y * 128.0));
    let vec_index = y_coord / 4u;
    let rem = y_coord % 4u;

    let value = grid[x_coord][vec_index][rem] / 10.0;

    return vec4f(value, value, value, 1.0);
}
