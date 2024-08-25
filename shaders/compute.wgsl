// x                : 0 | r
// y                : 1 | g
// x_0 & divergence : 2 | b
// y_0              : 3 | a
@group(0) @binding(0) var velocity_grid: texture_storage_2d<rgba8unorm, read_write>;
// pressure     : 0 | r
// pressure_1   : 1 | g
// x_1          : 2 | b
// y_1          : 3 | a
@group(0) @binding(1) var pressure_grid: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(2) var addition_grid: texture_storage_2d<rgba8unorm, read_write>;

// new x velocity   : 0 | r
// new y velocity   : 1 | g
// new density      : 2 | b
@group(0) @binding(3) var output_density: texture_storage_2d<rgba8unorm, read_write>;

fn lerp(a: f32, b: f32, k: f32) -> f32 {
    return a + k * (b - a);
}

fn get_neighbors_velocity(x: i32, y: i32) -> array<vec4<f32>, 4> {
    return array(
        textureLoad(velocity_grid, vec2i(x, y - 1)),
        textureLoad(velocity_grid, vec2i(x, y + 1)),
        textureLoad(velocity_grid, vec2i(x + 1, y)),
        textureLoad(velocity_grid, vec2i(x - 1, y)),
    );
}

fn get_neighbors_pressure(x: i32, y: i32) -> array<vec4<f32>, 4> {
    return array(
        textureLoad(pressure_grid, vec2i(x, y - 1)),
        textureLoad(pressure_grid, vec2i(x, y + 1)),
        textureLoad(pressure_grid, vec2i(x + 1, y)),
        textureLoad(pressure_grid, vec2i(x - 1, y)),
    );
}

fn jacobi (
    resolution: u32,
    alpha: f32,
    beta: f32,
    value_b: f32,
    value_north: f32,
    value_south: f32,
    value_east:  f32,
    value_west:  f32
) -> f32 {
    return (
        value_north +
        value_south +
        value_east +
        value_west +
        alpha * value_b
    ) / beta;
}

fn jacobi_2(
    resolution: u32,
    alpha: f32,
    beta: f32,
    value_b: vec2<f32>,
    value_north: vec2<f32>,
    value_south: vec2<f32>,
    value_east: vec2<f32>,
    value_west: vec2<f32>
) -> vec2<f32> {
    return (
        value_north +
        value_south +
        value_east +
        value_west +
        alpha * value_b
    ) / beta;
}

fn diffuse(
    resolution: u32,
    num_iters: i32,
    dt: f32,
    x: i32,
    y: i32,
    viscosity: f32,
) {
    let neighbors = get_neighbors_velocity(x, y);
    let res_f = 1.0 / f32(resolution);
    let alpha = (res_f * res_f) / (viscosity * dt);
    let beta = 4.0 + alpha;
    for (var i = 0i; i < num_iters; i++) {
        workgroupBarrier();
        let velocity = textureLoad(velocity_grid, vec2i(x, y));
        let res = jacobi_2(
            resolution,
            alpha,
            beta,
            velocity.xy,
            neighbors[0].zw,
            neighbors[1].zw,
            neighbors[2].zw,
            neighbors[3].zw
        );

        var new_velocity = textureLoad(pressure_grid, vec2i(x, y));
        new_velocity[2] = res[0];
        new_velocity[3] = res[1];

        textureStore(pressure_grid, vec2i(x, y), new_velocity);

        workgroupBarrier();

        let new_neighbors = get_neighbors_pressure(x, y);

        let res_2 = jacobi_2(
            resolution,
            alpha,
            beta,
            velocity.xy,
            new_neighbors[0].zw,
            new_neighbors[1].zw,
            new_neighbors[2].zw,
            new_neighbors[3].zw
        );

        new_velocity = textureLoad(velocity_grid, vec2i(x, y));
        new_velocity[2] = res_2[0];
        new_velocity[3] = res_2[1];
    }
}

fn advect_velocity(
    resolution: u32,
    is_y_velocity: bool,
    // swap: bool,
    dt: f32,
    x: i32,
    y: i32,
) -> f32 {
    var current_channel = i32(is_y_velocity);
    var past_channel = current_channel + 2;

    /* if swap {
        let temp = current_channel;
        current_channel = past_channel;
        past_channel = temp;
    } */

    let vel_x = textureLoad(velocity_grid, vec2i(x, y))[0];
    let vel_y = textureLoad(velocity_grid, vec2i(x, y))[1];

    let prev_x = clamp(f32(x) - dt * vel_x, 0.5, f32(resolution) + 0.5);
    let prev_y = clamp(f32(y) - dt * vel_y, 0.5, f32(resolution) + 0.5);

    let top_left_value = textureLoad(velocity_grid, vec2i(i32(prev_x), i32(prev_y)))[past_channel];
    let top_right_value = textureLoad(velocity_grid, vec2i(i32(prev_x) + 1, i32(prev_y)))[past_channel];

    let bottom_left_value = textureLoad(velocity_grid, vec2i(i32(prev_x), i32(prev_y) + 1))[past_channel];
    let bottom_right_value = textureLoad(velocity_grid, vec2i(i32(prev_x) + 1, i32(prev_y) + 1))[past_channel];

    let top_lerp = lerp(top_left_value, top_right_value, fract(prev_x));
    let bottom_lerp = lerp(bottom_left_value, bottom_right_value, fract(prev_x));

    let final_value = lerp(top_lerp, bottom_lerp, fract(prev_y));

    return final_value;
}

fn divergence(resolution: u32, x: i32, y: i32) -> f32 {
    let neighbors = get_neighbors_pressure(x, y);
    let div = ((neighbors[2][2] - neighbors[3][2]) + (neighbors[0][3] - neighbors[1][3])) / 2.0 * f32(resolution);

    var new_pressure = textureLoad(pressure_grid, vec2i(x, y));
    new_pressure[0] = 0.0;
    new_pressure[1] = 0.0;
    textureStore(pressure_grid, vec2i(x, y), new_pressure);
    
    var new_vel = textureLoad(velocity_grid, vec2i(x, y));
    new_vel[2] = div;

    return div;
}

fn gradient(resolution: u32, x: i32, y: i32) {
    let pressure_neighbors = get_neighbors_pressure(x, y);

    let old_velocity = textureLoad(pressure_grid, vec2i(x, y));

    var velocity = textureLoad(velocity_grid, vec2i(x, y));

    let new_velocity_x = old_velocity[0] - (pressure_neighbors[2][0] - pressure_neighbors[3][0]) / (2.0 * f32(resolution));
    let new_velocity_y = old_velocity[1] - (pressure_neighbors[0][0] - pressure_neighbors[1][0]) / (2.0 * f32(resolution));

    velocity[0] = new_velocity_x;
    velocity[1] = new_velocity_y;

    textureStore(velocity_grid, vec2i(x, y), velocity);
}

fn project(
    resolution: u32,
    x: i32,
    y: i32,
    num_iters: i32,
) {
    let neighbors = get_neighbors_pressure(x, y);
    let divergence = divergence(resolution, x, y);
    workgroupBarrier();


    for (var i = 0i; i < num_iters; i++) {
        workgroupBarrier();
        let res = jacobi(
            resolution,
            -1.0 / (f32(resolution) * f32(resolution)),
            4.0,
            divergence,
            neighbors[0][0],
            neighbors[1][0],
            neighbors[2][0],
            neighbors[3][0]
        );

        var new_pressure = textureLoad(pressure_grid, vec2i(x, y));
        new_pressure[1] = res;

        textureStore(pressure_grid, vec2i(x, y), new_pressure);

        workgroupBarrier();

        let res_2 = jacobi(
            resolution,
            -1.0 / (f32(resolution) * f32(resolution)),
            4.0,
            divergence,
            neighbors[0][1],
            neighbors[1][1],
            neighbors[2][1],
            neighbors[3][1]
        );

        new_pressure = textureLoad(pressure_grid, vec2i(x, y));
        new_pressure[0] = res_2;
        textureStore(pressure_grid, vec2i(x, y), new_pressure);

    }

    // gradient(resolution, x, y);
    let old_velocity = textureLoad(pressure_grid, vec2i(x, y));
    var new_velocity = textureLoad(velocity_grid, vec2i(x, y));

    new_velocity[0] = old_velocity[2];
    new_velocity[1] = old_velocity[3];
    textureStore(velocity_grid, vec2i(x, y), new_velocity);
}

fn boundary(x: i32, y: i32, is_pressure: bool, offset: vec2<i32>) {
    if is_pressure {
        let pressure = textureLoad(pressure_grid, vec2i(x, y) + offset);
        textureStore(pressure_grid, vec2i(x, y), pressure);
    } else {
        let velocity = textureLoad(velocity_grid, vec2i(x, y) + offset);
        textureStore(velocity_grid, vec2i(x, y), velocity * -1.0);
    }
}


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let resolution = 256u;
    let viscosity = 0.00001f;
    let diffusion = 0.0f;
    let num_iters = 20i;
    let dt = 0.01f;
    if x != 0 && y != 0 && x != i32(resolution) + 1 && y != i32(resolution) + 1 {

        let added_vel_x = textureLoad(addition_grid, vec2i(x, y))[0] * dt;
        let added_vel_y = textureLoad(addition_grid, vec2i(x, y))[1] * dt;

        let added_density = textureLoad(addition_grid, vec2i(x, y))[2];


        /* var current_dens = textureLoad(density_grid, vec2i(x, y));
        current_dens[0] += added_density;
        textureStore(velocity_grid, vec2i(x, y), current_vel); */


        workgroupBarrier();
        let advect_x = advect_velocity(resolution, false, dt, x, y);
        let advect_y = advect_velocity(resolution, true, dt, x, y);

        textureStore(velocity_grid, vec2i(x, y), vec4f(advect_x, advect_y, advect_x, advect_y));

        workgroupBarrier();

        diffuse(resolution, num_iters, dt, x, y, viscosity);

        workgroupBarrier();

        let old_pressure = textureLoad(pressure_grid, vec2i(x, y));
        var current_vel = textureLoad(velocity_grid, vec2i(x, y));
        current_vel[0] = old_pressure[0];
        current_vel[1] = old_pressure[1];
        current_vel[2] += added_vel_x;
        current_vel[3] += added_vel_y;

        textureStore(pressure_grid, vec2i(x, y), current_vel);

        workgroupBarrier();

        project(resolution, x, y, num_iters * 2);
        workgroupBarrier();

        let pressure = textureLoad(velocity_grid, vec2i(x, y)).xy;

        textureStore(output_density, vec2i(x - 1, y - 1), vec4f(pressure, 0.0, 1.0));
    } else {
        if x == 0 {
            boundary(x, y, false, vec2i(1, 0));
            boundary(x, y, true, vec2i(1, 0));
        }
        if x == i32(resolution) + 1 {
            boundary(x, y, false, vec2i(-1, 0));
            boundary(x, y, true, vec2i(-1, 0));
        }
        if y == 0 {
            boundary(x, y, false, vec2i(0, 1));
            boundary(x, y, true, vec2i(0, 1));
        }
        if y == i32(resolution) + 1 {
            boundary(x, y, false, vec2i(0, -1));
            boundary(x, y, true, vec2i(0, -1));
        }
    }
}
