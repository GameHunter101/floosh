// x   : 0 | r
// y   : 1 | g
// x_0 : 2 | b
// y_0 : 3 | a
@group(0) @binding(0) var velocity_grid: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var density_grid: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(2) var addition_grid: texture_storage_2d<rgba8unorm, read_write>;

// new x velocity   : 0 | r
// new y velocity   : 1 | g
// new density      : 2 | b
@group(0) @binding(3) var output_density: texture_storage_2d<rgba8unorm, read_write>;

fn set_boundary(resolution: u32, boundary_type: u32, x: i32, y: i32, current_value: f32, value_north: f32, value_south: f32, value_east: f32, value_west: f32) -> f32 {
    if x == 0 && y == 0 {
        return 0.5 * (value_east + value_south);
    }
    if x == i32(resolution + 1u) && y == 0 {
        return 0.5 * (value_west + value_south);
    }
    if x == 0 && y == i32(resolution + 1u) {
        return 0.5 * (value_east + value_north);
    }
    if x == i32(resolution + 1u) && y == i32(resolution + 1u) {
        return 0.5 * (value_west + value_north);
    }

    if x == 0i {
        if boundary_type == 2u { return -value_east;} else {return value_east;};
    }

    if x == i32(resolution + 1u) {
        if boundary_type == 2u { return -value_west;} else {return value_west;};
    }

    if y == 0i {
        if boundary_type == 2u { return -value_north;} else {return value_north;};
    }

    if y == i32(resolution + 1u) {
        if boundary_type == 2u { return -value_south;} else {return value_south;};
    }

    return current_value;
}

fn lerp(a: f32, b: f32, k: f32) -> f32 {
    return a + k * (b - a);
}

fn diffuse_velocity(
    resolution: u32,
    boundary_type: u32,
    is_y_velocity: bool,
    swap: bool,
    x: i32,
    y: i32,
    diffusion_coefficient: f32,
    num_iters: u32,
    dt: f32,
) {
    let coefficient = dt * diffusion_coefficient * f32(resolution) * f32(resolution);
    let divisor = 1.0 + 4.0 * coefficient;

    var current_channel = i32(is_y_velocity);
    var past_channel = current_channel + 2;

    if swap {
        let temp = current_channel;
        current_channel = past_channel;
        past_channel = temp;
    }

    for (var i: i32 = 0; i < i32(num_iters); i++) {
        let current_value = textureLoad(velocity_grid, vec2i(x, y));
        let previous_value = textureLoad(velocity_grid, vec2i(x, y))[past_channel];

        let east_value = textureLoad(velocity_grid, vec2i(x + 1, y))[current_channel];
        let west_value = textureLoad(velocity_grid, vec2i(x - 1, y))[current_channel];
        let north_value = textureLoad(velocity_grid, vec2i(x, y - 1))[current_channel];
        let south_value = textureLoad(velocity_grid, vec2i(x, y + 1))[current_channel];

        if (x + y) % 2 == 0 {
            let calculated_value = (previous_value + coefficient * (north_value + south_value + east_value + west_value)) / divisor;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(velocity_grid, vec2i(x, y), color);

        }
        workgroupBarrier();
        if (x+y) % 2 != 0 {
            let calculated_value = (previous_value + coefficient * (north_value + south_value + east_value + west_value)) / divisor;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(velocity_grid, vec2i(x, y), color);
        }
        workgroupBarrier();

        let boundary_value = set_boundary(resolution, boundary_type, x, y, current_value[current_channel], north_value, south_value, east_value, west_value);
        var color = current_value;
        color[current_channel] = boundary_value;
        textureStore(velocity_grid, vec2i(x, y), color);
    }
}

fn diffuse_density(
    resolution: u32,
    boundary_type: u32,
    swap: bool,
    x: i32,
    y: i32,
    diffusion_coefficient: f32,
    num_iters: u32,
    dt: f32,
) {
    let coefficient = dt * diffusion_coefficient * f32(resolution) * f32(resolution);
    let divisor = 1.0 + 4.0 * coefficient;

    var current_channel = 0;
    var past_channel = 1;

    if swap {
        let temp = current_channel;
        current_channel = past_channel;
        past_channel = temp;
    }

    for (var i: i32 = 0; i < i32(num_iters); i++) {
        let current_value = textureLoad(density_grid, vec2i(x, y));
        let previous_value = textureLoad(density_grid, vec2i(x, y))[past_channel];

        let east_value = textureLoad(density_grid, vec2i(x + 1, y))[current_channel];
        let west_value = textureLoad(density_grid, vec2i(x - 1, y))[current_channel];
        let north_value = textureLoad(density_grid, vec2i(x, y - 1))[current_channel];
        let south_value = textureLoad(density_grid, vec2i(x, y + 1))[current_channel];

        if (x + y) % 2 == 0 {
            let calculated_value = (previous_value + coefficient * (north_value + south_value + east_value + west_value)) / divisor;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(density_grid, vec2i(x, y), color);

        }
        workgroupBarrier();
        if (x+y) % 2 != 0 {
            let calculated_value = (previous_value + coefficient * (north_value + south_value + east_value + west_value)) / divisor;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(density_grid, vec2i(x, y), color);
        }
        workgroupBarrier();

        let boundary_value = set_boundary(resolution, boundary_type, x, y, current_value[current_channel], north_value, south_value, east_value, west_value);
        var color = current_value;
        color[current_channel] = boundary_value;
        textureStore(density_grid, vec2i(x, y), color);
    }
}

fn advect_velocity(
    resolution: u32,
    boundary_type: u32,
    is_y_velocity: bool,
    swap: bool,
    dt: f32,
    x: i32,
    y: i32,
) {
    var current_channel = i32(is_y_velocity);
    var past_channel = current_channel + 2;

    if swap {
        let temp = current_channel;
        current_channel = past_channel;
        past_channel = temp;
    }

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

    var color = textureLoad(velocity_grid, vec2i(x, y));

    color[current_channel] = final_value;

    textureStore(velocity_grid, vec2i(x, y), color);

    workgroupBarrier();

    let current_value = textureLoad(velocity_grid, vec2i(x, y));

    let north_value = textureLoad(velocity_grid, vec2i(x, y - 1))[current_channel];
    let south_value = textureLoad(velocity_grid, vec2i(x, y + 1))[current_channel];
    let east_value = textureLoad(velocity_grid, vec2i(x + 1, y))[current_channel];
    let west_value = textureLoad(velocity_grid, vec2i(x - 1, y))[current_channel];

    let boundary_value = set_boundary(resolution, boundary_type, x, y, current_value[current_channel], north_value, south_value, east_value, west_value);
    color = current_value;
    color[current_channel] = boundary_value;
    textureStore(velocity_grid, vec2i(x, y), color);
}

fn advect_density(
    resolution: u32,
    boundary_type: u32,
    swap: bool,
    dt: f32,
    x: i32,
    y: i32,
) {
    var current_channel = 0;
    var past_channel = 1;

    if swap {
        let temp = current_channel;
        current_channel = past_channel;
        past_channel = temp;
    }

    let vel_x = textureLoad(velocity_grid, vec2i(x, y))[0];
    let vel_y = textureLoad(velocity_grid, vec2i(x, y))[1];

    let prev_x = clamp(f32(x) - dt * vel_x, 0.5, f32(resolution) + 0.5);
    let prev_y = clamp(f32(y) - dt * vel_y, 0.5, f32(resolution) + 0.5);

    let top_left_value = textureLoad(density_grid, vec2i(i32(prev_x), i32(prev_y)))[past_channel];
    let top_right_value = textureLoad(density_grid, vec2i(i32(prev_x) + 1, i32(prev_y)))[past_channel];

    let bottom_left_value = textureLoad(density_grid, vec2i(i32(prev_x), i32(prev_y) + 1))[past_channel];
    let bottom_right_value = textureLoad(density_grid, vec2i(i32(prev_x) + 1, i32(prev_y) + 1))[past_channel];

    let top_lerp = lerp(top_left_value, top_right_value, fract(prev_x));
    let bottom_lerp = lerp(bottom_left_value, bottom_right_value, fract(prev_x));

    let final_value = lerp(top_lerp, bottom_lerp, fract(prev_y));

    var color = textureLoad(density_grid, vec2i(x, y));

    color[current_channel] = final_value;

    textureStore(density_grid, vec2i(x, y), color);

    workgroupBarrier();

    let current_value = textureLoad(density_grid, vec2i(x, y));

    let north_value = textureLoad(density_grid, vec2i(x, y - 1))[current_channel];
    let south_value = textureLoad(density_grid, vec2i(x, y + 1))[current_channel];
    let east_value = textureLoad(density_grid, vec2i(x + 1, y))[current_channel];
    let west_value = textureLoad(density_grid, vec2i(x - 1, y))[current_channel];

    let boundary_value = set_boundary(resolution, boundary_type, x, y, current_value[current_channel], north_value, south_value, east_value, west_value);
    color = current_value;
    color[current_channel] = boundary_value;
    textureStore(density_grid, vec2i(x, y), color);
}

fn project(
    resolution: u32,
    x: i32,
    y: i32,
    num_iters: u32,
) {
    let cell_size = 1.0 / f32(resolution);
    var current_value = textureLoad(velocity_grid, vec2i(x,y));

    let north_value = textureLoad(velocity_grid, vec2i(x, y - 1));
    let south_value = textureLoad(velocity_grid, vec2i(x, y + 1));
    let east_value = textureLoad(velocity_grid, vec2i(x + 1, y));
    let west_value = textureLoad(velocity_grid, vec2i(x - 1, y));

    // div
    current_value[2] = -0.5 * cell_size * (east_value[0] - west_value[0] + south_value[1] - north_value[1]);

    // p
    current_value[3] = 0.0;

    textureStore(velocity_grid, vec2i(x, y), current_value);

    workgroupBarrier();

    let boundary_value_div = set_boundary(resolution, 0u, x, y, current_value[2], north_value[2], south_value[2], east_value[2], west_value[2]);
    let boundary_value_p = set_boundary(resolution, 0u, x, y, current_value[3], north_value[3], south_value[3], east_value[3], west_value[3]);
    var color = current_value;
    current_value[2] = boundary_value_div;
    current_value[3] = boundary_value_p;
    textureStore(velocity_grid, vec2i(x, y), color);

    let current_channel = 3;
    let past_channel = 2;

    for (var i: i32 = 0; i < i32(num_iters); i++) {
        let current_value = textureLoad(velocity_grid, vec2i(x, y));
        let previous_value = textureLoad(velocity_grid, vec2i(x, y))[past_channel];

        let east_value = textureLoad(velocity_grid, vec2i(x + 1, y))[current_channel];
        let west_value = textureLoad(velocity_grid, vec2i(x - 1, y))[current_channel];
        let north_value = textureLoad(velocity_grid, vec2i(x, y - 1))[current_channel];
        let south_value = textureLoad(velocity_grid, vec2i(x, y + 1))[current_channel];

        if (x + y) % 2 == 0 {
            let calculated_value = (previous_value + (north_value + south_value + east_value + west_value)) / 4.0;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(velocity_grid, vec2i(x, y), color);

        }
        workgroupBarrier();
        if (x+y) % 2 != 0 {
            let calculated_value = (previous_value + (north_value + south_value + east_value + west_value)) / 4.0;
            var color = current_value;
            color[current_channel] = calculated_value;
            textureStore(velocity_grid, vec2i(x, y), color);
        }
        workgroupBarrier();

        let boundary_value = set_boundary(resolution, 0u, x, y, current_value[current_channel], north_value, south_value, east_value, west_value);
        var color = current_value;
        color[current_channel] = boundary_value;
        textureStore(velocity_grid, vec2i(x, y), color);
    }

    workgroupBarrier();

    color = textureLoad(velocity_grid, vec2i(x, y));
    color[0] -= 0.5 * (textureLoad(velocity_grid, vec2i(x, y + 1))[3] - textureLoad(velocity_grid, vec2i(x, y - 1))[3]) / cell_size;
    color[1] -= 0.5 * (textureLoad(velocity_grid, vec2i(x + 1, y))[3] - textureLoad(velocity_grid, vec2i(x - 1, y))[3]) / cell_size;

    textureStore(velocity_grid, vec2i(x, y), color);

    workgroupBarrier();

    current_value = textureLoad(velocity_grid, vec2i(x, y));

    let north_value_1 = textureLoad(velocity_grid, vec2i(x, y - 1));
    let south_value_1 = textureLoad(velocity_grid, vec2i(x, y + 1));
    let east_value_1 = textureLoad(velocity_grid, vec2i(x + 1, y));
    let west_value_1 = textureLoad(velocity_grid, vec2i(x - 1, y));

    let x_velocity = set_boundary(resolution, 1u, x, y, current_value[0], north_value_1[0], south_value_1[0], east_value_1[0], west_value_1[0]);
    let y_velocity = set_boundary(resolution, 2u, x, y, current_value[1], north_value_1[1], south_value_1[1], east_value_1[1], west_value_1[1]);

    color = current_value;
    color[0] = x_velocity;
    color[1] = y_velocity;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x) + 1;
    let y = i32(global_id.y) + 1;

    let resolution = 256u;
    let viscosity = 0.00001f;
    let diffusion = 0.0f;
    let num_iters = 5u;
    let dt = 0.01f;

    let added_vel_x = textureLoad(addition_grid, vec2i(x, y))[0] * dt;
    let added_vel_y = textureLoad(addition_grid, vec2i(x, y))[1] * dt;

    let added_density = textureLoad(addition_grid, vec2i(x, y))[2];

    var current_vel = textureLoad(velocity_grid, vec2i(x, y));
    current_vel[0] += added_vel_x;
    current_vel[1] += added_vel_y;
    textureStore(velocity_grid, vec2i(x, y), current_vel);

    var current_dens = textureLoad(density_grid, vec2i(x, y));
    current_dens[0] += added_density;
    textureStore(velocity_grid, vec2i(x, y), current_vel);


    diffuse_velocity(resolution, 1u, false, true, x, y, viscosity, num_iters, dt);
    workgroupBarrier();
    diffuse_velocity(resolution, 2u, true, true, x, y, viscosity, num_iters, dt);
    workgroupBarrier();

    project(resolution, x, y, num_iters);
    workgroupBarrier();

    advect_velocity(resolution, 1u, false, false, dt, x, y);
    advect_velocity(resolution, 2u, true, false, dt, x, y);
    workgroupBarrier();

    project(resolution, x, y, num_iters);
    workgroupBarrier();

    diffuse_density(resolution, 0u, true, x, y, diffusion, num_iters, dt);
    workgroupBarrier();

    advect_density(resolution, 0u, false, dt, x, y);
    workgroupBarrier();

    let density = textureLoad(velocity_grid, vec2i(x, y));
    let test = sqrt(density.x * density.x + density.y * density.y);
    var color = vec4f(0.0, 0.0, 0.0, 1.0);
    color[0] = test;
    color[1] = test;
    color[2] = test;
    
    textureStore(output_density, vec2i(x, y), color);
    // textureStore(output_density, vec2i(x, y), vec4f(1.0));
}
