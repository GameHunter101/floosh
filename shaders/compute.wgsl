// x    : 0 | r
// y    : 1 | g
// x_0  : 2 | b
// y_0  : 3 | a
@group(0) @binding(0) var velocity_grid: texture_storage_2d<rgba8unorm, read_write>;
// pressure     : 0 | r
// pressure_1   : 1 | g
// divergence   : 2 | b
// divergence_1 : 3 | a
@group(0) @binding(1) var pressure_grid: texture_storage_2d<rgba8unorm, read_write>;

struct AddedVelocities {
    values: array<VelocityItems>,
};

struct VelocityItems{
    vel: vec2<f32>,
    pos: vec2<f32>,
}

struct AddedDensities {
    values: array<DensityItems>,
};

struct DensityItems{
    dens: f32,
    pad: f32,
    pos: vec2<f32>,
}

@group(0) @binding(2) var<storage> added_velocities: AddedVelocities;
@group(0) @binding(3) var<storage> added_densities: AddedDensities;

// new x velocity   : 0 | r
// new y velocity   : 1 | g
// new density      : 2 | b
@group(0) @binding(4) var output_density: texture_storage_2d<rgba8unorm, read_write>;

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

fn get_neighbors_density(x: i32, y: i32) -> array<vec2<f32>, 4> {
    return array(
        textureLoad(output_density, vec2i(x, y - 1)).xy,
        textureLoad(output_density, vec2i(x, y + 1)).xy,
        textureLoad(output_density, vec2i(x + 1, y)).xy,
        textureLoad(output_density, vec2i(x - 1, y)).xy,
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
        let current_velocity = textureLoad(velocity_grid, vec2i(x, y));

        let res_2 = jacobi_2(
            resolution,
            alpha,
            beta,
            current_velocity.xy,
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

fn diffuse_density(
    resolution: u32,
    num_iters: i32,
    dt: f32,
    x: i32,
    y: i32,
    diffusion: f32,
) {
    let neighbors = get_neighbors_density(x, y);
    let res_f = 1.0 / f32(resolution);
    let alpha = (res_f * res_f) / (diffusion * dt);
    let beta = 4.0 + alpha;
    for (var i = 0i; i < num_iters; i++) {
        workgroupBarrier();
        let density = textureLoad(output_density, vec2i(x, y));
        let res = jacobi(
            resolution,
            alpha,
            beta,
            density[0],
            neighbors[0][1],
            neighbors[1][1],
            neighbors[2][1],
            neighbors[3][1]
        );

        var new_density = textureLoad(output_density, vec2i(x, y));
        new_density[0] = res;

        textureStore(output_density, vec2i(x, y), new_density);

        /* workgroupBarrier();

        let new_neighbors = get_neighbors_density(x, y);
        let current_density = textureLoad(output_density, vec2i(x, y));

        let res_2 = jacobi(
            resolution,
            alpha,
            beta,
            current_density[1],
            new_neighbors[0][0],
            new_neighbors[1][0],
            new_neighbors[2][0],
            new_neighbors[3][0]
        );

        new_density = textureLoad(output_density, vec2i(x, y));
        new_density[1] = res_2; */
    }
}

fn advect_velocity(
    resolution: u32,
    dt: f32,
    x: i32,
    y: i32,
) -> vec2<f32> {
    let velocity = textureLoad(velocity_grid, vec2i(x, y)).zw;

    var last_position = vec2f(f32(x), f32(y)) - velocity * dt * 4.0 * f32(resolution);

    last_position.x = clamp(last_position.x, 0.5, f32(resolution) + 0.5);
    last_position.y = clamp(last_position.y, 0.5, f32(resolution) + 0.5);

    let north_west_value = textureLoad(velocity_grid, vec2i(i32(last_position.x), i32(last_position.y))).zw;
    let north_east_value = textureLoad(velocity_grid, vec2i(i32(last_position.x) + 1, i32(last_position.y))).zw;
    let south_west_value = textureLoad(velocity_grid, vec2i(i32(last_position.x), i32(last_position.y) + 1)).zw;
    let south_east_value = textureLoad(velocity_grid, vec2i(i32(last_position.x) + 1, i32(last_position.y) + 1)).zw;

    let north_lerp = mix(north_west_value, north_east_value, fract(last_position.x));
    let south_lerp = mix(south_west_value, south_east_value, fract(last_position.x));

    let final_value = mix(north_lerp, south_lerp, fract(last_position.y));
    return final_value;
}

fn advect_density(
    resolution: u32,
    dt: f32,
    x: i32,
    y: i32,
) -> f32 {
    // let velocity = textureLoad(velocity_grid, vec2i(x, y)).xy;
    let velocity = advect_velocity(resolution, dt, x, y);

    let last_position = vec2f(f32(x), f32(y)) - velocity * dt * 4.0 * f32(resolution);

    let north_west_value = textureLoad(output_density, vec2i(i32(last_position.x), i32(last_position.y)))[0];
    let north_east_value = textureLoad(output_density, vec2i(i32(last_position.x) + 1, i32(last_position.y)))[0];
    let south_west_value = textureLoad(output_density, vec2i(i32(last_position.x), i32(last_position.y) + 1))[0];
    let south_east_value = textureLoad(output_density, vec2i(i32(last_position.x) + 1, i32(last_position.y) + 1))[0];

    let north_lerp = mix(north_west_value, north_east_value, fract(last_position.x));
    let south_lerp = mix(south_west_value, south_east_value, fract(last_position.x));

    let final_value = mix(north_lerp, south_lerp, fract(last_position.y));
    return final_value;
}

fn divergence(resolution: u32, x: i32, y: i32) -> f32 {
    let vel_neighbors = get_neighbors_velocity(x, y);

    let div = 0.5 * 4.0 * f32(resolution) * ((vel_neighbors[0][3] - vel_neighbors[1][3]) + (vel_neighbors[2][2] - vel_neighbors[3][2]));

    return div;
}

fn gradient_subtract(resolution: u32, x: i32, y: i32) -> vec2<f32> {
    let pressure_neighbors = get_neighbors_pressure(x, y);

    let old_velocity = textureLoad(pressure_grid, vec2i(x, y));

    var velocity = textureLoad(velocity_grid, vec2i(x, y));

    let new_velocity_x = old_velocity[2] - (pressure_neighbors[2][0] - pressure_neighbors[3][0]) / 2.0 * 4.0 * f32(resolution);
    let new_velocity_y = old_velocity[3] - (pressure_neighbors[0][0] - pressure_neighbors[1][0]) / 2.0 * 4.0 * f32(resolution);

    let gradient_vec = vec2f(new_velocity_x, new_velocity_y);

    return velocity.zw - gradient_vec;
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

fn calc_vel_splat(x: i32, y: i32, dt: f32, radius: f32) -> vec2<f32> {
    var accumulation = vec2f(0.0);
    for (var i = 0; i < i32(arrayLength(&added_velocities.values)); i++) {
        let new_vec = vec2f(f32(x), f32(y)) - added_velocities.values[i].pos;
        accumulation += added_velocities.values[i].vel * dt * vec2f(exp(-1.0 * dot(new_vec, new_vec) / radius));
    }

    return accumulation;
}

fn calc_dens_splat(x: i32, y: i32, radius: f32) -> f32 {
    var accumulation = 0.0;
    for (var i = 0; i < i32(arrayLength(&added_densities.values)); i++) {
        let new_vec = vec2f(f32(x), f32(y)) - added_densities.values[i].pos;
        accumulation += added_densities.values[i].dens * exp(-1.0 * dot(new_vec, new_vec) / radius);
    }

    return accumulation;
}

fn set_boundary(resolution: u32, x: i32, y: i32) -> vec2<f32> {
    if x == 0 && y == 0
        || x == 0 && y == i32(resolution) + 1
        || x == i32(resolution) + 1 && y == 0
        || x == i32(resolution) + 1 && y == i32(resolution) + 1
    {
        return vec2f(0.0, 0.0);
    }

    if x == 0 {
        let vel = textureLoad(velocity_grid, vec2i(x + 1, y)).xy;

        return vel * vec2f(-1.0, 1.0);
    } else if x == i32(resolution) + 1 {
        let vel = textureLoad(velocity_grid, vec2i(x - 1, y)).xy;
        
        return vel * vec2f(-1.0, 1.0);
    }

    if y == 0 {
        let vel = textureLoad(velocity_grid, vec2i(x, y + 1)).xy;

        return vel * vec2f(1.0, -1.0);
    } else if y == i32(resolution) + 1 {
        let vel = textureLoad(velocity_grid, vec2i(x, y - 1)).xy;

        return vel * vec2f(1.0, -1.0);
    }

    return textureLoad(velocity_grid, vec2i(x, y)).xy;
}

fn set_boundary_divergence(resolution: u32, x: i32, y: i32) -> f32 {
    if x == 0 && y == 0
        || x == 0 && y == i32(resolution) + 1
        || x == i32(resolution) + 1 && y == 0
        || x == i32(resolution) + 1 && y == i32(resolution) + 1
    {
        return 0.0;
    }

    if x == 0 {
        return textureLoad(pressure_grid, vec2i(x + 1, y)).w;
    } else if x == i32(resolution) + 1 {
        return textureLoad(pressure_grid, vec2i(x - 1, y)).w;
    }

    if y == 0 {
        return textureLoad(pressure_grid, vec2i(x, y + 1)).w;
    } else if y == i32(resolution) + 1 {
        return textureLoad(pressure_grid, vec2i(x, y - 1)).w;
    }

    return textureLoad(pressure_grid, vec2i(x, y)).w;
}

fn set_boundary_pressure(resolution: u32, x: i32, y: i32) -> f32 {
    if x == 0 && y == 0
        || x == 0 && y == i32(resolution) + 1
        || x == i32(resolution) + 1 && y == 0
        || x == i32(resolution) + 1 && y == i32(resolution) + 1
    {
        return 0.0;
    }

    if x == 0 {
        return textureLoad(pressure_grid, vec2i(x + 1, y)).y;
    } else if x == i32(resolution) + 1 {
        return textureLoad(pressure_grid, vec2i(x - 1, y)).y;
    }

    if y == 0 {
        return textureLoad(pressure_grid, vec2i(x, y + 1)).y;
    } else if y == i32(resolution) + 1 {
        return textureLoad(pressure_grid, vec2i(x, y - 1)).y;
    }

    return textureLoad(pressure_grid, vec2i(x, y)).y;
}

fn clear_pressure(viscosity: f32, x: i32, y: i32) -> f32 {
    return textureLoad(pressure_grid, vec2i(x, y))[0] * viscosity;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x) + 1;
    let y = i32(global_id.y) + 1;


    let resolution = 256u;
    let viscosity = 0.000001f;
    let diffusion = 0.95f;
    let velocity_diffusion = 0.95f;
    let num_iters = 30i;
    let dt = 0.1f;
    let radius = 35.0;

    let dens_addition = calc_dens_splat(x, y, radius);
    var last_dens = textureLoad(output_density, vec2i(x, y));
    let final_dens = max(0.0, last_dens[0] * diffusion + dens_addition);
    last_dens[0] = final_dens;
    textureStore(output_density, vec2i(x, y), last_dens);

    let vel_addition = calc_vel_splat(x, y, dt, radius);
    var old_vel = textureLoad(velocity_grid, vec2i(x, y));
    let new_vel = old_vel.xy * velocity_diffusion + vel_addition;
    old_vel[2] = new_vel[0];
    old_vel[3] = new_vel[1];
    textureStore(velocity_grid, vec2i(x, y), old_vel);

    let advected_velocity = advect_velocity(resolution, dt, x, y);

    old_vel[0] = advected_velocity[0];
    old_vel[1] = advected_velocity[1];
    textureStore(velocity_grid, vec2i(x, y), old_vel);

    let boundary_correction = set_boundary(resolution, x, y);
    
    old_vel[2] = boundary_correction[0];
    old_vel[3] = boundary_correction[1];

    var old_pressure = textureLoad(pressure_grid, vec2i(x, y));
    let calc_divergence = divergence(resolution, x, y);
    old_pressure[3] = calc_divergence;
    textureStore(pressure_grid, vec2i(x, y), old_pressure);
    workgroupBarrier();

    old_pressure[2] = set_boundary_divergence(resolution, x, y);
    textureStore(pressure_grid, vec2i(x, y), old_pressure);


    for (var i = 0; i < num_iters; i++) {
        workgroupBarrier();
        let pressure_neighbors = get_neighbors_pressure(x, y);
        old_pressure = textureLoad(pressure_grid, vec2i(x, y));

        let res = jacobi(
            resolution,
            -1.0 / (f32(resolution) * f32(resolution) * 16.0),
            4.0,
            old_pressure.z,
            pressure_neighbors[0].x,
            pressure_neighbors[1].x,
            pressure_neighbors[2].x,
            pressure_neighbors[3].x,
        );

        old_pressure[1] = res;
        textureStore(pressure_grid, vec2i(x, y), old_pressure);

        workgroupBarrier();

        old_pressure[0] = set_boundary_pressure(resolution, x, y);
        textureStore(pressure_grid, vec2i(x, y), old_pressure);

    }

    var current_velocity = textureLoad(velocity_grid, vec2i(x, y));
    let new_velocity = gradient_subtract(resolution, x, y);
    current_velocity[0] = new_velocity[0];
    current_velocity[1] = new_velocity[1];

    textureStore(velocity_grid, vec2i(x, y), current_velocity);

    var current_pressure = textureLoad(pressure_grid, vec2i(x, y));
    current_pressure[1] = clear_pressure(viscosity, x, y);
    textureStore(pressure_grid, vec2i(x, y), current_pressure);

    let advected_density = advect_density(resolution, dt, x, y);
    textureStore(output_density, vec2i(x, y), vec4f(vec3f(advected_density), 1.0));


    /* // if x != 0 && y != 0 && x != i32(resolution) + 1 && y != i32(resolution) + 1 {

        let added_vel_x = textureLoad(addition_grid, vec2i(x, y))[0] * dt;
        let added_vel_y = textureLoad(addition_grid, vec2i(x, y))[1] * dt;

        let added_density = textureLoad(addition_grid, vec2i(x, y))[2];

        var current_dens = textureLoad(output_density, vec2i(x, y));
        current_dens[0] += added_density;
        textureStore(output_density, vec2i(x, y), current_dens);

        workgroupBarrier();
        project(resolution, x, y, num_iters * 2);

        workgroupBarrier();
        let advected_velocity = advect_velocity(resolution, dt, x, y);

        textureStore(velocity_grid, vec2i(x, y), vec4f(advected_velocity, advected_velocity));

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

        /* let pressure = textureLoad(pressure_grid, vec2i(x, y))[0];

        textureStore(output_density, vec2i(x - 1, y - 1), vec4f(vec3f(pressure), 1.0)); */

        diffuse_density(resolution, num_iters, dt, x - 1, y - 1, diffusion);

        // let advected_density = advect_density(resolution, dt, x - 1, y - 1);
        let advected_density = textureLoad(output_density, vec2i(x - 1, y - 1))[1];
        let vel = textureLoad(velocity_grid, vec2i(x, y)).xy;

        textureStore(output_density, vec2i(x - 1, y - 1), vec4f(vec3f(advected_density), 1.0));
    /* } else {
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
    } */
    */
}
