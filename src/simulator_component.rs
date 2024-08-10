#![allow(clippy::needless_range_loop)]
use cool_utils::data_structures::ring_buffer::RingBuffer;
use gamezap::new_component;
use nalgebra::Vector2;

static OVERRELAXATION_FACTOR: f32 = 1.9;

#[derive(Debug, Default, Clone, Copy)]
pub struct CellData {
    density: f32,
    pressure: f32,
    is_fluid: i32,
}

impl CellData {}

new_component!(SimulatorComponent {
    resolution: usize,
    diffusion_coefficient: f32,
    fluid_density: f32,
    grid_u: Vec<Vec<f32>>,
    grid_v: Vec<Vec<f32>>,
    density_grid: Vec<Vec<f32>>,
    pressure_grid: Vec<Vec<f32>>,
    barrier_grid: Vec<Vec<f32>>,
    cell_size: f32
});

impl SimulatorComponent {
    pub fn new(resolution: usize, fluid_density: f32, diffusion_coefficient: f32) -> Self {
        let zero_grid = vec![vec![0.0; resolution + 2]; resolution + 2];
        let barrier_grid = (0..resolution + 2)
            .map(|i| {
                (0..resolution + 2)
                    .map(|j| {
                        if i == 0 || i == resolution + 1 || j == 0 || j == resolution + 1 {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        SimulatorComponent {
            resolution,
            diffusion_coefficient,
            fluid_density,
            grid_u: zero_grid.clone(),
            grid_v: zero_grid.clone(),
            density_grid: zero_grid.clone(),
            pressure_grid: zero_grid,
            barrier_grid,
            cell_size: 1.0 / resolution as f32,
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        }
    }

    fn add_forces(&mut self, forces: Vec<Vec<Vector2<f32>>>, dt: f32) {
        for i in 1..=self.resolution {
            for j in 1..=self.resolution {
                /* let force = forces[i][j];

                self.staggered_grid[0_i32][i][j].right_velocity += force.x * dt / 2.0;
                self.staggered_grid[0_i32][i][j].left_velocity += force.x * dt / 2.0;

                self.staggered_grid[0_i32][i][j].top_velocity += force.y * dt / 2.0;
                self.staggered_grid[0_i32][i][j].bottom_velocity += force.y * dt / 2.0; */
            }
        }
    }

    fn solve_divergence(&mut self, dt: f32) {
        for _ in 0..20 {
            for i in 1..=self.resolution {
                for j in 1..=self.resolution {
                    let divergence = OVERRELAXATION_FACTOR
                        * (self.grid_v[i][j + 1] - self.grid_v[i][j] + self.grid_u[i + 1][j]
                            - self.grid_u[i][j]);

                    let surrounding_tiles = self.barrier_grid[i + 1][j]
                        + self.barrier_grid[i - 1][j]
                        + self.barrier_grid[i][j + 1]
                        + self.barrier_grid[i][j - 1];

                    if surrounding_tiles == 0.0 {
                        continue;
                    }

                    self.pressure_grid[i][j] +=
                        divergence / surrounding_tiles * self.fluid_density * self.cell_size / dt;
                }
            }
        }
    }

    fn advect(resolution: usize, grid: &mut [Vec<f32>], dt: f32, u: &[Vec<f32>], v: &[Vec<f32>]) {
        for i in 1..=resolution {
            for j in 1..=resolution {
                let last_x = i as f32 - u[i][j] * dt;
                let last_y =
                    j as f32 - (v[i - 1][j] + v[i - 1][j + 1] + v[i][j] + v[i][j + 1]) * dt / 4.0;

                let top_left_value = grid[last_x as usize][last_y as usize + 1];
                let top_right_value = grid[last_x as usize + 1][last_y as usize + 1];
                let bottom_left_value = grid[last_x as usize][last_y as usize];
                let bottom_right_value = grid[last_x as usize + 1][last_y as usize];

                let top_lerp = Self::lerp(top_left_value, top_right_value, last_x.fract());
                let bottom_lerp = Self::lerp(bottom_left_value, bottom_right_value, last_x.fract());

                let final_lerp_value = Self::lerp(bottom_lerp, top_lerp, last_y.fract());

                grid[i][j] = final_lerp_value;
            }
        }
    }

    fn lerp(a: f32, b: f32, k: f32) -> f32 {
        a + k * (b - a)
    }

    fn simulate(&mut self, dt: f32) {
        self.solve_divergence(dt);

        let mut grid_u = self.grid_u.clone();
        Self::advect(self.resolution, &mut grid_u, dt, &self.grid_u, &self.grid_v);
        self.grid_u = grid_u;

        let mut grid_v = self.grid_v.clone();
        Self::advect(self.resolution, &mut grid_v, dt, &self.grid_u, &self.grid_v);
        self.grid_u = grid_v;

        Self::advect(
            self.resolution,
            &mut self.density_grid,
            dt,
            &self.grid_u,
            &self.grid_v,
        );
    }

    fn density_grid(&self) -> [[f32; 128]; 128] {
        self.grid_u[1..=self.resolution]
            .iter()
            .map(|row| {
                let arr: [f32; 128] = row.clone()[1..=self.resolution]
                    .try_into()
                    .unwrap();
                arr
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

impl ComponentSystem for SimulatorComponent {
    fn initialize(
        &mut self,
        _device: Arc<Device>,
        _queue: Arc<Queue>,
        _component_map: &AllComponents,
        _concept_manager: Rc<Mutex<ConceptManager>>,
        _engine_details: Option<Rc<Mutex<EngineDetails>>>,
        _engine_systems: Option<Rc<Mutex<EngineSystems>>>,
        _ui_manager: Rc<Mutex<gamezap::ui_manager::UiManager>>,
    ) {
        for i in 50..60 {
            for j in 50..60 {
                self.density_grid[i][j] = 1.0;
                self.grid_u[i][j] = 1.0;
            }
        }

        // self.add_forces(forces, dt)
    }

    fn update(
        &mut self,
        _device: Arc<Device>,
        queue: Arc<Queue>,
        _component_map: &mut AllComponents,
        engine_details: Rc<Mutex<EngineDetails>>,
        _engine_systems: Rc<Mutex<EngineSystems>>,
        _concept_manager: Rc<Mutex<ConceptManager>>,
        _active_camera_id: Option<EntityId>,
        _entities: &mut Vec<Entity>,
        materials: Option<&mut (Vec<Material>, usize)>,
        _compute_pipelines: &[ComputePipeline],
    ) {
        let details = engine_details.lock().unwrap();
        let dt = details.last_frame_duration.as_secs_f32() * 100.0;

        self.simulate(dt);

        let materials = materials.unwrap();
        let selected_material = &mut materials.0[materials.1];
        if let Some((_, buffer)) = &selected_material.uniform_buffer_bind_group() {
            let grid_2 = self.density_grid();
            println!("{:?}", &grid_2[50][50]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[grid_2]));
        }
    }
}
