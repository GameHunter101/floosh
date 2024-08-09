#![allow(clippy::needless_range_loop)]
use cool_utils::data_structures::ring_buffer::RingBuffer;
use gamezap::new_component;

new_component!(SimulatorComponent {
    resolution: usize,
    viscosity: f32,
    diffusion_coefficient: f32,
    density_grids: RingBuffer<Vec<Vec<f32>>>,
    velocity_x_grids: RingBuffer<Vec<Vec<f32>>>,
    velocity_y_grids: RingBuffer<Vec<Vec<f32>>>
});

impl SimulatorComponent {
    pub fn new(resolution: usize, viscosity: f32, diffusion_coefficient: f32) -> Self {
        let densities = vec![vec![0.0_f32; resolution + 2]; resolution + 2];
        let velocities = vec![vec![0.0_f32; resolution + 2]; resolution + 2];

        SimulatorComponent {
            resolution,
            viscosity,
            diffusion_coefficient,
            density_grids: RingBuffer::new(vec![densities; 2]),
            velocity_x_grids: RingBuffer::new(vec![velocities.clone(); 2]),
            velocity_y_grids: RingBuffer::new(vec![velocities; 2]),
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        }
    }

    fn add_source(resolution: usize, ring_buffer: &mut RingBuffer<Vec<Vec<f32>>>, dt: f32) {
        for i in 0..resolution + 2 {
            for j in 0..resolution + 2 {
                let source_val = ring_buffer[1_i32][i][j] * dt;
                ring_buffer[0_i32][i][j] += source_val;
            }
        }
    }

    fn diffuse(
        resolution: usize,
        diffusion_coefficient: f32,
        grid: &mut RingBuffer<Vec<Vec<f32>>>,
        dt: f32,
        boundary_type: i32,
    ) {
        let final_coefficient = diffusion_coefficient * dt * resolution as f32 * resolution as f32;

        for _ in 0..20 {
            for i in 1..=resolution {
                for j in 1..=resolution {
                    grid[0_i32][i][j] = (grid[1_i32][i][j]
                        + (grid[0_i32][i - 1][j]
                            + grid[0_i32][i + 1][j]
                            + grid[0_i32][i][j - 1]
                            + grid[0_i32][i][j + 1])
                            * final_coefficient)
                        / (1.0 + 4.0 * final_coefficient);
                }
            }

            Self::set_boundary(resolution, &mut grid[0_i32], boundary_type);
        }
    }

    fn advect(
        resolution: usize,
        grid: &mut RingBuffer<Vec<Vec<f32>>>,
        velocity_x: &[Vec<f32>],
        velocity_y: &[Vec<f32>],
        dt: f32,
        boundary_type: i32,
    ) {
        let prev_dt = dt * resolution as f32;

        for i in 1..=resolution {
            for j in 1..=resolution {
                let x = (i as f32 - prev_dt * velocity_x[i][j]).clamp(0.5, resolution as f32 + 0.5);
                let y = (j as f32 - prev_dt * velocity_y[i][j]).clamp(0.5, resolution as f32 + 0.5);

                let i_0 = x as usize;
                let i_1 = i_0 + 1;

                let j_0 = y as usize;
                let j_1 = j_0 + 1;

                let s_1 = x - i_0 as f32;
                let s_0 = 1.0 - s_1;

                let t_1 = y - j_0 as f32;
                let t_0 = 1.0 - t_1;

                grid[0_i32][i][j] = (grid[1_i32][i_0][j_0] * t_0 + grid[1_i32][i_0][j_1] * t_1)
                    * s_0
                    + (grid[1_i32][i_1][j_0] * t_0 + grid[1_i32][i_1][j_1] * t_1) * s_1;
            }
        }

        Self::set_boundary(resolution, &mut grid[0_i32], boundary_type)
    }

    fn set_boundary(resolution: usize, grid: &mut [Vec<f32>], boundary_type: i32) {
        for i in 1..=resolution {
            grid[0][i] = if boundary_type == 1 {
                -grid[1][i]
            } else {
                grid[1][i]
            };
            grid[resolution + 1][i] = if boundary_type == 1 {
                -grid[resolution][i]
            } else {
                grid[resolution][i]
            };

            grid[i][0] = if boundary_type == 2 {
                -grid[i][1]
            } else {
                grid[i][1]
            };
            grid[i][resolution + 1] = if boundary_type == 2 {
                -grid[i][resolution]
            } else {
                grid[i][resolution]
            };
        }

        grid[0][0] = (grid[1][0] + grid[0][1]) * 0.5;
        grid[0][resolution + 1] = (grid[1][resolution + 1] + grid[0][resolution]) * 0.5;
        grid[resolution + 1][0] = (grid[resolution][0] + grid[resolution + 1][1]) * 0.5;
        grid[resolution + 1][resolution + 1] =
            (grid[resolution][resolution + 1] + grid[resolution + 1][resolution]) * 0.5;
    }

    fn project(
        resolution: usize,
        u: &mut [Vec<f32>],
        v: &mut [Vec<f32>],
        p: &mut [Vec<f32>],
        div: &mut [Vec<f32>],
    ) {
        let h = 1.0 / resolution as f32;
        for i in 1..=resolution {
            for j in 1..=resolution {
                div[i][j] = -0.5 * h * (u[i + 1][j] - u[i - 1][j] + v[i][j + 1] - v[i][j - 1]);

                p[i][j] = 0.0;
            }
        }

        Self::set_boundary(resolution, div, 0);
        Self::set_boundary(resolution, p, 0);

        for _ in 0..20 {
            for i in 1..=resolution {
                for j in 1..=resolution {
                    p[i][j] =
                        (div[i][j] + p[i - 1][j] + p[i + 1][j] + p[i][j - 1] + p[i][j + 1]) / 4.0;
                }
            }
            Self::set_boundary(resolution, p, 0);
        }

        for i in 1..=resolution {
            for j in 1..=resolution {
                u[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]) / h;
                v[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]) / h;
            }
        }

        Self::set_boundary(resolution, u, 1);
        Self::set_boundary(resolution, v, 2);
    }

    fn velocity_step(&mut self, dt: f32) {
        Self::add_source(self.resolution, &mut self.velocity_x_grids, dt);
        Self::add_source(self.resolution, &mut self.velocity_y_grids, dt);

        self.velocity_x_grids.rotate_left(1);
        Self::diffuse(
            self.resolution,
            self.viscosity,
            &mut self.velocity_x_grids,
            dt,
            1,
        );
        self.velocity_y_grids.rotate_left(1);
        Self::diffuse(
            self.resolution,
            self.viscosity,
            &mut self.velocity_y_grids,
            dt,
            2,
        );

        let mut old_x_vel = self.velocity_x_grids[1_i32].clone();
        let mut old_y_vel = self.velocity_y_grids[1_i32].clone();

        Self::project(
            self.resolution,
            &mut self.velocity_x_grids[0_i32],
            &mut self.velocity_y_grids[0_i32],
            &mut old_x_vel,
            &mut old_y_vel,
        );

        self.velocity_x_grids[0_i32] = old_x_vel;
        self.velocity_y_grids[0_i32] = old_y_vel;

        self.velocity_x_grids.rotate_left(1);
        self.velocity_y_grids.rotate_left(1);

        let old_x_grid = self.velocity_x_grids[1_i32].clone();
        Self::advect(
            self.resolution,
            &mut self.velocity_x_grids,
            &old_x_grid,
            &self.velocity_y_grids[1_i32],
            dt,
            1,
        );

        let old_y_grid = self.velocity_y_grids[1_i32].clone();
        Self::advect(
            self.resolution,
            &mut self.velocity_y_grids,
            &self.velocity_x_grids[1_i32],
            &old_y_grid,
            dt,
            2,
        );

        let mut old_x_vel = self.velocity_x_grids[1_i32].clone();
        let mut old_y_vel = self.velocity_y_grids[1_i32].clone();

        Self::project(
            self.resolution,
            &mut self.velocity_x_grids[0_i32],
            &mut self.velocity_y_grids[0_i32],
            &mut old_x_vel,
            &mut old_y_vel,
        );

        self.velocity_x_grids[0_i32] = old_x_vel;
        self.velocity_y_grids[0_i32] = old_y_vel;
    }

    fn density_step(&mut self, dt: f32) {
        // println!("{}", self.density_grids[0_i32][50][50]);
        Self::add_source(self.resolution, &mut self.density_grids, dt);
        // println!("{}", self.density_grids[0_i32][50][50]);
        self.density_grids.rotate_left(1);
        Self::diffuse(
            self.resolution,
            self.diffusion_coefficient,
            &mut self.density_grids,
            dt,
            0,
        );
        // println!("{}", self.density_grids[0_i32][50][50]);
        self.density_grids.rotate_left(1);
        Self::advect(
            self.resolution,
            &mut self.density_grids,
            &self.velocity_x_grids[0_i32],
            &self.velocity_y_grids[0_i32],
            dt,
            0,
        );
        // println!("{}", self.density_grids[0_i32][50][50]);
    }

    fn density_grid(&self) -> [[f32; 128]; 128] {
        self.density_grids[0_i32][1..=self.resolution]
            .iter()
            .map(|row| {
                let arr: [f32; 128] = row.clone()[1..=self.resolution]
                    /* .iter()
                    .map(|v| v.magnitude())
                    .collect::<Vec<_>>() */
                    .try_into()
                    .unwrap();
                arr
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn velocity_grid(&self) -> [[f32;128]; 128] {
        (1..=self.resolution).map(|i| {
            let col: [f32; 128] = (1..=self.resolution).map(|j| {
                (self.velocity_x_grids[0_i32][i][j].powi(2) + self.velocity_y_grids[0_i32][i][j].powi(2)).sqrt()
            }).collect::<Vec<_>>().try_into().unwrap();
            col
        }).collect::<Vec<_>>().try_into().unwrap()
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
        for i in 0..10 {
            for j in 0..10 {
                self.density_grids[0_i32][i][50 + j] = 1.0;
            }
        }

        for i in 1..=self.resolution {
            for j in 1..=10 {
                self.velocity_y_grids[0_i32][i][j] = 100000.0;
            }
        }
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
        let dt = details.last_frame_duration.as_secs_f32();

        println!("{}", self.density_grids[0_i32][50][50]);

        /* for i in 0..10 {
            for j in 0..10 {
                self.density_grids[1_i32][50 + i][50 + j] = 1.0;
                // self.velocity_x_grids[0_i32][50 + i][50 + j] = 1.0;
            }
        } */

        for i in 1..=self.resolution {
            for j in 1..=10 {
                self.velocity_y_grids[1_i32][i][j] = 100000.0;
            }
        }

        self.velocity_step(dt);
        self.density_step(dt);

        let materials = materials.unwrap();
        let selected_material = &mut materials.0[materials.1];
        if let Some((_, buffer)) = &selected_material.uniform_buffer_bind_group() {
            let grid_2 = self.density_grid();
            // println!("{:?}", &grid_2[50][20]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[grid_2]));
        }
    }
}
