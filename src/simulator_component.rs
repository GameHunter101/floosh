#![allow(clippy::needless_range_loop)]
use cool_utils::data_structures::ring_buffer::RingBuffer;
use gamezap::new_component;
use nalgebra::Vector2;

new_component!(SimulatorComponent {
    resolution: usize,
    viscosity: f32,
    diffusion_coefficient: f32,
    density_grids: RingBuffer<Vec<Vec<f32>>>,
    velocity_grids: RingBuffer<Vec<Vec<Vector2<f32>>>>
});

impl SimulatorComponent {
    pub fn new(resolution: usize, viscosity: f32, diffusion_coefficient: f32) -> Self {
        let densities = vec![vec![0.0_f32; resolution + 2]; resolution + 2];
        let forces = vec![vec![Vector2::zeros(); resolution + 2]; resolution + 2];

        SimulatorComponent {
            resolution,
            viscosity,
            diffusion_coefficient,
            density_grids: RingBuffer::new(vec![densities; 2]),
            velocity_grids: RingBuffer::new(vec![forces; 2]),
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        }
    }

    fn project(&mut self) {
        let cell_dimension = 1.0 / self.resolution as f32;

        for i in 1..=self.resolution {
            for j in 1..=self.resolution {
                self.velocity_grids[1_i32][i][j].y = -0.5
                    * cell_dimension
                    * (self.velocity_grids[0_i32][i + 1][j].x
                        - self.velocity_grids[0_i32][i - 1][j].x
                        + self.velocity_grids[0_i32][i][j + 1].y
                        - self.velocity_grids[0_i32][i][j - 1].y);

                self.velocity_grids[1_i32][i][j].x = 0.0;
            }
        }

        Self::set_boundary(self.resolution, &mut self.velocity_grids[1_i32], true);

        for _ in 0..20 {
            for i in 1..=self.resolution {
                for j in 1..=self.resolution {
                    self.velocity_grids[1_i32][i][j].x = (self.velocity_grids[1_i32][i][j].y
                        + self.velocity_grids[1_i32][i - 1][j].x
                        + self.velocity_grids[1_i32][i + 1][j].x
                        + self.velocity_grids[1_i32][i][j - 1].x
                        + self.velocity_grids[1_i32][i][j + 1].x)
                        / 4.0;
                }
            }
            Self::set_boundary(self.resolution, &mut self.velocity_grids[1_i32], true);
        }

        for i in 1..=self.resolution {
            for j in 1..=self.resolution {
                self.velocity_grids[0_i32][i][j].x -= 0.5
                    * (self.velocity_grids[1_i32][i + 1][j].x
                        - self.velocity_grids[1_i32][i - 1][j].x)
                    / cell_dimension;
                self.velocity_grids[0_i32][i][j].y -= 0.5
                    * (self.velocity_grids[1_i32][i][j + 1].x
                        - self.velocity_grids[1_i32][i][j - 1].x)
                    / cell_dimension;
            }
        }

        Self::set_boundary(self.resolution, &mut self.velocity_grids[0_i32], false);
    }

    fn velocity_step(&mut self, dt: f32) {
        Self::add_source(self.resolution, &mut self.velocity_grids, dt);
        self.velocity_grids.rotate_left(1);
        Self::diffuse(
            self.resolution,
            self.viscosity,
            &mut self.velocity_grids,
            dt,
            true,
        );
        self.project();
        self.velocity_grids.rotate_left(1);
        let old_vels = self.velocity_grids[1_i32].clone();
        Self::advect(
            self.resolution,
            &mut self.velocity_grids,
            &old_vels,
            dt,
            false,
        );
        self.project();
    }

    fn add_source<
        T: std::fmt::Debug + Clone + Copy + std::ops::Mul<f32, Output = T> + std::ops::AddAssign,
    >(
        resolution: usize,
        ring_buffer: &mut RingBuffer<Vec<Vec<T>>>,
        dt: f32,
    ) {
        for i in 0..resolution + 2 {
            for j in 0..resolution + 2 {
                let source_val = ring_buffer[1_i32][i][j] * dt;
                ring_buffer[0_i32][i][j] += source_val;
            }
        }
    }

    fn diffuse<
        T: std::fmt::Debug
            + Clone
            + Copy
            + std::ops::Mul<f32, Output = T>
            + std::ops::AddAssign
            + std::ops::Add<T, Output = T>
            + std::ops::Div<f32, Output = T>,
    >(
        resolution: usize,
        diffusion_coefficient: f32,
        grid: &mut RingBuffer<Vec<Vec<T>>>,
        dt: f32,
        is_standard: bool,
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

            Self::set_boundary(resolution, &mut grid[0_i32], is_standard);
        }
    }

    fn advect<
        T: std::fmt::Debug
            + Clone
            + Copy
            + std::ops::Mul<f32, Output = T>
            + std::ops::AddAssign
            + std::ops::Add<T, Output = T>
            + std::ops::Div<f32, Output = T>,
    >(
        resolution: usize,
        grid: &mut RingBuffer<Vec<Vec<T>>>,
        velocity: &[Vec<Vector2<f32>>],
        dt: f32,
        is_standard: bool,
    ) {
        let prev_dt = dt * resolution as f32;

        for i in 1..=resolution {
            for j in 1..=resolution {
                let mut vector = Vector2::new(
                    i as f32 - prev_dt * velocity[i][j].x,
                    j as f32 - prev_dt * velocity[i][j].y,
                );

                vector.x = vector.x.clamp(0.5, resolution as f32 + 0.5);
                let i_0 = vector.x as usize;
                let i_1 = i_0 + 1;

                vector.y = vector.y.clamp(0.5, resolution as f32 + 0.5);
                let j_0 = vector.y as usize;
                let j_1 = j_0 + 1;

                let s_1 = vector.x - i_0 as f32;
                let s_0 = 1.0 - s_1;

                let t_1 = vector.y - j_0 as f32;
                let t_0 = 1.0 - t_1;

                grid[0_i32][i][j] = (grid[1_i32][i_0][j_0] * t_0 + grid[1_i32][i_0][j_1] * t_1)
                    * s_0
                    + (grid[1_i32][i_1][j_0] * t_0 + grid[1_i32][i_1][j_1] * t_1) * s_1;
            }
        }

        Self::set_boundary(resolution, &mut grid[0_i32], is_standard)
    }

    fn set_boundary<
        T: std::fmt::Debug
            + Clone
            + Copy
            + std::ops::Mul<f32, Output = T>
            + std::ops::AddAssign
            + std::ops::Add<T, Output = T>,
    >(
        resolution: usize,
        grid: &mut [Vec<T>],
        is_standard: bool,
    ) {
        for i in 1..=resolution {
            grid[0][i] = if is_standard {
                grid[1][i]
            } else {
                grid[1][i] * -1.0
            };
            grid[resolution + 1][i] = if is_standard {
                grid[resolution][i]
            } else {
                grid[resolution][i] * -1.0
            };

            grid[i][0] = if is_standard {
                grid[i][1]
            } else {
                grid[i][1] * -1.0
            };
            grid[i][resolution + 1] = if is_standard {
                grid[i][resolution]
            } else {
                grid[i][resolution] * -1.0
            };
        }

        grid[0][0] = (grid[1][0] + grid[0][1]) * 0.5;
        grid[0][resolution + 1] = (grid[1][resolution + 1] + grid[0][resolution]) * 0.5;
        grid[resolution + 1][0] = (grid[resolution][0] + grid[resolution + 1][1]) * 0.5;
        grid[resolution + 1][resolution + 1] =
            (grid[resolution][resolution + 1] + grid[resolution + 1][resolution]) * 0.5;
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
            true,
        );
        // println!("{}", self.density_grids[0_i32][50][50]);
        self.density_grids.rotate_left(1);
        Self::advect(
            self.resolution,
            &mut self.density_grids,
            &self.velocity_grids[0_i32],
            dt,
            true,
        );
        // println!("{}", self.density_grids[0_i32][50][50]);
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
                self.density_grids[0_i32][50 + i][50 + j] = 1.0;
                self.velocity_grids[0_i32][50 + i][50 + j] = Vector2::new(1.0, 0.0);
            }
        }

        /* for i in 1..50 {
            for j in 1..=self.resolution {
                self.velocity_grids[0_i32][i][j] = Vector2::new(1.0, 0.0);
            }
        } */
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

        /* for i in 0..10 {
            for j in 0..10 {
                self.density_grids[0_i32][50 + i][50 + j] = 1.0;
            }
        } */

        for i in 1..50 {
            for j in 1..=self.resolution {
                self.velocity_grids[0_i32][i][j] = Vector2::new(1.0, 0.0);
            }
        }

        self.velocity_step(dt);
        self.density_step(dt);

        let materials = materials.unwrap();
        let selected_material = &mut materials.0[materials.1];
        if let Some((_, buffer)) = &selected_material.uniform_buffer_bind_group() {
            let grid_2: [[f32; 128]; 128] = self.density_grids[0_i32][1..=self.resolution]
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
                .unwrap();
            // println!("{:?}", &grid_2[20][20]);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[grid_2]));
        }
    }
}
