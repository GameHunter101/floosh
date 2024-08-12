#![allow(clippy::needless_range_loop)]
use cool_utils::data_structures::ring_buffer::RingBuffer;
use gamezap::new_component;
use nalgebra::Vector2;
use nalgebra_sparse::CsrMatrix;

static OVERRELAXATION_FACTOR: f32 = 1.9;

#[derive(Debug, Default, Clone, Copy)]
pub struct CellData {
    pressure: f32,
    is_fluid: i32,
}

impl CellData {}

new_component!(SimulatorComponent {
    resolution: usize,
    fluid_density: f32,
    grid_u: RingBuffer<Vec<Vec<f32>>>,
    grid_v: RingBuffer<Vec<Vec<f32>>>,
    // grid_contents: Vec<Vec<CellData>>,
    pressure_grid: Vec<Vec<f32>>,
    pressure_divergence_mat: CsrMatrix<f32>,
    cell_size: f32
});

impl SimulatorComponent {
    pub fn new(resolution: usize, fluid_density: f32) -> Self {
        let grid_u = vec![vec![0.0; resolution + 2]; resolution + 2];
        let grid_v = vec![vec![0.0; resolution + 2]; resolution + 2];

        /* let grid_contents = (0..resolution + 2)
        .map(|i| {
            (0..resolution + 2)
                .map(|j| {
                    if i == 0 || i == resolution + 1 || j == 0 || j == resolution + 1 {
                        CellData::default()
                    } else {
                        CellData {
                            pressure: 0.0,
                            is_fluid: 1.0,
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>(); */

        let pressure_grid = vec![vec![0.0; resolution]; resolution];
        let pressure_divergence_mat = Self::calculate_cell_pressure_divergence(resolution);

        SimulatorComponent {
            resolution,
            fluid_density,
            grid_u: RingBuffer::new(vec![grid_u; 2]),
            grid_v: RingBuffer::new(vec![grid_v; 2]),
            pressure_grid,
            pressure_divergence_mat,
            cell_size: 1.0 / resolution as f32,
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        }
    }

    fn calculate_cell_pressure_divergence(resolution: usize) -> CsrMatrix<f32> {
        let mut values = Vec::with_capacity(
            5 * (resolution - 2) * (resolution - 2) + 4 * 3 + 4 * (resolution - 2) * 4,
        );
        let mut column_indices = Vec::with_capacity(
            5 * (resolution - 2) * (resolution - 2) + 4 * 3 + 4 * (resolution - 2) * 4,
        );
        let mut row_indices = Vec::with_capacity(resolution * resolution + 1);
        // dbg!(row_indices.capacity());
        row_indices.push(0);

        for i in 0..resolution {
            for j in 0..resolution {
                let cell_count = i * resolution + j;
                let mut neighbors = Vec::new();

                if i > 0 {
                    neighbors.push((i - 1) * resolution + j);
                }
                if i + 1 < resolution {
                    neighbors.push((i + 1) * resolution + j);
                }
                if j > 0 {
                    neighbors.push(i * resolution + (j - 1));
                }
                if j + 1 < resolution {
                    neighbors.push(i * resolution + (j + 1));
                }

                neighbors.push(cell_count);
                neighbors.sort();

                for neighbor in &neighbors {
                    if *neighbor == cell_count {
                        values.push(neighbors.len() as f32);
                    } else {
                        values.push(1.0_f32);
                    }
                    column_indices.push(*neighbor);
                }

                // values.push(neighbors.len() as f32);
                // column_indices.push(cell_count);
                row_indices.push(values.len());
            }
        }
        // row_indices.push(values.len());
        // dbg!(row_indices.len());

        CsrMatrix::try_from_csr_data(
            resolution * resolution,
            resolution * resolution,
            row_indices,
            column_indices,
            values,
        )
        .unwrap()
    }

    fn lerp(a: f32, b: f32, k: f32) -> f32 {
        a + k * (b - a)
    }

    fn advect(
        resolution: usize,
        dt: f32,
        grid: &mut RingBuffer<Vec<Vec<f32>>>,
        u: &[Vec<f32>],
        v: &[Vec<f32>],
    ) {
        for i in 1..=resolution {
            for j in 1..=resolution {
                let prev_x = (i as f32 - dt * u[i][j]).clamp(0.5, resolution as f32 + 0.5);
                let prev_y = (j as f32
                    - dt * (v[i][j - 1] + v[i][j] + v[i + 1][j - 1] + v[i + 1][j]) / 4.0)
                    .clamp(0.5, resolution as f32 + 0.5);

                let top_left_value = grid[0_i32][prev_x as usize][prev_y as usize];
                let top_right_value = grid[0_i32][prev_x as usize + 1][prev_y as usize];

                let bottom_left_value = grid[0_i32][prev_x as usize + 1][prev_y as usize];
                let bottom_right_value = grid[0_i32][prev_x as usize + 1][prev_y as usize + 1];

                let top_lerp = Self::lerp(top_left_value, top_right_value, prev_x.fract());
                let bottom_lerp = Self::lerp(bottom_left_value, bottom_right_value, prev_x.fract());

                let final_value = Self::lerp(top_lerp, bottom_lerp, prev_y.fract());

                grid[1_i32][i][j] = final_value;
            }
        }
    }

    fn project(
        resolution: usize,
        dt: f32,
        u: &mut [Vec<f32>],
        v: &mut [Vec<f32>],
        pressure_grid: &mut [Vec<f32>],
        pressure_divergence_mat: &CsrMatrix<f32>,
    ) {
        let mut divergence_vector = Vec::with_capacity(resolution * resolution); // Vec::with_capacity(resolution * resolution);

        for i in 1..=resolution {
            for j in 1..=resolution {
                if i == 1 {
                    if j == 1 {
                        divergence_vector.push(u[i][j + 1] + v[i + 1][j]);
                        continue;
                    }
                    if j == resolution {
                        divergence_vector.push(u[i][j] - v[i + 1][j]);
                        continue;
                    }
                    divergence_vector.push(u[i][j + 1] - u[i][j] + v[i + 1][j]);
                    continue;
                }
                if i == resolution {
                    if j == 1 {
                        divergence_vector.push(u[i][j + 1] - v[i][j]);
                        continue;
                    }
                    if j == resolution {
                        divergence_vector.push(u[i][j] + v[i][j]);
                        continue;
                    }
                    divergence_vector.push(u[i][j + 1] - u[i][j] + v[i][j]);
                    continue;
                }
                if j == 1 {
                    divergence_vector.push(u[i][j + 1] + v[i][j] - v[i + 1][j]);
                    continue;
                }
                if j == resolution {
                    divergence_vector.push(u[i][j] + v[i][j] - v[i + 1][j]);
                    continue;
                }

                divergence_vector.push(u[i][j] - u[i][j + 1] + v[i][j] - v[i + 1][j]);
            }
        }

        let divergence_vector = nalgebra::DVector::from_column_slice(&divergence_vector);

        let mut estimate = nalgebra::DVector::<f32>::zeros(divergence_vector.len());

        let mut residual = divergence_vector.clone();

        let mut direction = residual.clone();

        let mut error_squared_magnitude = residual.magnitude_squared();

        for k in 0..20 {
            if error_squared_magnitude < 0.01 {
                break;
            }
            let dir_clone = direction.clone();

            let ap = pressure_divergence_mat * &direction;
            let alpha = error_squared_magnitude / (direction.dot(&ap));

            estimate += alpha * dir_clone;
            residual -= alpha * ap;

            let new_error_squared_magnitude = residual.magnitude_squared();

            let beta = new_error_squared_magnitude / error_squared_magnitude;

            error_squared_magnitude = new_error_squared_magnitude;

            direction *= beta;

            direction += &residual;
        }

        for (cell_id, val) in estimate.iter().enumerate() {
            let i = cell_id / resolution;
            let j = cell_id % resolution;
            pressure_grid[i][j] = *val;
        }

        for i in 1..=resolution {
            for j in 1..=resolution {
                let cell_id = (i - 1) * resolution + (j - 1);
                let div = divergence_vector[cell_id];

                if i == 1 {
                    if j == 1 {
                        u[i][j + 1] -= div / 2.0;
                        v[i + 1][j] -= div / 2.0;
                        continue;
                    }

                    if j == resolution {
                        u[i][j] += div / 2.0;
                        v[i + 1][j] -= div / 2.0;
                        continue;
                    }

                    u[i][j] += div / 3.0;
                    u[i][j + 1] -= div / 3.0;
                    v[i + 1][j] -= div / 3.0;
                    continue;
                }
                if i == resolution {
                    if j == 1 {
                        u[i][j + 1] -= div / 2.0;
                        v[i][j] += div / 2.0;
                        continue;
                    }
                    if j == resolution {
                        u[i][j] += div / 2.0;
                        v[i][j] += div / 2.0;
                        continue;
                    }

                    u[i][j] += div / 3.0;
                    u[i][j + 1] -= div / 3.0;
                    v[i][j] += div / 3.0;
                    continue;
                }
                if j == 1 {
                    u[i][j + 1] -= div / 3.0;
                    v[i][j] += div / 3.0;
                    v[i + 1][j] -= div / 3.0;
                    continue;
                }
                if j == resolution {
                    u[i][j] += div / 3.0;
                    v[i][j] += div / 3.0;
                    v[i + 1][j] -= div / 3.0;
                    continue;
                }

                u[i][j] += div / 4.0;
                u[i][j + 1] -= div / 4.0;
                v[i][j] += div / 4.0;
                v[i + 1][j] -= div / 4.0;
            }
        }
    }

    fn simulate(&mut self, dt: f32) {
        Self::project(
            self.resolution,
            dt,
            &mut self.grid_u[0_i32],
            &mut self.grid_v[0_i32],
            &mut self.pressure_grid,
            &self.pressure_divergence_mat,
        );

        let mut grid_u = self.grid_u.clone();
        Self::advect(
            self.resolution,
            dt,
            &mut grid_u,
            &self.grid_u[0_i32],
            &self.grid_v[0_i32],
        );
        self.grid_u = grid_u;

        let mut grid_v = self.grid_v.clone();
        Self::advect(
            self.resolution,
            dt,
            &mut grid_v,
            &self.grid_u[0_i32],
            &self.grid_v[0_i32],
        );
        self.grid_v = grid_v;

        self.grid_u.rotate_left(1);
        self.grid_v.rotate_left(1);
    }

    /* fn velocity_grid(&self) -> [[f32; 128]; 128] {
        (1..=self.resolution)
            .map(|i| {
                let new_row: [f32; 128] = (1..=self.resolution)
                    .map(|j| {
                        Vector2::new(self.grid_u[0_i32][i][j], self.grid_v[0_i32][i][j]).magnitude()
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                new_row
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn pressure_grid(&self) -> [[f32; 128]; 128] {
        self.pressure_grid
            .iter()
            .map(|row| {
                let new_row: [f32; 128] = row.to_vec().try_into().unwrap();
                new_row
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    } */
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
                // self.density_grid[i][j] = 1.0;
                self.grid_u[0_i32][i][j] = 1.0;
                self.grid_v[0_i32][i][j] = 0.5;
            }
        }

        // self.add_forces(forces, dt)
    }

    fn update(
        &mut self,
        device: Arc<Device>,
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
        let dt = details.last_frame_duration.as_secs_f32() * 5.0;

        /* for i in 2..=self.resolution {
            for j in 2..=self.resolution {
                self.grid_v[0_i32][i][j] = 1.0;
            }
        } */
        self.simulate(dt);
        // println!("frame");

        let rgba =
            image::RgbaImage::from_fn(self.resolution as u32, self.resolution as u32, |x, y| {
                image::Rgba(
                    [Self::lerp(0.0, 255.0, self.pressure_grid[y as usize][x as usize]) as u8; 4],
                )
            });

        let tex = gamezap::texture::Texture::from_rgba(
            &device,
            &queue,
            &rgba,
            Some("Pressure texture"),
            false,
            false,
        )
        .unwrap();

        let materials = materials.unwrap();
        materials.0[0].update_textures(device, vec![&tex]);

        /* let selected_material = &mut materials.0[materials.1];
        if let Some((_, buffer)) = &selected_material.uniform_buffer_bind_group() {
            let grid_2 = self.pressure_grid();
            // println!("{:?}", &grid_2[50][50]);
            let bytes: [u8; 4 * 128 * 128] = zerocopy::transmute!(grid_2);
            queue.write_buffer(buffer, 0, &bytes);
        } */
    }
}
