use cool_utils::data_structures::ring_buffer::RingBuffer;
use gamezap::new_component;

new_component!(SimulatorComponent {
    resolution: usize,
    compute_index: usize,
    diffusion_coefficient: f32,
    viscosity: f32,
    grid_u: Vec<Vec<f32>>,
    grid_v: Vec<Vec<f32>>,
    grid_u_0: Vec<Vec<f32>>,
    grid_v_0: Vec<Vec<f32>>,
    density_grid: Vec<Vec<f32>>,
    scratchpad: Vec<Vec<f32>>
});

impl SimulatorComponent {
    pub fn new(
        concept_manager: Rc<Mutex<ConceptManager>>,
        compute_index: usize,
        resolution: usize,
        diffusion_coefficient: f32,
        viscosity: f32,
    ) -> Self {
        let zeroes = vec![vec![0.0; resolution + 2]; resolution + 2];

        let mut component = SimulatorComponent {
            resolution,
            compute_index,
            diffusion_coefficient,
            viscosity,
            grid_u: zeroes.clone(),
            grid_v: zeroes.clone(),
            grid_u_0: zeroes.clone(),
            grid_v_0: zeroes.clone(),
            density_grid: zeroes.clone(),
            scratchpad: zeroes,
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        };

        let mut concepts: HashMap<String, Box<dyn Any>> = HashMap::new();
        concepts.insert(
            "forces_x".to_string(),
            Box::new(vec![vec![0.0_f32; resolution]; resolution]),
        );
        concepts.insert(
            "forces_y".to_string(),
            Box::new(vec![vec![0.0_f32; resolution]; resolution]),
        );
        concepts.insert(
            "added_densities".to_string(),
            Box::new(vec![vec![0.0_f32; resolution]; resolution]),
        );
        concepts.insert(
            "last_mouse_info".to_string(),
            Box::new(((0, 0), std::time::Instant::now())),
        );

        component.register_component(concept_manager, concepts);

        component
    }

    fn set_boundary(resolution: usize, boundary_type: u32, grid: &mut [Vec<f32>]) {
        for i in 1..=resolution {
            grid[i][0] = if boundary_type == 2 { -1.0 } else { 1.0 } * grid[i][1];
            grid[i][resolution + 1] =
                if boundary_type == 2 { -1.0 } else { 1.0 } * grid[i][resolution];

            grid[0][i] = if boundary_type == 1 { -1.0 } else { 1.0 } * grid[1][i];
            grid[resolution + 1][i] =
                if boundary_type == 1 { -1.0 } else { 1.0 } * grid[resolution][i];
        }

        grid[0][0] = 0.5 * (grid[0][1] + grid[1][0]);
        grid[0][resolution + 1] = 0.5 * (grid[0][resolution] + grid[1][resolution + 1]);
        grid[resolution + 1][0] = 0.5 * (grid[resolution + 1][1] + grid[resolution][0]);
        grid[resolution + 1][resolution + 1] =
            0.5 * (grid[resolution + 1][resolution] + grid[resolution][resolution + 1]);
    }

    fn linear_system_solver(
        resolution: usize,
        boundary_type: u32,
        grid: &mut [Vec<f32>],
        grid_0: &[Vec<f32>],
        coefficient: f32,
        divisor: f32,
        num_iters: u32,
    ) {
        for _ in 0..num_iters {
            for i in 1..=resolution {
                for j in 1..=resolution {
                    grid[i][j] = (grid_0[i][j]
                        + coefficient
                            * (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1]))
                        / divisor;
                }
            }
            Self::set_boundary(resolution, boundary_type, grid);
        }
    }

    fn lerp(a: f32, b: f32, k: f32) -> f32 {
        a + k * (b - a)
    }

    fn diffuse(
        resolution: usize,
        boundary_type: u32,
        grid: &mut [Vec<f32>],
        grid_0: &[Vec<f32>],
        diffusion_coefficient: f32,
        num_iters: u32,
        dt: f32,
    ) {
        let coefficient = dt * diffusion_coefficient * resolution as f32 * resolution as f32;
        Self::linear_system_solver(
            resolution,
            boundary_type,
            grid,
            grid_0,
            coefficient,
            1.0 + 4.0 * coefficient,
            num_iters,
        );
    }

    fn advect(
        resolution: usize,
        boundary_type: u32,
        dt: f32,
        grid: &mut [Vec<f32>],
        grid_0: &[Vec<f32>],
        u: &[Vec<f32>],
        v: &[Vec<f32>],
    ) {
        for i in 1..=resolution {
            for j in 1..=resolution {
                let prev_x = (j as f32 - dt * u[i][j]).clamp(0.5, resolution as f32 + 0.5);
                let prev_y = (i as f32 - dt * v[i][j]).clamp(0.5, resolution as f32 + 0.5);

                let top_left_value = grid_0[prev_y as usize][prev_x as usize];
                let top_right_value = grid_0[prev_y as usize][prev_x as usize + 1];

                let bottom_left_value = grid_0[prev_y as usize + 1][prev_x as usize];
                let bottom_right_value = grid_0[prev_y as usize + 1][prev_x as usize + 1];

                let top_lerp = Self::lerp(top_left_value, top_right_value, prev_x.fract());
                let bottom_lerp = Self::lerp(bottom_left_value, bottom_right_value, prev_x.fract());

                let final_value = Self::lerp(top_lerp, bottom_lerp, prev_y.fract());

                grid[i][j] = final_value;
            }
        }

        Self::set_boundary(resolution, boundary_type, grid);
    }

    fn project(
        resolution: usize,
        u: &mut [Vec<f32>],
        v: &mut [Vec<f32>],
        p: &mut [Vec<f32>],
        div: &mut [Vec<f32>],
        num_iters: u32,
    ) {
        let cell_size = 1.0 / resolution as f32;

        for i in 1..=resolution {
            for j in 1..=resolution {
                div[i][j] =
                    -0.5 * cell_size * (u[i + 1][j] - u[i - 1][j] + v[i][j + 1] - v[i][j - 1]);
                p[i][j] = 0.0;
            }
        }

        Self::set_boundary(resolution, 0, div);
        Self::set_boundary(resolution, 0, p);

        Self::linear_system_solver(resolution, 0, p, div, 1.0, 4.0, num_iters);

        for i in 1..=resolution {
            for j in 1..=resolution {
                u[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]) / cell_size;
                v[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]) / cell_size;
            }
        }

        Self::set_boundary(resolution, 1, u);
        Self::set_boundary(resolution, 2, v);
    }

    fn simulate(&mut self, dt: f32, num_iters: u32) {
        let grid_u = self.grid_u.clone();
        Self::diffuse(
            self.resolution,
            1,
            &mut self.grid_u_0,
            &grid_u,
            self.viscosity,
            num_iters,
            dt,
        );
        let grid_v = self.grid_v.clone();
        Self::diffuse(
            self.resolution,
            2,
            &mut self.grid_v_0,
            &grid_v,
            self.viscosity,
            num_iters,
            dt,
        );

        Self::project(
            self.resolution,
            &mut self.grid_u_0,
            &mut self.grid_v_0,
            &mut self.grid_u,
            &mut self.grid_v,
            num_iters,
        );

        let grid_u_0 = self.grid_u_0.clone();
        Self::advect(
            self.resolution,
            1,
            dt,
            &mut self.grid_u,
            &grid_u_0,
            &self.grid_u_0,
            &self.grid_v_0,
        );

        let grid_v_0 = self.grid_v_0.clone();
        Self::advect(
            self.resolution,
            2,
            dt,
            &mut self.grid_v,
            &grid_v_0,
            &self.grid_u_0,
            &self.grid_v_0,
        );

        Self::project(
            self.resolution,
            &mut self.grid_u,
            &mut self.grid_v,
            &mut self.grid_u_0,
            &mut self.grid_v_0,
            num_iters,
        );

        Self::diffuse(
            self.resolution,
            0,
            &mut self.scratchpad,
            &self.density_grid,
            self.diffusion_coefficient,
            num_iters,
            dt,
        );
        Self::advect(
            self.resolution,
            0,
            dt,
            &mut self.density_grid,
            &self.scratchpad,
            &self.grid_u,
            &self.grid_v,
        );
    }
}

impl ComponentSystem for SimulatorComponent {
    fn register_component(
        &mut self,
        concept_manager: Rc<Mutex<ConceptManager>>,
        data: HashMap<String, Box<dyn Any>>,
    ) {
        concept_manager
            .lock()
            .unwrap()
            .register_component_concepts(self.id, data);
    }

    fn update(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        _component_map: &mut AllComponents,
        engine_details: Rc<Mutex<EngineDetails>>,
        _engine_systems: Rc<Mutex<EngineSystems>>,
        concept_manager: Rc<Mutex<ConceptManager>>,
        _active_camera_id: Option<EntityId>,
        _entities: &mut Vec<Entity>,
        materials: Option<&mut (Vec<Material>, usize)>,
        compute_pipelines: &mut [ComputePipeline],
    ) {
        let details = engine_details.lock().unwrap();
        let dt = details.last_frame_duration.as_secs_f32() * 10.0;

        let mut concept_manager = concept_manager.lock().unwrap();

        let added_rgba = {
            let forces_x = concept_manager
                .get_concept::<Vec<Vec<f32>>>(self.id, "forces_x".to_string())
                .unwrap();

            let forces_y = concept_manager
                .get_concept::<Vec<Vec<f32>>>(self.id, "forces_y".to_string())
                .unwrap();

            let added_densities = concept_manager
                .get_concept::<Vec<Vec<f32>>>(self.id, "added_densities".to_string())
                .unwrap();

            for i in 0..self.resolution {
                for j in 0..self.resolution {
                    self.grid_u[i + 1][j + 1] += forces_x[i][j] * dt;
                    self.grid_v[i + 1][j + 1] += forces_y[i][j] * dt;

                    self.density_grid[i + 1][j + 1] += added_densities[i][j];
                }
            }

            image::RgbaImage::from_fn(
                self.resolution as u32 + 2,
                self.resolution as u32 + 2,
                |x, y| {
                    let rgba = if x > 0
                        && x < self.resolution as u32 + 1
                        && y > 0
                        && y < self.resolution as u32 + 1
                    {
                        let x_vel =
                            Self::lerp(0.0, 255.0, forces_x[y as usize - 1][x as usize - 1]) as u8;
                        let y_vel =
                            Self::lerp(0.0, 255.0, forces_y[y as usize - 1][x as usize - 1]) as u8;

                        let dens =
                            Self::lerp(0.0, 255.0, added_densities[y as usize - 1][x as usize - 1])
                                as u8;

                        [x_vel, y_vel, dens, 1]
                    } else {
                        [0; 4]
                    };
                    image::Rgba(rgba)
                },
            )
        };

        /* let added_densities = concept_manager
            .get_concept_mut::<Vec<Vec<f32>>>(self.id, "added_densities".to_string())
            .unwrap();

        *added_densities = vec![vec![0.0; self.resolution]; self.resolution];

        let forces_x = concept_manager
            .get_concept_mut::<Vec<Vec<f32>>>(self.id, "forces_x".to_string())
            .unwrap();

        *forces_x = vec![vec![0.0; self.resolution]; self.resolution];

        let forces_y = concept_manager
            .get_concept_mut::<Vec<Vec<f32>>>(self.id, "forces_y".to_string())
            .unwrap();
        *forces_y = vec![vec![0.0; self.resolution]; self.resolution]; */

        // self.simulate(dt, 5);

        /* let rgba =
        image::RgbaImage::from_fn(self.resolution as u32, self.resolution as u32, |x, y| {
            let rgba = vec![
                Self::lerp(
                    0.0,
                    255.0,
                    self.density_grid[y as usize + 1][x as usize + 1],
                ) as u8,
                Self::lerp(
                    0.0,
                    255.0,
                    self.density_grid[y as usize + 1][x as usize + 1],
                ) as u8,
                Self::lerp(
                    0.0,
                    255.0,
                    self.density_grid[y as usize + 1][x as usize + 1],
                ) as u8,
                255,
            ];
            image::Rgba(rgba.try_into().unwrap())
        }); */

        let added_tex = Rc::new(
            gamezap::texture::Texture::from_rgba(
                &device,
                &queue,
                &added_rgba,
                Some("Added data texture"),
                true,
                true,
            )
            .unwrap(),
        );

        compute_pipelines[self.compute_index].update_pipeline_assets(
            device.clone(),
            vec![(
                gamezap::compute::ComputePackagedData::Texture(added_tex.clone()),
                2,
            )],
        );
        let materials = materials.unwrap();
        materials.0[0].update_textures(
            device,
            &[(
                compute_pipelines[self.compute_index].pipeline_assets[3]
                    .as_texture()
                    .unwrap()
                    .clone(),
                0,
            )],
        );
        // materials.0[0].update_textures(device, &[(added_tex, 0)]);
    }

    fn on_event(
        &self,
        event: &sdl2::event::Event,
        _component_map: &HashMap<EntityId, Vec<gamezap::ecs::component::Component>>,
        concept_manager: Rc<Mutex<ConceptManager>>,
        _active_camera_id: Option<EntityId>,
        engine_details: &EngineDetails,
        _engine_systems: &EngineSystems,
    ) {
        let scale = 3;
        let brush_size = 20_i32;
        let vel_mul = 20.0;
        let dens_mul = 2.0;
        let fall_off = 3.0;

        if let sdl2::event::Event::MouseMotion {
            mousestate, x, y, ..
        } = event
        {
            let mut concept_manager = concept_manager.lock().unwrap();

            if mousestate.left() {
                let ((last_x, last_y), last_instant) = *concept_manager
                    .get_concept::<((i32, i32), std::time::Instant)>(
                        self.id,
                        "last_mouse_info".to_string(),
                    )
                    .unwrap();
                if (std::time::Instant::now() - last_instant)
                    >= std::time::Duration::from_millis(10)
                {
                    *concept_manager
                        .get_concept_mut::<((i32, i32), std::time::Instant)>(
                            self.id,
                            "last_mouse_info".to_string(),
                        )
                        .unwrap() = ((*x, *y), std::time::Instant::now());
                }
                let forces_x = concept_manager
                    .get_concept_mut::<Vec<Vec<f32>>>(self.id, "forces_x".to_string())
                    .unwrap();
                for i in 0..brush_size * 2 {
                    for j in 0..brush_size * 2 {
                        let i = i - brush_size;
                        let j = j - brush_size;

                        let recalculated_y =
                            (*y / scale + i).clamp(0, self.resolution as i32 - 1) as usize;
                        let recalculated_x =
                            (*x / scale + j).clamp(0, self.resolution as i32 - 1) as usize;

                        let dist = ((i * i + j * j) as f32).sqrt();
                        if dist < brush_size as f32 {
                            let diff = (x - last_x) as f32;
                            let vel = diff
                                / (20.0/* * engine_details.last_frame_duration.as_millis() as f32 */);

                            forces_x[recalculated_y][recalculated_x] =
                                vel * vel_mul * (-1.0 * brush_size as f32 - dist);
                        }
                    }
                }
                let forces_y = concept_manager
                    .get_concept_mut::<Vec<Vec<f32>>>(self.id, "forces_y".to_string())
                    .unwrap();

                for i in 0..brush_size * 2 {
                    for j in 0..brush_size * 2 {
                        let recalculated_y =
                            (*y / scale + i).clamp(0, self.resolution as i32 - 1) as usize;
                        let recalculated_x =
                            (*x / scale + j).clamp(0, self.resolution as i32 - 1) as usize;

                        let i = i - brush_size;
                        let j = j - brush_size;
                        let dist = ((i * i + j * j) as f32).sqrt();
                        if dist < brush_size as f32 {
                            let diff = (y - last_y) as f32;
                            let vel = diff
                                / (20.0/* * engine_details.last_frame_duration.as_millis() as f32 */);

                            forces_y[recalculated_y][recalculated_x] = vel * vel_mul * dist;
                        }
                    }
                }
            }

            if mousestate.right() {
                let scancodes = &engine_details.pressed_scancodes;
                let added_densities = concept_manager
                    .get_concept_mut::<Vec<Vec<f32>>>(self.id, "added_densities".to_string())
                    .unwrap();

                for i in 0..brush_size * 2 {
                    for j in 0..brush_size * 2 {
                        let i = i - brush_size;
                        let j = j - brush_size;
                        if ((i * i + j * j) as f32).sqrt() < brush_size as f32 {
                            added_densities
                                [((*y / scale + i) as usize).clamp(0, self.resolution - 1)]
                                [((*x / scale + j) as usize).clamp(0, self.resolution - 1)] +=
                                dens_mul
                                    * if scancodes.contains(&sdl2::keyboard::Scancode::LShift) {
                                        -2.0
                                    } else {
                                        2.0
                                    };
                        }
                    }
                }
            }
        }
    }
}
