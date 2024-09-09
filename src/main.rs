use std::rc::Rc;

use fps_component::FpsComponent;
use gamezap::{
    ecs::{components::mesh_component::MeshComponent, material::Material, scene::Scene},
    model::Vertex,
    GameZap,
};
use simulator_component::SimulatorComponent;

pub mod fps_component;
pub mod simulator_component;

#[tokio::main]
async fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let event_pump = sdl_context.event_pump().unwrap();
    let application_title = "Floosh";
    let window_size = (256 * 3, 256 * 3);
    let window = video_subsystem
        .window(application_title, window_size.0, window_size.1)
        .resizable()
        .build()
        .unwrap();

    let mut engine = GameZap::builder()
        .window_and_renderer(
            sdl_context,
            video_subsystem,
            event_pump,
            window,
            wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        )
        .antialiasing()
        .build()
        .await;

    let device = engine.renderer.device.clone();
    let queue = engine.renderer.queue.clone();

    let mut scene = Scene::default();

    let concept_manager = scene.get_concept_manager();

    let sim_res = 256;

    let simulation_pipeline_index = scene
        .create_compute_pipeline(
            device.clone(),
            queue.clone(),
            "shaders/compute.wgsl",
            (sim_res as u32, sim_res as u32, 1),
            gamezap::compute::ComputePipelineType {
                input_data: vec![
                    gamezap::compute::ComputeData::TextureData((
                        gamezap::compute::ComputeTextureData::Dimensions((
                            sim_res as u32 + 2,
                            sim_res as u32 + 2,
                        )),
                        true,
                    )),
                    gamezap::compute::ComputeData::TextureData((
                        gamezap::compute::ComputeTextureData::Dimensions((
                            sim_res as u32 + 2,
                            sim_res as u32 + 2,
                        )),
                        true,
                    )),
                    gamezap::compute::ComputeData::ArrayData(bytemuck::cast_slice(&[0.0_f32; 4])),
                    gamezap::compute::ComputeData::ArrayData(bytemuck::cast_slice(&[0.0_f32; 4]))
                ],
                output_data_type: vec![gamezap::compute::ComputeOutput::Texture((
                    sim_res as u32,
                    sim_res as u32,
                ))],
            },
        )
        .unwrap();

    let simulator_component = SimulatorComponent::new(
        concept_manager.clone(),
        simulation_pipeline_index,
        sim_res,
        0.0000,
        // 1.0,
        0.00001,
    );

    let canvas_mesh = MeshComponent::new(
        concept_manager,
        vec![
            Vertex {
                position: [-1.0, 1.0, 0.0],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 0.0],
                tex_coords: [0.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [1.0, -1.0, 0.0],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0, 0.0],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
        ],
        vec![0, 1, 2, 0, 2, 3],
    );

    // let bytes: [u8; 4 * 128 * 128] = zerocopy::transmute!([[0.0_f32; 128]; 128]);

    let simulator_material = Material::new(
        "shaders/vert.wgsl",
        "shaders/frag.wgsl",
        vec![
            Rc::new(
                gamezap::texture::Texture::blank_texture(
                    &device.clone(),
                    &queue.clone(),
                    sim_res as u32 + 2,
                    sim_res as u32 + 2,
                    Some("dens tex"),
                    true,
                )
                .unwrap(),
            ),
            Rc::new(
                gamezap::texture::Texture::blank_texture(
                    &device.clone(),
                    &queue.clone(),
                    sim_res as u32 + 2,
                    sim_res as u32 + 2,
                    Some("dens tex 2"),
                    true,
                )
                .unwrap(),
            ),
        ],
        // Some(&bytes),
        None,
        // Some(bytemuck::cast_slice(&[data])),
        true,
        device.clone(),
    );

    let _simulation = scene.create_entity(
        0,
        true,
        vec![Box::new(simulator_component), Box::new(canvas_mesh)],
        Some((vec![simulator_material], 0)),
    );

    let fps_component = FpsComponent::default();

    let _fps_display = scene.create_entity(0, true, vec![Box::new(fps_component)], None);

    engine.create_scene(scene);

    engine.main_loop();
}
