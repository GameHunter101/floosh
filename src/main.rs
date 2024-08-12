use gamezap::{
    ecs::{components::mesh_component::MeshComponent, material::Material, scene::Scene},
    model::Vertex,
    GameZap,
};
use simulator_component::SimulatorComponent;

pub mod simulator_component;

#[tokio::main]
async fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let event_pump = sdl_context.event_pump().unwrap();
    let application_title = "Floosh";
    let window_size = (512, 512);
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

    let simulator_component = SimulatorComponent::new(512, 1.0);

    let canvas_mesh = MeshComponent::new(
        concept_manager,
        vec![
            Vertex {
                position: [-1.0, 1.0, 0.0],
                tex_coords: [0.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 0.0],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [1.0, -1.0, 0.0],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0, 0.0],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
        ],
        vec![0, 1, 2, 0, 2, 3],
    );

    // let bytes: [u8; 4 * 128 * 128] = zerocopy::transmute!([[0.0_f32; 128]; 128]);

    let simulator_material = Material::new(
        "shaders/vert.wgsl",
        "shaders/frag.wgsl",
        vec![gamezap::texture::Texture::blank_texture(
            &device.clone(),
            &queue,
            256,
            256,
            Some("test tex"),
            true,
        ).unwrap()],
        // Some(&bytes),
        None,
        // Some(bytemuck::cast_slice(&[data])),
        true,
        device,
    );

    let _simulation = scene.create_entity(
        0,
        true,
        vec![Box::new(simulator_component), Box::new(canvas_mesh)],
        Some((vec![simulator_material], 0)),
    );

    engine.create_scene(scene);

    engine.main_loop();
}
