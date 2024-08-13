use gamezap::new_component;

new_component!(FpsComponent {});

impl Default for FpsComponent {
    fn default() -> Self {
        FpsComponent { 
            parent: 0,
            id: (u32::MAX, TypeId::of::<Self>(), u32::MAX),
        }
    }
}

impl ComponentSystem for FpsComponent {
    fn ui_draw(
        &mut self,
        _device: Arc<Device>,
        _queue: Arc<Queue>,
        _ui_manager: &mut gamezap::ui_manager::UiManager,
        ui_frame: &mut imgui::Ui,
        _component_map: &mut AllComponents,
        _concept_manager: Rc<Mutex<ConceptManager>>,
        engine_details: Rc<Mutex<EngineDetails>>,
        _engine_systems: Rc<Mutex<EngineSystems>>,
    ) {
        let details = engine_details.lock().unwrap();
        let fps = details.last_frame_duration.as_millis();

        ui_frame
            .window("fps")
            .menu_bar(false)
            .title_bar(false)
            .movable(false)
            .position([0.0; 2], imgui::Condition::Always)
            .build(|| {
                ui_frame.text(format!("Frame Time: {fps}"));
            });
    }
}
