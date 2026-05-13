# RustVK

A modular Vulkan renderer written in Rust using raw [`ash`](https://github.com/ash-rs/ash) bindings. No high-level abstractions — every Vulkan object is managed explicitly with RAII ownership through Rust's `Drop` trait.

Currently renders a lit, rotating cube with Blinn-Phong shading, 4x MSAA, and a physically-attenuated point light. The architecture is designed to grow: each renderer subsystem is an isolated, typed struct that can be extended or replaced independently.

---

## Features

- **Raw Vulkan via `ash`** — direct API calls, no wgpu or vulkano
- **Blinn-Phong shading** — ambient, diffuse, and specular with quadratic point-light attenuation
- **4x MSAA** — multisampled color and depth resolved to swapchain at end of render pass
- **Double-buffered frame pacing** — two frames in flight, per-frame command buffers, fences, and uniform buffers
- **Correct semaphore lifecycle** — `render_finished` semaphores indexed per swapchain image to avoid presentation-engine hazards; acquire semaphores use a rotating pool of `MAX_FRAMES_IN_FLIGHT + 1`
- **Staged geometry upload** — vertex and index buffers in `DEVICE_LOCAL` memory via transient staging buffers
- **GLSL shaders compiled at build time** — `build.rs` invokes `glslc`; SPIR-V embedded via `include_bytes!`
- **Swapchain recreation** — resize and `ERROR_OUT_OF_DATE_KHR` handled cleanly; `ERROR_SURFACE_LOST_KHR` propagates without unsafe re-entry
- **Validation layers in debug builds** — `VK_LAYER_KHRONOS_validation` auto-enabled, routed through Rust's `log` facade

---

## Preview

> Screenshot coming soon.

---

## Architecture

Each subsystem owns its Vulkan handles and is responsible for its own teardown. `VulkanRenderer` composes them and enforces destruction order through field declaration order (Rust drops fields in the order they are declared).

```
src/
├── main.rs                  # winit event loop, cube rotation, resize
├── scene/
│   └── mod.rs               # Vertex, Camera, Light, UniformBufferObject, Scene
├── shaders/
│   ├── shader.vert           # MVP transform, normal matrix
│   └── shader.frag           # Blinn-Phong point light
└── renderer/
    ├── mod.rs               # VulkanRenderer — top-level orchestrator
    ├── instance.rs          # VulkanInstance — entry, instance, debug messenger
    ├── device.rs            # VulkanDevice — physical/logical device, queues
    ├── swapchain.rs         # VulkanSwapchain — format selection, image views
    ├── msaa.rs              # VulkanMsaaBuffer — TRANSIENT_ATTACHMENT color at 4x
    ├── depth.rs             # VulkanDepthBuffer — depth image with format probe
    ├── render_pass.rs       # VulkanRenderPass — 3-attachment MSAA pass
    ├── framebuffer.rs       # VulkanFramebuffers — one per swapchain image
    ├── pipeline.rs          # VulkanPipeline — full graphics PSO
    ├── descriptor.rs        # VulkanDescriptorSets — pool, layout, per-frame UBO sets
    ├── buffer.rs            # VulkanBuffer — vertex, index, uniform, staging upload
    ├── command.rs           # VulkanCommandPool — per-frame CBs, one-shot helper
    └── sync.rs              # VulkanSync — semaphores and fences
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Rust stable | 1.78+ |
| Vulkan 1.2+ | |
| `glslc` | From the [LunarG Vulkan SDK](https://vulkan.lunarg.com/); must be on `PATH` at build time |
| `VK_LAYER_KHRONOS_validation` | Optional; install via the Vulkan SDK for debug output |

**Platform notes**

- **Linux (Wayland)**: tested on KDE Plasma with NVIDIA (driver 595.x). Present mode is forced to `FIFO` to avoid `wp_tearing_control_v1` protocol conflicts on some compositor/driver combinations.
- **Linux (X11)** and **Windows**: should work; surface creation is handled by `ash-window`.
- **macOS**: not tested; would require MoltenVK.

---

## Build & Run

```bash
git clone https://github.com/CanReader/RustVK.git
cd RustVK

# Debug (validation layers enabled)
cargo run

# Release
cargo run --release

# Verbose Vulkan output
RUST_LOG=debug cargo run
```

`build.rs` compiles the GLSL shaders to SPIR-V automatically. Pre-compiled `.spv` files are checked in, so the project can also be built without the Vulkan SDK installed.

---

## Rendering Pipeline

```
Vertex & Index buffers (DEVICE_LOCAL)
         |
         v
   Vertex shader
   MVP transform, normal matrix (transpose-inverse model)
         |
         v
   Fragment shader — Blinn-Phong
   attenuation = 1 / (1 + 0.045d + 0.0075d²)
   result = (ambient + diffuse * atten + specular * atten) * albedo
         |
         v
   4x MSAA color + depth attachments
         |
         v
   Resolve to 1x swapchain image
         |
         v
   vkQueuePresentKHR
```

**Uniform buffer (per frame, `HOST_COHERENT`)**

```glsl
layout(binding = 0) uniform UBO {
    mat4 model;       // cube rotation
    mat4 view;        // fixed camera at (0, 2, 5)
    mat4 proj;        // perspective 45 FOV, Vulkan NDC
    vec3 lightPos;
    vec3 lightColor;  // warm white (1.0, 0.95, 0.85)
    vec3 viewPos;
};
```

---

## Frame Loop

```
wait_for_fences(in_flight_fences[frame])
acquire_next_image(image_available_semaphores[acquire_sem_index])
reset_fences
update_uniform_buffer(frame)
record_command_buffer(image_index, frame)
queue_submit(
    wait:   image_available_semaphores[acquire_sem_index],
    signal: render_finished_semaphores[image_index],
    fence:  in_flight_fences[frame]
)
queue_present(wait: render_finished_semaphores[image_index])
current_frame      = (frame + 1) % MAX_FRAMES_IN_FLIGHT
acquire_sem_index  = (acquire_sem_index + 1) % (MAX_FRAMES_IN_FLIGHT + 1)
```

`render_finished` semaphores are indexed by `image_index`, not `frame`. The presentation engine holds a semaphore until the corresponding swapchain image is released back to the application. Indexing by frame causes `VUID-vkQueueSubmit-pSignalSemaphores-00067` and GPU faults when the swapchain has more images than frames in flight.

---

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| [`ash`](https://crates.io/crates/ash) | 0.38 | Raw Vulkan bindings |
| [`winit`](https://crates.io/crates/winit) | 0.30 | Cross-platform windowing |
| [`ash-window`](https://crates.io/crates/ash-window) | 0.13 | Vulkan surface from winit |
| [`raw-window-handle`](https://crates.io/crates/raw-window-handle) | 0.6 | Platform window handles |
| [`cgmath`](https://crates.io/crates/cgmath) | 0.18 | Math (vectors, matrices, projections) |
| [`bytemuck`](https://crates.io/crates/bytemuck) | 1 | Zero-copy GPU data casting |
| [`log`](https://crates.io/crates/log) + [`env_logger`](https://crates.io/crates/env_logger) | 0.4 / 0.11 | Logging |

---

## Roadmap

- [ ] glTF model loading
- [ ] PBR material system (metallic-roughness)
- [ ] Shadow mapping
- [ ] Normal maps
- [ ] Multiple lights
- [ ] ImGui integration for scene controls
- [ ] Render graph abstraction

---

## License

MIT. See [LICENSE](LICENSE).
