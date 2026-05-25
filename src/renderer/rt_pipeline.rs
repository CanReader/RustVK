use ash::vk;
use super::device::VulkanDevice;
use super::buffer::VulkanBuffer;

/// Loaded SPIR-V data as a `Vec<u32>`.
fn load_spv(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("SPIR-V file {} has non-aligned size", path).into());
    }
    // SAFETY: bytes is heap-allocated Vec<u8> with alignment 1; we reinterpret as u32.
    // bytemuck::cast_slice would also work but we have no import here.
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(words)
}

/// Holds the RT pipeline and its shader binding table.
pub struct RtPipeline {
    pub pipeline:        vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    #[allow(dead_code)]
    pub sbt_buffer:      VulkanBuffer,   // kept alive for device address validity
    pub rgen_region:     vk::StridedDeviceAddressRegionKHR,
    pub miss_region:     vk::StridedDeviceAddressRegionKHR,
    pub hit_region:      vk::StridedDeviceAddressRegionKHR,
    pub callable_region: vk::StridedDeviceAddressRegionKHR,
    device_ref:          ash::Device,
}

impl RtPipeline {
    /// Build the RT pipeline and SBT.
    ///
    /// `ds_layout` — the descriptor set layout to bind (set 0).
    /// `shader_dir` — directory containing raygen.spv, miss.spv, shadow.spv, closesthit.spv.
    pub fn new(
        device:     &VulkanDevice,
        ds_layout:  vk::DescriptorSetLayout,
        shader_dir: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let rt_loader = device.rt_pipeline_loader.as_ref()
            .ok_or("RT pipeline loader not present")?;
        let rt_props  = &device.rt_props;

        // ── Shader modules ────────────────────────────────────────────────────
        let rgen_spv   = load_spv(&format!("{}/raygen.spv",     shader_dir))?;
        let miss_spv   = load_spv(&format!("{}/miss.spv",       shader_dir))?;
        let shadow_spv = load_spv(&format!("{}/shadow.spv",     shader_dir))?;
        let chit_spv   = load_spv(&format!("{}/closesthit.spv", shader_dir))?;

        let make_module = |spv: &[u32]| -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
            let info = vk::ShaderModuleCreateInfo {
                code_size: spv.len() * 4,
                p_code:    spv.as_ptr(),
                ..Default::default()
            };
            Ok(unsafe { device.device.create_shader_module(&info, None)? })
        };

        let rgen_module   = make_module(&rgen_spv)?;
        let miss_module   = make_module(&miss_spv)?;
        let shadow_module = make_module(&shadow_spv)?;
        let chit_module   = make_module(&chit_spv)?;

        let entry = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };

        // ── Shader stages ─────────────────────────────────────────────────────
        let stages = [
            // 0: raygen
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::RAYGEN_KHR,
                module: rgen_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            // 1: sky miss
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::MISS_KHR,
                module: miss_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            // 2: shadow miss
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::MISS_KHR,
                module: shadow_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
            // 3: closest hit
            vk::PipelineShaderStageCreateInfo {
                stage:  vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                module: chit_module,
                p_name: entry.as_ptr(),
                ..Default::default()
            },
        ];

        // ── Shader groups ─────────────────────────────────────────────────────
        // Group 0: raygen (general)
        // Group 1: sky miss (general)
        // Group 2: shadow miss (general)
        // Group 3: closest hit
        let groups = [
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty:                          vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader:              0,
                closest_hit_shader:          vk::SHADER_UNUSED_KHR,
                any_hit_shader:              vk::SHADER_UNUSED_KHR,
                intersection_shader:         vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty:                          vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader:              1,
                closest_hit_shader:          vk::SHADER_UNUSED_KHR,
                any_hit_shader:              vk::SHADER_UNUSED_KHR,
                intersection_shader:         vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty:                          vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader:              2,
                closest_hit_shader:          vk::SHADER_UNUSED_KHR,
                any_hit_shader:              vk::SHADER_UNUSED_KHR,
                intersection_shader:         vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty:                          vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                general_shader:              vk::SHADER_UNUSED_KHR,
                closest_hit_shader:          3,
                any_hit_shader:              vk::SHADER_UNUSED_KHR,
                intersection_shader:         vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
        ];

        // ── Pipeline layout ───────────────────────────────────────────────────
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count:          1,
            p_set_layouts:             &ds_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges:    std::ptr::null(),
            ..Default::default()
        };
        let pipeline_layout = unsafe {
            device.device.create_pipeline_layout(&layout_info, None)?
        };

        // ── RT Pipeline ───────────────────────────────────────────────────────
        let rt_create_info = vk::RayTracingPipelineCreateInfoKHR {
            stage_count:                    stages.len() as u32,
            p_stages:                       stages.as_ptr(),
            group_count:                    groups.len() as u32,
            p_groups:                       groups.as_ptr(),
            max_pipeline_ray_recursion_depth: 2,  // primary + shadow
            layout:                         pipeline_layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            rt_loader.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&rt_create_info),
                None,
            ).map_err(|(_, e)| e)?[0]
        };

        // Destroy shader modules now that the pipeline is compiled
        unsafe {
            device.device.destroy_shader_module(rgen_module,   None);
            device.device.destroy_shader_module(miss_module,   None);
            device.device.destroy_shader_module(shadow_module, None);
            device.device.destroy_shader_module(chit_module,   None);
        }

        // ── Shader Binding Table ──────────────────────────────────────────────
        let handle_size      = rt_props.shader_group_handle_size as usize;
        let handle_alignment = rt_props.shader_group_handle_alignment as usize;
        let base_alignment   = rt_props.shader_group_base_alignment as usize;

        // stride = next multiple of handle_alignment >= handle_size
        let handle_stride = align_up(handle_size, handle_alignment);

        // Each region (rgen, miss×2, hit) starts at a base_alignment boundary.
        // rgen region: 1 entry
        // miss region: 2 entries (sky + shadow)
        // hit  region: 1 entry
        let rgen_region_size  = align_up(handle_stride,          base_alignment);
        let miss_region_size  = align_up(handle_stride * 2,      base_alignment);
        let hit_region_size   = align_up(handle_stride,          base_alignment);
        let sbt_total         = rgen_region_size + miss_region_size + hit_region_size;

        // Fetch all 4 group handles at once
        let num_groups   = groups.len();
        let handles_bytes = unsafe {
            rt_loader.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                num_groups as u32,
                num_groups * handle_size,
            )?
        };

        // Allocate HOST_VISIBLE | HOST_COHERENT SBT buffer with SHADER_DEVICE_ADDRESS
        let sbt_buffer = VulkanBuffer::new_sbt(device, sbt_total as vk::DeviceSize)?;

        // Write handles into the SBT
        {
            let ptr = unsafe {
                device.device.map_memory(sbt_buffer.memory, 0, sbt_total as vk::DeviceSize, vk::MemoryMapFlags::empty())?
            } as *mut u8;
            let sbt = unsafe { std::slice::from_raw_parts_mut(ptr, sbt_total) };

            // Group 0: raygen — at offset 0
            let rgen_handle = &handles_bytes[0 * handle_size .. 1 * handle_size];
            sbt[0..handle_size].copy_from_slice(rgen_handle);

            // Group 1: sky miss — at rgen_region_size
            let miss0_handle = &handles_bytes[1 * handle_size .. 2 * handle_size];
            let miss_base = rgen_region_size;
            sbt[miss_base .. miss_base + handle_size].copy_from_slice(miss0_handle);

            // Group 2: shadow miss — at rgen_region_size + handle_stride
            let miss1_handle = &handles_bytes[2 * handle_size .. 3 * handle_size];
            let shadow_base = rgen_region_size + handle_stride;
            sbt[shadow_base .. shadow_base + handle_size].copy_from_slice(miss1_handle);

            // Group 3: closest hit — at rgen_region_size + miss_region_size
            let chit_handle = &handles_bytes[3 * handle_size .. 4 * handle_size];
            let hit_base = rgen_region_size + miss_region_size;
            sbt[hit_base .. hit_base + handle_size].copy_from_slice(chit_handle);

            unsafe { device.device.unmap_memory(sbt_buffer.memory); }
        }

        let sbt_address = sbt_buffer.device_address(&device.device);

        let rgen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address,
            stride:         handle_stride as vk::DeviceSize,
            size:           rgen_region_size as vk::DeviceSize,
        };
        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + rgen_region_size as vk::DeviceSize,
            stride:         handle_stride as vk::DeviceSize,
            size:           miss_region_size as vk::DeviceSize,
        };
        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + (rgen_region_size + miss_region_size) as vk::DeviceSize,
            stride:         handle_stride as vk::DeviceSize,
            size:           hit_region_size as vk::DeviceSize,
        };
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        Ok(Self {
            pipeline,
            pipeline_layout,
            sbt_buffer,
            rgen_region,
            miss_region,
            hit_region,
            callable_region,
            device_ref: device.device.clone(),
        })
    }
}

impl Drop for RtPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_pipeline(self.pipeline, None);
            self.device_ref.destroy_pipeline_layout(self.pipeline_layout, None);
        }
        // sbt_buffer drops automatically
    }
}

#[inline]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}
