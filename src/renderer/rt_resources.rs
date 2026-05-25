use ash::vk;
use super::device::VulkanDevice;
use super::buffer::VulkanBuffer;
use super::acceleration_structure::AccelerationStructure;
use super::command::VulkanCommandPool;
use super::sync::MAX_FRAMES_IN_FLIGHT;

/// RGBA32F storage images and the RT descriptor sets (one per frame in flight).
pub struct RtResources {
    pub accum_image:      vk::Image,
    pub accum_image_view: vk::ImageView,
    accum_memory:         vk::DeviceMemory,

    pub out_image:      vk::Image,
    pub out_image_view: vk::ImageView,
    out_memory:         vk::DeviceMemory,

    pub descriptor_pool:    vk::DescriptorPool,
    pub descriptor_layout:  vk::DescriptorSetLayout,
    pub descriptor_sets:    Vec<vk::DescriptorSet>,  // one per frame in flight

    #[allow(dead_code)]
    pub extent: vk::Extent2D,

    device_ref: ash::Device,
}

impl RtResources {
    pub fn new(
        device:      &VulkanDevice,
        cmd_pool:    &VulkanCommandPool,
        extent:      vk::Extent2D,
        tlas:        &AccelerationStructure,
        ubo_buffers: &[VulkanBuffer],
        vertex_buf:  &VulkanBuffer,
        index_buf:   &VulkanBuffer,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        // ── Storage images ────────────────────────────────────────────────────
        let (accum_image, accum_memory) = Self::create_storage_image(device, extent)?;
        let (out_image,   out_memory)   = Self::create_storage_image(device, extent)?;

        let accum_image_view = Self::create_image_view(device, accum_image)?;
        let out_image_view   = Self::create_image_view(device, out_image)?;

        // Transition both to GENERAL
        let cb = cmd_pool.begin_single_time_commands()?;
        for &img in &[accum_image, out_image] {
            let barrier = vk::ImageMemoryBarrier {
                old_layout:          vk::ImageLayout::UNDEFINED,
                new_layout:          vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image:               img,
                subresource_range:   vk::ImageSubresourceRange {
                    aspect_mask:      vk::ImageAspectFlags::COLOR,
                    base_mip_level:   0,
                    level_count:      1,
                    base_array_layer: 0,
                    layer_count:      1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };
            unsafe {
                device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[barrier],
                );
            }
        }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;

        // ── Descriptor set layout ─────────────────────────────────────────────
        // Binding 0: TLAS
        // Binding 1: storage image (accum)
        // Binding 2: storage image (out)
        // Binding 3: uniform buffer (RT UBO)
        // Binding 4: storage buffer (vertex data)
        // Binding 5: storage buffer (index data)
        let rt_stages = vk::ShaderStageFlags::RAYGEN_KHR
                      | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                      | vk::ShaderStageFlags::ANY_HIT_KHR;

        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding:          0,
                descriptor_type:  vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          1,
                descriptor_type:  vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          2,
                descriptor_type:  vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          3,
                descriptor_type:  vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          4,
                descriptor_type:  vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding:          5,
                descriptor_type:  vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags:      rt_stages,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings:    bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_layout = unsafe {
            device.device.create_descriptor_set_layout(&layout_info, None)?
        };

        // ── Descriptor pool — enough for MAX_FRAMES_IN_FLIGHT sets ───────────
        let count = MAX_FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: count },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE,              descriptor_count: count * 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER,             descriptor_count: count },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER,             descriptor_count: count * 2 },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets:        count,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes:    pool_sizes.as_ptr(),
            ..Default::default()
        };
        let descriptor_pool = unsafe {
            device.device.create_descriptor_pool(&pool_info, None)?
        };

        // ── Allocate one descriptor set per frame in flight ───────────────────
        let layouts: Vec<vk::DescriptorSetLayout> = vec![descriptor_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: count,
            p_set_layouts:        layouts.as_ptr(),
            ..Default::default()
        };
        let descriptor_sets = unsafe {
            device.device.allocate_descriptor_sets(&alloc_info)?
        };

        // ── Write descriptors for each frame set ──────────────────────────────
        // TLAS, both images, vertex and index buffers are the same across all sets.
        // Only the UBO binding differs (each set points to its own UBO buffer).
        let tlas_handle = tlas.handle;

        let accum_info = vk::DescriptorImageInfo {
            image_view:   accum_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler:      vk::Sampler::null(),
        };
        let out_info = vk::DescriptorImageInfo {
            image_view:   out_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler:      vk::Sampler::null(),
        };
        let vb_info = vk::DescriptorBufferInfo {
            buffer: vertex_buf.buffer, offset: 0, range: vertex_buf.size,
        };
        let ib_info = vk::DescriptorBufferInfo {
            buffer: index_buf.buffer,  offset: 0, range: index_buf.size,
        };

        for (i, &set) in descriptor_sets.iter().enumerate() {
            let mut write_tlas = vk::WriteDescriptorSetAccelerationStructureKHR {
                acceleration_structure_count: 1,
                p_acceleration_structures:    &tlas_handle,
                ..Default::default()
            };
            let ubo_info = vk::DescriptorBufferInfo {
                buffer: ubo_buffers[i].buffer,
                offset: 0,
                range:  ubo_buffers[i].size,
            };
            let writes = [
                vk::WriteDescriptorSet {
                    p_next:            &mut write_tlas as *mut _ as *const std::ffi::c_void,
                    dst_set:           set,
                    dst_binding:       0,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set:           set, dst_binding: 1,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info:      &accum_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set:           set, dst_binding: 2,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info:      &out_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set:           set, dst_binding: 3,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info:     &ubo_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set:           set, dst_binding: 4,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info:     &vb_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set:           set, dst_binding: 5,
                    descriptor_count:  1,
                    descriptor_type:   vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info:     &ib_info,
                    ..Default::default()
                },
            ];
            unsafe { device.device.update_descriptor_sets(&writes, &[]) };
        }

        Ok(Self {
            accum_image,
            accum_image_view,
            accum_memory,
            out_image,
            out_image_view,
            out_memory,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            extent,
            device_ref: device.device.clone(),
        })
    }

    fn create_storage_image(
        device: &VulkanDevice,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format:     vk::Format::R32G32B32A32_SFLOAT,
            extent: vk::Extent3D {
                width:  extent.width,
                height: extent.height,
                depth:  1,
            },
            mip_levels:     1,
            array_layers:   1,
            samples:        vk::SampleCountFlags::TYPE_1,
            tiling:         vk::ImageTiling::OPTIMAL,
            usage:          vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            sharing_mode:   vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = unsafe { device.device.create_image(&image_info, None)? };
        let mem_req  = unsafe { device.device.get_image_memory_requirements(image) };
        let mem_type = device.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:   mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };
        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_image_memory(image, mem, 0)?;
            mem
        };
        Ok((image, memory))
    }

    fn create_image_view(
        device: &VulkanDevice,
        image:  vk::Image,
    ) -> Result<vk::ImageView, Box<dyn std::error::Error>> {
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format:    vk::Format::R32G32B32A32_SFLOAT,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                base_mip_level:   0,
                level_count:      1,
                base_array_layer: 0,
                layer_count:      1,
            },
            ..Default::default()
        };
        Ok(unsafe { device.device.create_image_view(&view_info, None)? })
    }
}

impl Drop for RtResources {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image_view(self.accum_image_view, None);
            self.device_ref.destroy_image_view(self.out_image_view, None);
            self.device_ref.destroy_image(self.accum_image, None);
            self.device_ref.destroy_image(self.out_image, None);
            self.device_ref.free_memory(self.accum_memory, None);
            self.device_ref.free_memory(self.out_memory, None);
            self.device_ref.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device_ref.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}
