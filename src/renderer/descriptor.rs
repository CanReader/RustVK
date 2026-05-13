use ash::vk;
use super::device::VulkanDevice;
use super::buffer::VulkanBuffer;

pub struct VulkanDescriptorSets {
    pub layout: vk::DescriptorSetLayout,
    pub pool:   vk::DescriptorPool,
    pub sets:   Vec<vk::DescriptorSet>,
    device_ref: ash::Device,
}

impl VulkanDescriptorSets {
    pub fn new(
        device:          &VulkanDevice,
        uniform_buffers: &[VulkanBuffer],
        ubo_size:        vk::DeviceSize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let count = uniform_buffers.len() as u32;

        // Layout: binding 0 = uniform buffer, vertex + fragment
        let ubo_binding = vk::DescriptorSetLayoutBinding {
            binding:            0,
            descriptor_type:    vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count:   1,
            stage_flags:        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        };

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings:    &ubo_binding,
            ..Default::default()
        };

        let layout = unsafe { device.device.create_descriptor_set_layout(&layout_info, None)? };

        // Pool with `count` UBO descriptors
        let pool_size = vk::DescriptorPoolSize {
            ty:               vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: count,
        };
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets:        count,
            pool_size_count: 1,
            p_pool_sizes:    &pool_size,
            ..Default::default()
        };

        let pool = unsafe { device.device.create_descriptor_pool(&pool_info, None)? };

        // Allocate one set per frame
        let layouts: Vec<vk::DescriptorSetLayout> = vec![layout; count as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool:      pool,
            descriptor_set_count: count,
            p_set_layouts:        layouts.as_ptr(),
            ..Default::default()
        };

        let sets = unsafe { device.device.allocate_descriptor_sets(&alloc_info)? };

        // Write UBO descriptors
        for (i, &set) in sets.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i].buffer,
                offset: 0,
                range:  ubo_size,
            };
            let write = vk::WriteDescriptorSet {
                dst_set:            set,
                dst_binding:        0,
                dst_array_element:  0,
                descriptor_count:   1,
                descriptor_type:    vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info:      &buf_info,
                ..Default::default()
            };
            unsafe { device.device.update_descriptor_sets(&[write], &[]) };
        }

        Ok(Self { layout, pool, sets, device_ref: device.device.clone() })
    }
}

impl Drop for VulkanDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_descriptor_pool(self.pool, None);
            self.device_ref.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
