use ash::vk;
use super::device::VulkanDevice;

pub struct VulkanCommandPool {
    pub pool:            vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    device_ref:          ash::Device,
}

impl VulkanCommandPool {
    pub fn new(device: &VulkanDevice, frames_in_flight: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let pool_info = vk::CommandPoolCreateInfo {
            flags:              vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: device.queue_families.graphics_family,
            ..Default::default()
        };

        let pool = unsafe { device.device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool:        pool,
            level:               vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: frames_in_flight as u32,
            ..Default::default()
        };

        let command_buffers = unsafe { device.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self { pool, command_buffers, device_ref: device.device.clone() })
    }

    /// Allocate and begin a one-shot command buffer for synchronous transfers.
    pub fn begin_single_time_commands(&self) -> Result<vk::CommandBuffer, Box<dyn std::error::Error>> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool:        self.pool,
            level:               vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let cb = unsafe { self.device_ref.allocate_command_buffers(&alloc_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { self.device_ref.begin_command_buffer(cb, &begin_info)? };

        Ok(cb)
    }

    /// End, submit, wait for idle, and free a one-shot command buffer.
    pub fn end_single_time_commands(
        &self,
        cb:    vk::CommandBuffer,
        queue: vk::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device_ref.end_command_buffer(cb)?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers:    &cb,
                ..Default::default()
            };

            self.device_ref.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            self.device_ref.queue_wait_idle(queue)?;
            self.device_ref.free_command_buffers(self.pool, &[cb]);
        }
        Ok(())
    }
}

impl Drop for VulkanCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_command_pool(self.pool, None);
        }
    }
}
