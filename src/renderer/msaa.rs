use ash::vk;
use super::device::VulkanDevice;

pub const MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_4;

pub struct VulkanMsaaBuffer {
    pub image:      vk::Image,
    pub image_view: vk::ImageView,
    memory:         vk::DeviceMemory,
    device_ref:     ash::Device,
}

impl VulkanMsaaBuffer {
    pub fn new(
        device: &VulkanDevice,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo {
            image_type:     vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: extent.width, height: extent.height, depth: 1 },
            mip_levels:     1,
            array_layers:   1,
            format,
            tiling:         vk::ImageTiling::OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
            // TRANSIENT means the resolved result is thrown away; the GPU can keep this in tile memory.
            usage:          vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                          | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            samples:        MSAA_SAMPLES,
            sharing_mode:   vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image = unsafe { device.device.create_image(&image_info, None)? };

        let mem_req  = unsafe { device.device.get_image_memory_requirements(image) };
        let mem_type = device.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

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

        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                base_mip_level:   0,
                level_count:      1,
                base_array_layer: 0,
                layer_count:      1,
            },
            ..Default::default()
        };

        let image_view = unsafe { device.device.create_image_view(&view_info, None)? };

        Ok(Self { image, image_view, memory, device_ref: device.device.clone() })
    }
}

impl Drop for VulkanMsaaBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image_view(self.image_view, None);
            self.device_ref.destroy_image(self.image, None);
            self.device_ref.free_memory(self.memory, None);
        }
    }
}
