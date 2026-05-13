use ash::vk;
use super::device::VulkanDevice;

pub struct VulkanDepthBuffer {
    pub image:      vk::Image,
    pub image_view: vk::ImageView,
    pub memory:     vk::DeviceMemory,
    pub format:     vk::Format,
    device_ref:     ash::Device,
}

impl VulkanDepthBuffer {
    pub fn new(
        instance: &ash::Instance,
        device:   &VulkanDevice,
        extent:   vk::Extent2D,
        samples:  vk::SampleCountFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let format = Self::find_depth_format(instance, device)?;

        let image_info = vk::ImageCreateInfo {
            image_type:   vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width:  extent.width,
                height: extent.height,
                depth:  1,
            },
            mip_levels:     1,
            array_layers:   1,
            format,
            tiling:         vk::ImageTiling::OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
            usage:          vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            samples,
            sharing_mode:   vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image = unsafe { device.device.create_image(&image_info, None)? };

        let mem_req  = unsafe { device.device.get_image_memory_requirements(image) };
        let mem_type = device.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:  mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };

        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_image_memory(image, mem, 0)?;
            mem
        };

        let aspect = if format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        };

        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask:      aspect,
                base_mip_level:   0,
                level_count:      1,
                base_array_layer: 0,
                layer_count:      1,
            },
            ..Default::default()
        };

        let image_view = unsafe { device.device.create_image_view(&view_info, None)? };

        Ok(Self { image, image_view, memory, format, device_ref: device.device.clone() })
    }

    fn find_depth_format(
        instance: &ash::Instance,
        device:   &VulkanDevice,
    ) -> Result<vk::Format, Box<dyn std::error::Error>> {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        for &fmt in &candidates {
            let props = unsafe {
                instance.get_physical_device_format_properties(device.physical_device, fmt)
            };
            if props.optimal_tiling_features.contains(
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            ) {
                return Ok(fmt);
            }
        }
        Err("No suitable depth format found".into())
    }
}

impl Drop for VulkanDepthBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_image_view(self.image_view, None);
            self.device_ref.destroy_image(self.image, None);
            self.device_ref.free_memory(self.memory, None);
        }
    }
}
