use ash::vk;
use super::device::VulkanDevice;

pub struct VulkanSwapchain {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain:        vk::SwapchainKHR,
    pub images:           Vec<vk::Image>,
    pub image_views:      Vec<vk::ImageView>,
    pub format:           vk::Format,
    pub extent:           vk::Extent2D,
    device_ref:           ash::Device,
}

impl VulkanSwapchain {
    pub fn new(
        instance:       &ash::Instance,
        device:         &VulkanDevice,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
        width:          u32,
        height:         u32,
        // Pass the previous swapchain so the ICD can retire it (and transfer Wayland
        // protocol objects like wp_tearing_control_v1) instead of creating duplicates.
        old_swapchain:  vk::SwapchainKHR,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(device.physical_device, surface)?
        };
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(device.physical_device, surface)?
        };
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(device.physical_device, surface)?
        };

        let format   = Self::choose_format(&formats);
        let mode     = Self::choose_present_mode(&modes);
        let extent   = Self::choose_extent(&caps, width, height);
        let img_count = Self::choose_image_count(&caps);

        let queue_family_indices = device.queue_families.unique_families();
        let (sharing_mode, qfi_count, qfi_ptr) = if queue_family_indices.len() > 1 {
            (
                vk::SharingMode::CONCURRENT,
                queue_family_indices.len() as u32,
                queue_family_indices.as_ptr(),
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, std::ptr::null())
        };

        let create_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count:    img_count,
            image_format:       format.format,
            image_color_space:  format.color_space,
            image_extent:       extent,
            image_array_layers: 1,
            image_usage:        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: qfi_count,
            p_queue_family_indices:   qfi_ptr,
            pre_transform:      caps.current_transform,
            composite_alpha:    vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode:       mode,
            clipped:            vk::TRUE,
            old_swapchain,
            ..Default::default()
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, &device.device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let image_views = images.iter().map(|&img| {
            let view_info = vk::ImageViewCreateInfo {
                image: img,
                view_type:  vk::ImageViewType::TYPE_2D,
                format:     format.format,
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
            unsafe { device.device.create_image_view(&view_info, None) }
        }).collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format: format.format,
            extent,
            device_ref: device.device.clone(),
        })
    }

    fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats.iter().copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(formats[0])
    }

    fn choose_present_mode(_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        // FIFO is always available and avoids the Wayland wp_tearing_control_v1
        // protocol conflict that MAILBOX triggers on some compositors.
        vk::PresentModeKHR::FIFO
    }

    fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR, width: u32, height: u32) -> vk::Extent2D {
        if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width:  width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        }
    }

    fn choose_image_count(caps: &vk::SurfaceCapabilitiesKHR) -> u32 {
        let desired = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            desired.min(caps.max_image_count)
        } else {
            desired
        }
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            for &view in &self.image_views {
                self.device_ref.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
