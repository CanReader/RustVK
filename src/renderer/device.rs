use ash::{vk, Device, Instance};

/// Queue family indices selected for graphics and presentation.
#[derive(Clone, Debug)]
pub struct QueueFamilies {
    pub graphics_family: u32,
    pub present_family:  u32,
}

impl QueueFamilies {
    pub fn unique_families(&self) -> Vec<u32> {
        let mut v = vec![self.graphics_family];
        if self.present_family != self.graphics_family {
            v.push(self.present_family);
        }
        v
    }
}

/// Wraps the physical device, logical device, and selected queues.
pub struct VulkanDevice {
    pub physical_device: vk::PhysicalDevice,
    pub device:          Device,
    pub graphics_queue:  vk::Queue,
    pub present_queue:   vk::Queue,
    pub queue_families:  QueueFamilies,
    pub mem_properties:  vk::PhysicalDeviceMemoryProperties,
}

impl VulkanDevice {
    pub fn new(
        instance:       &Instance,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let physical_device = Self::pick_physical_device(instance, surface_loader, surface)?;
        let queue_families  = Self::find_queue_families(instance, physical_device, surface_loader, surface)?;

        let queue_priority = 1.0_f32;
        let unique = queue_families.unique_families();
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique.iter().map(|&qf| {
            vk::DeviceQueueCreateInfo {
                queue_family_index: qf,
                queue_count:        1,
                p_queue_priorities: &queue_priority,
                ..Default::default()
            }
        }).collect();

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

        let features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count:    queue_create_infos.len() as u32,
            p_queue_create_infos:       queue_create_infos.as_ptr(),
            enabled_extension_count:    device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features:         &features,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_family, 0) };
        let present_queue  = unsafe { device.get_device_queue(queue_families.present_family, 0) };

        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            physical_device,
            device,
            graphics_queue,
            present_queue,
            queue_families,
            mem_properties,
        })
    }

    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties:  vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        for i in 0..self.mem_properties.memory_type_count {
            let mem_type = self.mem_properties.memory_types[i as usize];
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(properties) {
                return Ok(i);
            }
        }
        // Fallback: any type with the required properties
        for i in 0..self.mem_properties.memory_type_count {
            let mem_type = self.mem_properties.memory_types[i as usize];
            if mem_type.property_flags.contains(properties) {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".into())
    }

    fn pick_physical_device(
        instance:       &Instance,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
    ) -> Result<vk::PhysicalDevice, Box<dyn std::error::Error>> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        if devices.is_empty() {
            return Err("No Vulkan-capable physical devices found".into());
        }

        let mut chosen   = None;
        let mut fallback = None;
        for &pd in &devices {
            if !Self::device_is_suitable(instance, pd, surface_loader, surface) {
                continue;
            }
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                chosen = Some(pd);
                break;
            }
            fallback = Some(pd);
        }

        chosen.or(fallback).ok_or_else(|| "No suitable physical device found".into())
    }

    fn device_is_suitable(
        instance:       &Instance,
        device:         vk::PhysicalDevice,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
    ) -> bool {
        let ext_ok = unsafe { instance.enumerate_device_extension_properties(device) }
            .map(|exts| {
                let swapchain_ext = ash::khr::swapchain::NAME;
                exts.iter().any(|e| unsafe {
                    std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == swapchain_ext
                })
            })
            .unwrap_or(false);

        if !ext_ok { return false; }

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(device, surface)
        }.unwrap_or_default();
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(device, surface)
        }.unwrap_or_default();

        !formats.is_empty() && !modes.is_empty()
    }

    fn find_queue_families(
        instance:       &Instance,
        device:         vk::PhysicalDevice,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
    ) -> Result<QueueFamilies, Box<dyn std::error::Error>> {
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut graphics = None;
        let mut present  = None;

        for (i, fam) in families.iter().enumerate() {
            if fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics = Some(i as u32);
            }
            let supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(device, i as u32, surface)?
            };
            if supports_present {
                present = Some(i as u32);
            }
            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        match (graphics, present) {
            (Some(g), Some(p)) => Ok(QueueFamilies { graphics_family: g, present_family: p }),
            _ => Err("Failed to find required queue families".into()),
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
