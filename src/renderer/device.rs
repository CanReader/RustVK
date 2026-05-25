use ash::{vk, Device, Instance};
use std::ffi::CStr;

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
    pub physical_device:    vk::PhysicalDevice,
    pub device:             Device,
    pub graphics_queue:     vk::Queue,
    pub present_queue:      vk::Queue,
    pub queue_families:     QueueFamilies,
    pub mem_properties:     vk::PhysicalDeviceMemoryProperties,

    // Ray tracing support (None when the extensions are absent)
    pub rt_supported:       bool,
    pub accel_loader:       Option<ash::khr::acceleration_structure::Device>,
    pub rt_pipeline_loader: Option<ash::khr::ray_tracing_pipeline::Device>,
    // 'static lifetime: the struct owns its own memory so the borrow is always valid.
    pub rt_props:           vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
}

impl VulkanDevice {
    pub fn new(
        instance:       &Instance,
        surface_loader: &ash::khr::surface::Instance,
        surface:        vk::SurfaceKHR,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let physical_device = Self::pick_physical_device(instance, surface_loader, surface)?;
        let queue_families  = Self::find_queue_families(instance, physical_device, surface_loader, surface)?;

        // Probe extension support
        let available_exts = unsafe {
            instance.enumerate_device_extension_properties(physical_device)?
        };
        let has_ext = |name: &CStr| -> bool {
            available_exts.iter().any(|e| unsafe {
                CStr::from_ptr(e.extension_name.as_ptr()) == name
            })
        };
        let rt_available = has_ext(ash::khr::acceleration_structure::NAME)
            && has_ext(ash::khr::ray_tracing_pipeline::NAME)
            && has_ext(ash::khr::deferred_host_operations::NAME)
            && has_ext(ash::khr::buffer_device_address::NAME);

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

        let device = if rt_available {
            // ── RT path: pNext feature chain ─────────────────────────────────
            // Build chain manually via raw p_next pointers.
            // Order: DeviceCreateInfo.p_next -> features2
            //        features2.p_next        -> bda_features
            //        bda_features.p_next     -> accel_features
            //        accel_features.p_next   -> rt_features  -> null

            let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures {
                buffer_device_address: vk::TRUE,
                ..Default::default()
            };
            let mut accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                acceleration_structure: vk::TRUE,
                ..Default::default()
            };
            let mut rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
                ray_tracing_pipeline: vk::TRUE,
                ..Default::default()
            };

            // Build pNext chain manually. The Vulkan driver reads these via raw pointers,
            // so Rust's "value assigned but never read" warning is a false positive.
            #[allow(unused_assignments)]
            {
                bda_features.p_next   = &mut accel_features as *mut _ as *mut std::ffi::c_void;
                accel_features.p_next = &mut rt_features    as *mut _ as *mut std::ffi::c_void;
                // rt_features.p_next stays null (already default)
            }

            // features2 holds the base Vulkan features; we delegate to the pNext chain
            let features2 = vk::PhysicalDeviceFeatures2 {
                p_next: &mut bda_features as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };

            let mut device_extensions: Vec<*const std::os::raw::c_char> = vec![
                ash::khr::swapchain::NAME.as_ptr(),
                ash::khr::deferred_host_operations::NAME.as_ptr(),
                ash::khr::acceleration_structure::NAME.as_ptr(),
                ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
                ash::khr::buffer_device_address::NAME.as_ptr(),
            ];
            if has_ext(ash::khr::spirv_1_4::NAME) {
                device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
            }

            // NOTE: p_enabled_features MUST be null when using PhysicalDeviceFeatures2 in pNext
            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count:    queue_create_infos.len() as u32,
                p_queue_create_infos:       queue_create_infos.as_ptr(),
                enabled_extension_count:    device_extensions.len() as u32,
                pp_enabled_extension_names: device_extensions.as_ptr(),
                p_enabled_features:         std::ptr::null(),
                p_next: &features2 as *const _ as *const std::ffi::c_void,
                ..Default::default()
            };

            unsafe { instance.create_device(physical_device, &create_info, None)? }
        } else {
            // ── Rasterizer-only path ──────────────────────────────────────────
            let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let features = vk::PhysicalDeviceFeatures { ..Default::default() };

            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count:    queue_create_infos.len() as u32,
                p_queue_create_infos:       queue_create_infos.as_ptr(),
                enabled_extension_count:    device_extensions.len() as u32,
                pp_enabled_extension_names: device_extensions.as_ptr(),
                p_enabled_features:         &features,
                ..Default::default()
            };

            unsafe { instance.create_device(physical_device, &create_info, None)? }
        };

        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_family, 0) };
        let present_queue  = unsafe { device.get_device_queue(queue_families.present_family, 0) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Build RT loader instances and fetch RT pipeline properties
        let (accel_loader, rt_pipeline_loader, rt_props) = if rt_available {
            let accel_loader    = ash::khr::acceleration_structure::Device::new(instance, &device);
            let rt_pipe_loader  = ash::khr::ray_tracing_pipeline::Device::new(instance, &device);

            // Query VkPhysicalDeviceRayTracingPipelinePropertiesKHR via pNext chain
            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2   = vk::PhysicalDeviceProperties2 {
                p_next: &mut rt_props as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe { instance.get_physical_device_properties2(physical_device, &mut props2); }

            (Some(accel_loader), Some(rt_pipe_loader), rt_props)
        } else {
            (None, None, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default())
        };

        Ok(Self {
            physical_device,
            device,
            graphics_queue,
            present_queue,
            queue_families,
            mem_properties,
            rt_supported: rt_available,
            accel_loader,
            rt_pipeline_loader,
            rt_props,
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
