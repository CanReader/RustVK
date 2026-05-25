use ash::{vk, Entry, Instance};
use raw_window_handle::HasDisplayHandle;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;

/// Wraps the Vulkan instance, entry point, and optional debug messenger.
pub struct VulkanInstance {
    pub entry:    Entry,
    pub instance: Instance,
    debug_utils:  Option<ash::ext::debug_utils::Instance>,
    messenger:    Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanInstance {
    pub fn new(
        window:            &winit::window::Window,
        enable_validation: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };

        let app_name    = CString::new("RustVK").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        // Required surface extensions from winit
        let mut extensions = ash_window::enumerate_required_extensions(
            window.display_handle()?.as_raw()
        )?.to_vec();

        if enable_validation {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layer_names: Vec<*const i8> = if enable_validation {
            if Self::check_validation_layer_support(&entry) {
                vec![validation_layer.as_ptr()]
            } else {
                log::warn!("VK_LAYER_KHRONOS_validation not available, running without validation");
                vec![]
            }
        } else {
            vec![]
        };

        let debug_create_info = Self::debug_messenger_create_info();

        // p_next chain: attach debug messenger to instance creation if validation enabled
        let p_next: *const c_void = if enable_validation && !layer_names.is_empty() {
            &debug_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
        } else {
            std::ptr::null()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count:   extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            enabled_layer_count:       layer_names.len() as u32,
            pp_enabled_layer_names:    layer_names.as_ptr(),
            p_next,
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let (debug_utils, messenger) = if enable_validation && !layer_names.is_empty() {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe {
                du.create_debug_utils_messenger(&Self::debug_messenger_create_info(), None)?
            };
            (Some(du), Some(messenger))
        } else {
            (None, None)
        };

        Ok(Self { entry, instance, debug_utils, messenger })
    }

    fn check_validation_layer_support(entry: &Entry) -> bool {
        let layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let available = unsafe { entry.enumerate_instance_layer_properties() }
            .unwrap_or_default();
        available.iter().any(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
            name == layer_name.as_c_str()
        })
    }

    fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity:
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            message_type:
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(debug_callback),
            ..Default::default()
        }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(du), Some(msg)) = (self.debug_utils.take(), self.messenger.take()) {
                du.destroy_debug_utils_messenger(msg, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity:   vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type:  vk::DebugUtilsMessageTypeFlagsEXT,
    data:       *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {}", msg);
    } else {
        log::warn!("[Vulkan] {}", msg);
    }
    vk::FALSE
}
