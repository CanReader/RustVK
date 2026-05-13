use ash::vk;
use super::device::VulkanDevice;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

// Extra acquire semaphore so there is always one that is not pending a
// presentation-engine signal for a previously-acquired image.
const NUM_ACQUIRE_SEMS: usize = MAX_FRAMES_IN_FLIGHT + 1;

pub struct VulkanSync {
    // One per swapchain image: used as signal in submit and wait in present.
    // Indexing by image_index (not frame) ensures the semaphore is only
    // reused after the presentation engine has re-released that image slot.
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    // Rotating pool of acquire semaphores (NUM_ACQUIRE_SEMS entries).
    pub image_available_semaphores: Vec<vk::Semaphore>,

    // One fence per frame in flight.
    pub in_flight_fences: Vec<vk::Fence>,

    device_ref: ash::Device,
}

impl VulkanSync {
    pub fn new(
        device:             &VulkanDevice,
        num_swapchain_images: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sem_info   = vk::SemaphoreCreateInfo { ..Default::default() };
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let mut render_finished = Vec::with_capacity(num_swapchain_images);
        let mut image_available = Vec::with_capacity(NUM_ACQUIRE_SEMS);
        let mut in_flight       = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..num_swapchain_images {
            unsafe {
                render_finished.push(device.device.create_semaphore(&sem_info, None)?);
            }
        }
        for _ in 0..NUM_ACQUIRE_SEMS {
            unsafe {
                image_available.push(device.device.create_semaphore(&sem_info, None)?);
            }
        }
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                in_flight.push(device.device.create_fence(&fence_info, None)?);
            }
        }

        Ok(Self {
            render_finished_semaphores: render_finished,
            image_available_semaphores: image_available,
            in_flight_fences:           in_flight,
            device_ref:                 device.device.clone(),
        })
    }
}

impl Drop for VulkanSync {
    fn drop(&mut self) {
        unsafe {
            for &s in &self.render_finished_semaphores {
                self.device_ref.destroy_semaphore(s, None);
            }
            for &s in &self.image_available_semaphores {
                self.device_ref.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight_fences {
                self.device_ref.destroy_fence(f, None);
            }
        }
    }
}
