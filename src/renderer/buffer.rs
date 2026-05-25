use ash::vk;
use super::device::VulkanDevice;
use super::command::VulkanCommandPool;
use crate::scene::Vertex;

pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size:   vk::DeviceSize,
    device_ref: ash::Device,
}

impl VulkanBuffer {
    /// Create a raw buffer with the given usage and memory properties.
    pub fn new(
        device:     &VulkanDevice,
        size:       vk::DeviceSize,
        usage:      vk::BufferUsageFlags,
        mem_props:  vk::MemoryPropertyFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.device.create_buffer(&buffer_info, None)? };

        let mem_req  = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let mem_type = device.find_memory_type(mem_req.memory_type_bits, mem_props)?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:   mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };

        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_buffer_memory(buffer, mem, 0)?;
            mem
        };

        Ok(Self { buffer, memory, size, device_ref: device.device.clone() })
    }

    /// Vertex buffer: staged upload to DEVICE_LOCAL memory.
    pub fn new_vertex(
        device:   &VulkanDevice,
        cmd_pool: &VulkanCommandPool,
        data:     &[Vertex],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice(data);
        Self::upload_via_staging(device, cmd_pool, bytes, vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    /// Index buffer: staged upload to DEVICE_LOCAL memory.
    pub fn new_index(
        device:   &VulkanDevice,
        cmd_pool: &VulkanCommandPool,
        data:     &[u32],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice(data);
        Self::upload_via_staging(device, cmd_pool, bytes, vk::BufferUsageFlags::INDEX_BUFFER)
    }

    /// Uniform buffer: HOST_VISIBLE | HOST_COHERENT, no staging needed.
    pub fn new_uniform(
        device: &VulkanDevice,
        size:   usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(
            device,
            size as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Query the device address of this buffer (requires SHADER_DEVICE_ADDRESS usage).
    pub fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo {
            buffer: self.buffer,
            ..Default::default()
        };
        unsafe { device.get_buffer_device_address(&info) }
    }

    /// Create a DEVICE_LOCAL buffer with SHADER_DEVICE_ADDRESS memory allocation flag.
    /// Used for RT acceleration structure inputs and the SBT.
    pub fn new_rt_device_local(
        device: &VulkanDevice,
        size:   vk::DeviceSize,
        usage:  vk::BufferUsageFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.device.create_buffer(&buffer_info, None)? };
        let mem_req = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let mem_type = device.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Chain MemoryAllocateFlagsInfo to request DEVICE_ADDRESS
        let mut alloc_flags = vk::MemoryAllocateFlagsInfo {
            flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            ..Default::default()
        };
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:   mem_req.size,
            memory_type_index: mem_type,
            p_next: &mut alloc_flags as *mut _ as *const std::ffi::c_void,
            ..Default::default()
        };
        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_buffer_memory(buffer, mem, 0)?;
            mem
        };
        Ok(Self { buffer, memory, size, device_ref: device.device.clone() })
    }

    /// Create an RT-compatible vertex buffer (DEVICE_LOCAL + device address + AS build input + STORAGE).
    pub fn new_rt_vertex(
        device:   &VulkanDevice,
        cmd_pool: &VulkanCommandPool,
        data:     &[Vertex],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice::<Vertex, u8>(data);
        let size  = bytes.len() as vk::DeviceSize;

        let staging = Self::new(
            device, size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = device.device.map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            device.device.unmap_memory(staging.memory);
        }

        let final_usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let gpu_buf = Self::new_rt_device_local(device, size, final_usage)?;

        let cb = cmd_pool.begin_single_time_commands()?;
        let copy_region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
        unsafe { device.device.cmd_copy_buffer(cb, staging.buffer, gpu_buf.buffer, &[copy_region]); }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;
        Ok(gpu_buf)
    }

    /// Create an RT-compatible index buffer (DEVICE_LOCAL + device address + AS build input + STORAGE).
    pub fn new_rt_index(
        device:   &VulkanDevice,
        cmd_pool: &VulkanCommandPool,
        data:     &[u32],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice::<u32, u8>(data);
        let size  = bytes.len() as vk::DeviceSize;

        let staging = Self::new(
            device, size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = device.device.map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            device.device.unmap_memory(staging.memory);
        }

        let final_usage = vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let gpu_buf = Self::new_rt_device_local(device, size, final_usage)?;

        let cb = cmd_pool.begin_single_time_commands()?;
        let copy_region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
        unsafe { device.device.cmd_copy_buffer(cb, staging.buffer, gpu_buf.buffer, &[copy_region]); }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;
        Ok(gpu_buf)
    }

    /// Create a HOST_VISIBLE | HOST_COHERENT buffer with SHADER_DEVICE_ADDRESS.
    /// Used for the shader binding table.
    pub fn new_sbt(
        device: &VulkanDevice,
        size:   vk::DeviceSize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                 | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.device.create_buffer(&buffer_info, None)? };
        let mem_req  = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let mem_type = device.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mut alloc_flags = vk::MemoryAllocateFlagsInfo {
            flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            ..Default::default()
        };
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size:   mem_req.size,
            memory_type_index: mem_type,
            p_next: &mut alloc_flags as *mut _ as *const std::ffi::c_void,
            ..Default::default()
        };
        let memory = unsafe {
            let mem = device.device.allocate_memory(&alloc_info, None)?;
            device.device.bind_buffer_memory(buffer, mem, 0)?;
            mem
        };
        Ok(Self { buffer, memory, size, device_ref: device.device.clone() })
    }

    /// Upload raw bytes to a DEVICE_LOCAL buffer via a staging buffer.
    pub fn upload_via_staging(
        device:      &VulkanDevice,
        cmd_pool:    &VulkanCommandPool,
        data:        &[u8],
        final_usage: vk::BufferUsageFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = data.len() as vk::DeviceSize;

        // --- CPU staging buffer ---
        let staging = Self::new(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let ptr = device.device.map_memory(
                staging.memory, 0, size, vk::MemoryMapFlags::empty()
            )? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            device.device.unmap_memory(staging.memory);
        }

        // --- GPU-local destination buffer ---
        let gpu_buf = Self::new(
            device,
            size,
            final_usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy staging → GPU
        let cb = cmd_pool.begin_single_time_commands()?;
        let copy_region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
        unsafe {
            device.device.cmd_copy_buffer(cb, staging.buffer, gpu_buf.buffer, &[copy_region]);
        }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;

        // `staging` is dropped here — frees its buffer and memory
        Ok(gpu_buf)
    }

    /// Write new data into a persistently-mapped host-coherent uniform buffer.
    pub fn update_uniform<T: bytemuck::Pod>(
        &self,
        device: &VulkanDevice,
        data:   &T,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bytemuck::bytes_of(data);
        unsafe {
            let ptr = device.device.map_memory(
                self.memory, 0, self.size, vk::MemoryMapFlags::empty()
            )? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            device.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device_ref.destroy_buffer(self.buffer, None);
            self.device_ref.free_memory(self.memory, None);
        }
    }
}
