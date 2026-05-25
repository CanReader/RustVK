use ash::vk;
use super::device::VulkanDevice;
use super::buffer::VulkanBuffer;
use super::command::VulkanCommandPool;

/// Owns a VkAccelerationStructureKHR, its backing buffer, and its device address.
pub struct AccelerationStructure {
    pub handle:         vk::AccelerationStructureKHR,
    #[allow(dead_code)]
    pub buffer:         VulkanBuffer,   // kept for lifetime / drop
    pub device_address: vk::DeviceAddress,
    #[allow(dead_code)]
    device_ref:         ash::Device,
    accel_loader:       ash::khr::acceleration_structure::Device,
}

impl AccelerationStructure {
    /// Build a BLAS from a set of triangles.
    ///
    /// `vertex_address`  — device address of the vertex buffer (float data)
    /// `vertex_count`    — number of vertices
    /// `vertex_stride`   — stride in bytes (48 for our Vertex)
    /// `index_address`   — device address of the index buffer
    /// `index_count`     — number of indices (must be a multiple of 3)
    pub fn build_blas(
        device:           &VulkanDevice,
        cmd_pool:         &VulkanCommandPool,
        vertex_address:   vk::DeviceAddress,
        vertex_count:     u32,
        vertex_stride:    vk::DeviceSize,
        index_address:    vk::DeviceAddress,
        index_count:      u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let accel_loader = device.accel_loader.as_ref()
            .ok_or("RT not supported")?;

        // Triangle geometry description
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR {
            vertex_format:  vk::Format::R32G32B32_SFLOAT,
            vertex_data:    vk::DeviceOrHostAddressConstKHR { device_address: vertex_address },
            vertex_stride,
            max_vertex:     vertex_count - 1,
            index_type:     vk::IndexType::UINT32,
            index_data:     vk::DeviceOrHostAddressConstKHR { device_address: index_address },
            transform_data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
            ..Default::default()
        };

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::TRIANGLES,
            geometry: vk::AccelerationStructureGeometryDataKHR { triangles },
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        };

        let primitive_count = index_count / 3;

        // Query sizes
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty:    vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode:  vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries:   &geometry,
            ..Default::default()
        };

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[primitive_count],
                &mut size_info,
            );
        }

        // Backing buffer for the AS
        let as_buffer = VulkanBuffer::new_rt_device_local(
            device,
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        // Create the AS
        let as_create_info = vk::AccelerationStructureCreateInfoKHR {
            buffer: as_buffer.buffer,
            size:   size_info.acceleration_structure_size,
            ty:     vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            ..Default::default()
        };
        let handle = unsafe { accel_loader.create_acceleration_structure(&as_create_info, None)? };

        // Scratch buffer
        let scratch = VulkanBuffer::new_rt_device_local(
            device,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let scratch_address = scratch.device_address(&device.device);

        // Build
        let mut build_info_filled = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty:    vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode:  vk::BuildAccelerationStructureModeKHR::BUILD,
            dst_acceleration_structure: handle,
            geometry_count: 1,
            p_geometries:   &geometry,
            scratch_data:   vk::DeviceOrHostAddressKHR { device_address: scratch_address },
            ..Default::default()
        };

        let range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        };

        let cb = cmd_pool.begin_single_time_commands()?;
        unsafe {
            accel_loader.cmd_build_acceleration_structures(
                cb,
                std::slice::from_ref(&build_info_filled),
                &[&[range_info]],
            );
        }
        // Memory barrier: AS build -> shader read
        let barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            ..Default::default()
        };
        unsafe {
            device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;

        // Fetch device address of the completed AS
        let addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
            acceleration_structure: handle,
            ..Default::default()
        };
        let device_address = unsafe {
            accel_loader.get_acceleration_structure_device_address(&addr_info)
        };

        // Suppress unused warning for build_info_filled
        let _ = &mut build_info_filled;

        Ok(Self {
            handle,
            buffer: as_buffer,
            device_address,
            device_ref: device.device.clone(),
            accel_loader: accel_loader.clone(),
        })
    }

    /// Build a TLAS containing a single identity-transform instance of the given BLAS.
    pub fn build_tlas(
        device:       &VulkanDevice,
        cmd_pool:     &VulkanCommandPool,
        blas_address: vk::DeviceAddress,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let accel_loader = device.accel_loader.as_ref()
            .ok_or("RT not supported")?;

        // Single instance with identity transform
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),  // customIndex=0, mask=0xFF
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),  // sbtOffset=0, flags=0
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_address,
            },
        };

        // Upload instance buffer with DEVICE_ADDRESS flag
        let instance_bytes = unsafe {
            std::slice::from_raw_parts(
                &instance as *const _ as *const u8,
                std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
            )
        };
        let instance_size = instance_bytes.len() as vk::DeviceSize;

        // Staging
        let staging = VulkanBuffer::new(
            device, instance_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = device.device.map_memory(staging.memory, 0, instance_size, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping(instance_bytes.as_ptr(), ptr, instance_bytes.len());
            device.device.unmap_memory(staging.memory);
        }

        let instance_buf = VulkanBuffer::new_rt_device_local(
            device,
            instance_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let cb = cmd_pool.begin_single_time_commands()?;
        let copy = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: instance_size };
        unsafe { device.device.cmd_copy_buffer(cb, staging.buffer, instance_buf.buffer, &[copy]); }
        // Barrier: transfer write -> AS build read
        let buf_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: instance_buf.buffer,
            offset: 0,
            size:   vk::WHOLE_SIZE,
            ..Default::default()
        };
        unsafe {
            device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[], &[buf_barrier], &[],
            );
        }

        let instance_address = instance_buf.device_address(&device.device);

        // Geometry
        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR {
            array_of_pointers: vk::FALSE,
            data: vk::DeviceOrHostAddressConstKHR { device_address: instance_address },
            ..Default::default()
        };
        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR { instances: instances_data },
            ..Default::default()
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty:    vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode:  vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries:   &geometry,
            ..Default::default()
        };

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[1u32],  // 1 instance
                &mut size_info,
            );
        }

        let as_buffer = VulkanBuffer::new_rt_device_local(
            device,
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let as_create_info = vk::AccelerationStructureCreateInfoKHR {
            buffer: as_buffer.buffer,
            size:   size_info.acceleration_structure_size,
            ty:     vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            ..Default::default()
        };
        let handle = unsafe { accel_loader.create_acceleration_structure(&as_create_info, None)? };

        let scratch = VulkanBuffer::new_rt_device_local(
            device,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let scratch_address = scratch.device_address(&device.device);

        let build_info_filled = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty:    vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode:  vk::BuildAccelerationStructureModeKHR::BUILD,
            dst_acceleration_structure: handle,
            geometry_count: 1,
            p_geometries:   &geometry,
            scratch_data:   vk::DeviceOrHostAddressKHR { device_address: scratch_address },
            ..Default::default()
        };

        let range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: 1,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        };

        unsafe {
            accel_loader.cmd_build_acceleration_structures(
                cb,
                std::slice::from_ref(&build_info_filled),
                &[&[range_info]],
            );
        }
        // Final barrier: AS build write -> RT shader read
        let barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            ..Default::default()
        };
        unsafe {
            device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
        cmd_pool.end_single_time_commands(cb, device.graphics_queue)?;

        let addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
            acceleration_structure: handle,
            ..Default::default()
        };
        let device_address = unsafe {
            accel_loader.get_acceleration_structure_device_address(&addr_info)
        };

        Ok(Self {
            handle,
            buffer: as_buffer,
            device_address,
            device_ref: device.device.clone(),
            accel_loader: accel_loader.clone(),
        })
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.accel_loader.destroy_acceleration_structure(self.handle, None);
        }
        // self.buffer is dropped automatically
    }
}
