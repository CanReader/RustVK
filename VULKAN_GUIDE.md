# Vulkan From Zero — A Complete Guide Using SleakCraft

This guide teaches the entire Vulkan API from scratch, using the real code inside
`Engine/Engine/src/VulkanRenderer.cpp` and its companion files as living examples.
Every concept is tied to what you can read in this codebase.

---

## Table of Contents

1. [Why Vulkan Exists](#1-why-vulkan-exists)
2. [The Vulkan Mental Model](#2-the-vulkan-mental-model)
3. [Instance — The Root Object](#3-instance--the-root-object)
4. [Validation Layers and the Debug Messenger](#4-validation-layers-and-the-debug-messenger)
5. [Surface — Connecting Vulkan to the Window](#5-surface--connecting-vulkan-to-the-window)
6. [Physical Device — Picking the GPU](#6-physical-device--picking-the-gpu)
7. [Logical Device and Queue Families](#7-logical-device-and-queue-families)
8. [Swapchain — Image Presentation](#8-swapchain--image-presentation)
9. [Image Views — How to Read an Image](#9-image-views--how-to-read-an-image)
10. [Render Passes — The Rendering Contract](#10-render-passes--the-rendering-contract)
11. [Framebuffers — The Render Target](#11-framebuffers--the-render-target)
12. [Shaders and SPIR-V](#12-shaders-and-spir-v)
13. [Descriptor Set Layouts — What Shaders Declare They Need](#13-descriptor-set-layouts--what-shaders-declare-they-need)
14. [Descriptor Pools and Descriptor Sets — Binding Resources](#14-descriptor-pools-and-descriptor-sets--binding-resources)
15. [Pipeline Layout — The Glue Between Pipeline and Descriptors](#15-pipeline-layout--the-glue-between-pipeline-and-descriptors)
16. [Graphics Pipeline — The Giant Baked Object](#16-graphics-pipeline--the-giant-baked-object)
17. [Command Pools and Command Buffers](#17-command-pools-and-command-buffers)
18. [GPU Memory — Understanding Memory Types](#18-gpu-memory--understanding-memory-types)
19. [Buffers — Vertex, Index, Uniform, and Staging](#19-buffers--vertex-index-uniform-and-staging)
20. [Images and Samplers — Textures in Vulkan](#20-images-and-samplers--textures-in-vulkan)
21. [Image Layout Transitions](#21-image-layout-transitions)
22. [Synchronization — Fences, Semaphores, and Pipeline Barriers](#22-synchronization--fences-semaphores-and-pipeline-barriers)
23. [Multi-Frame-in-Flight — Overlapping CPU and GPU Work](#23-multi-frame-in-flight--overlapping-cpu-and-gpu-work)
24. [Depth Testing](#24-depth-testing)
25. [MSAA — Multisampled Anti-Aliasing](#25-msaa--multisampled-anti-aliasing)
26. [Push Constants — Fast Per-Draw Data](#26-push-constants--fast-per-draw-data)
27. [The Frame Loop — Putting Everything Together](#27-the-frame-loop--putting-everything-together)
28. [Swapchain Recreation — Handling Window Resize](#28-swapchain-recreation--handling-window-resize)
29. [Advanced: Deferred Rendering and the GBuffer](#29-advanced-deferred-rendering-and-the-gbuffer)
30. [Advanced: Shadow Mapping](#30-advanced-shadow-mapping)
31. [Advanced: Specialized Pipelines](#31-advanced-specialized-pipelines)
32. [Cleanup — Destroying Everything in the Right Order](#32-cleanup--destroying-everything-in-the-right-order)

---

## 1. Why Vulkan Exists

Before Vulkan, OpenGL was the dominant cross-platform graphics API. OpenGL was designed
in the early 1990s for a very different world: single-threaded CPUs, GPUs that had almost
no programmable logic, and operating systems that owned the driver. The OpenGL driver
became enormous over decades. It held invisible global state, compiled your shaders
lazily at draw time (causing stutters), guessed your intent, and validated everything you
did at runtime.

Vulkan was released in 2016 as the answer to several hard problems:

**Global state caused bugs that were invisible until a completely different draw call.**
In OpenGL every state change — current texture, blend mode, depth test — was global.
Binding a texture for object A silently affected object B if you forgot to rebind.

**The driver hid too much.** The driver decided when to compile shaders, when to allocate
memory, how to handle threading. On good hardware this was fine. On mobile, consoles, or
anything outside desktop Windows+NVIDIA it could be catastrophic.

**Multi-threading was impossible.** OpenGL had one context per thread, and sharing
resources across threads required complex extensions and fence objects bolted on late.

**Performance was unpredictable.** The driver validated every draw call. A production
game could spend 30% of its CPU time in the driver doing things that were entirely
unnecessary.

Vulkan fixes all of this by moving responsibility to the application:

- You create and destroy every object explicitly.
- You describe rendering operations up front as "render passes" so the driver can
  optimize at creation time, not at draw time.
- You record commands into command buffers from any thread you like, then submit
  them to queues.
- You manage memory yourself (or delegate to a library like VMA).
- You synchronize CPU and GPU work yourself with explicit fences and semaphores.
- You handle resource state transitions (image layouts) yourself.

The cost is that Vulkan is extremely verbose. A "hello triangle" in Vulkan requires
roughly 800 lines of code that OpenGL does in 80. But in exchange you get predictable
performance, correct multi-threading, and zero hidden cost.

This is exactly what SleakEngine does. `VulkanRenderer.cpp` is ~5,900 lines and manages
every one of these objects explicitly.

---

## 2. The Vulkan Mental Model

Think of Vulkan as a pipeline with three phases:

```
Setup Phase (happens once at startup)
  VkInstance → VkPhysicalDevice → VkDevice
  VkSurface → VkSwapchain → VkImageViews
  VkRenderPass → VkPipeline
  VkDescriptorSetLayout → VkDescriptorPool → VkDescriptorSet
  VkCommandPool → VkCommandBuffer (allocate)
  VkSemaphore, VkFence (sync primitives)

Per-Frame Phase (happens every 16ms)
  Wait for fence → Acquire swapchain image
  vkBeginCommandBuffer
    vkCmdBeginRenderPass
    vkCmdBindPipeline
    vkCmdBindDescriptorSets
    vkCmdBindVertexBuffers / BindIndexBuffer
    vkCmdPushConstants
    vkCmdDraw / vkCmdDrawIndexed
    vkCmdEndRenderPass
  vkEndCommandBuffer
  vkQueueSubmit → vkQueuePresentKHR

Teardown Phase (happens once at shutdown)
  vkDeviceWaitIdle
  Destroy everything in reverse creation order
```

This is the shape of `VulkanRenderer::Initialize()`, `BeginRender()`/`EndRender()`,
and `Cleanup()` in SleakEngine. Keep this three-phase model in your head at all times.

---

## 3. Instance — The Root Object

A `VkInstance` is the root of the entire Vulkan context. It loads the Vulkan library,
discovers available GPUs, and activates any validation or extension layers you want.

```cpp
VkApplicationInfo appInfo{};
appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
appInfo.pApplicationName   = "SleakCraft";
appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
appInfo.pEngineName        = "SleakEngine";
appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
appInfo.apiVersion         = VK_API_VERSION_1_0;

VkInstanceCreateInfo createInfo{};
createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
createInfo.pApplicationInfo = &appInfo;

VkInstance instance;
vkCreateInstance(&createInfo, nullptr, &instance);
```

The `sType` field is present on every Vulkan struct. It tells the driver which struct
type it is seeing. You must always set it correctly or the driver will crash.

### Extensions

To display anything on screen Vulkan needs platform extensions. SDL3 handles this for
us. In SleakEngine, `SDL_Vulkan_LoadLibrary` and `SDL_Vulkan_CreateSurface` use the
`VK_KHR_surface` and platform surface extensions automatically.

For the debug messenger (covered next) you also need `VK_EXT_debug_utils`. This
extension is only available when the Vulkan validation SDK is installed.

---

## 4. Validation Layers and the Debug Messenger

Out of the box, Vulkan does almost no error checking. If you pass the wrong struct,
your GPU will silently produce garbage or crash with no message.

**Validation layers** are optional modules that sit between your app and the driver.
They intercept every Vulkan call and verify it is correct, then forward it to the real
driver. The most important one is `VK_LAYER_KHRONOS_validation`.

```cpp
const char* validationLayer = "VK_LAYER_KHRONOS_validation";
// Add to createInfo.ppEnabledLayerNames when creating the instance
```

You also set up a **debug messenger** that routes validation errors to your logger:

```cpp
// From VulkanRenderer.cpp:
void VulkanRenderer::SetupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    PopulateDebugMessengerCreateInfo(createInfo);

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func) {
        func(instance, &createInfo, nullptr, &debugMessenger);
        // Also load the destroy function pointer for cleanup
        vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    }
}
```

Notice that `vkCreateDebugUtilsMessengerEXT` is not loaded automatically — it is an
extension function. You must look it up with `vkGetInstanceProcAddr`. SleakEngine does
exactly this. The callback you provide prints errors like:

```
[VULKAN VALIDATION] ERROR: vkBindImageMemory: image was not created by this device
```

Always enable validation layers during development. Turn them off in release builds
where their overhead (~5-15%) is unacceptable.

---

## 5. Surface — Connecting Vulkan to the Window

Vulkan is display-server agnostic. It has no built-in concept of a window. Instead,
you create a `VkSurfaceKHR` that represents the window surface Vulkan will render into.

On Linux (Wayland), this requires `VK_KHR_wayland_surface`. On Windows it requires
`VK_KHR_win32_surface`. SDL3 hides this platform complexity:

```cpp
// From VulkanRenderer.cpp:
bool VulkanRenderer::CreateSurface() {
    SDL_Vulkan_LoadLibrary(NULL);
    bool result = SDL_Vulkan_CreateSurface(
        sdlWindow->GetSDLWindow(), instance, nullptr, &surface);

    if (!result || surface == VK_NULL_HANDLE) {
        SLEAK_ERROR("Caught an SDL error! {}", SDL_GetError());
        return false;
    }
    return true;
}
```

The surface must exist before you choose a physical device, because device selection
must verify the device can actually present to your surface.

---

## 6. Physical Device — Picking the GPU

A `VkPhysicalDevice` represents an actual GPU in the system. You do not create it —
you enumerate them and pick the best one.

```cpp
uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
std::vector<VkPhysicalDevice> gpus(deviceCount);
vkEnumeratePhysicalDevices(instance, &deviceCount, gpus.data());
```

In SleakEngine, the renderer stores the list: `std::vector<VkPhysicalDevice> GPUs`.

A physical device has:

- **Properties** — name, vendor ID, device type (discrete vs integrated), limits
- **Memory properties** — how many heaps, which are device-local (VRAM), which are
  host-visible (system RAM)
- **Features** — does it support geometry shaders? anisotropic filtering? tessellation?
- **Queue families** — groups of queues that support different operations

```cpp
VkPhysicalDeviceProperties props;
vkGetPhysicalDeviceProperties(physicalDevice, &props);
// props.deviceName → "NVIDIA GeForce RTX 4090"
// props.limits.framebufferColorSampleCounts → which MSAA modes are available

VkPhysicalDeviceMemoryProperties memProps;
vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
// memProps.memoryHeaps[i].size → VRAM or system RAM size
// memProps.memoryTypes[i].propertyFlags → VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT etc
```

SleakEngine uses MSAA information from the device during setup:

```cpp
// From VulkanRenderer.cpp line ~1435:
VkSampleCountFlags counts =
    deviceProperties.limits.framebufferColorSampleCounts
  & deviceProperties.limits.framebufferDepthSampleCounts;
m_maxMsaaSampleCount = 1;
if (counts & VK_SAMPLE_COUNT_8_BIT) m_maxMsaaSampleCount = 8;
```

---

## 7. Logical Device and Queue Families

The `VkPhysicalDevice` is just a description of the hardware. To actually use it you
create a `VkDevice` — the logical device. You also specify which queue families you
want to use.

### Queue Families

A GPU is organized into **queue families**. Each family supports certain operations:

- **Graphics** — rendering, rasterization, vertex/fragment shaders
- **Compute** — compute shaders, GPGPU
- **Transfer** — copying memory between buffers and images
- **Present** — presenting frames to a display surface

To find which families exist on your device:

```cpp
uint32_t familyCount = 0;
vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
std::vector<VkQueueFamilyProperties> families(familyCount);
vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

for (uint32_t i = 0; i < familyCount; ++i) {
    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        QueueIDs.GraphicsIndex = i;
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        QueueIDs.ComputeIndex = i;
    if (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
        QueueIDs.TransferIndex = i;

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
    if (presentSupport)
        QueueIDs.PresentIndex = i;
}
```

On many desktop GPUs, graphics and present are the same family index. On some hardware
they differ. SleakEngine handles both cases (see the `VK_SHARING_MODE_CONCURRENT` vs
`VK_SHARING_MODE_EXCLUSIVE` in `CreateSwapChain`).

If compute or transfer are not found as separate families, they fall back to the
graphics family:

```cpp
// From VulkanRenderer.cpp line ~1413:
if (QueueIDs.ComputeIndex == UINT32_MAX)
    QueueIDs.ComputeIndex = QueueIDs.GraphicsIndex;
if (QueueIDs.TransferIndex == UINT32_MAX)
    QueueIDs.TransferIndex = QueueIDs.GraphicsIndex;
```

### Creating the Logical Device

You tell Vulkan which queue families you want and at what priority (0.0–1.0):

```cpp
// From VulkanRenderer.cpp line ~1447:
std::vector<const char*> requiredExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

VkDeviceCreateInfo deviceInfo{};
deviceInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
deviceInfo.pQueueCreateInfos       = queueCreateInfos.data();
deviceInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
deviceInfo.ppEnabledExtensionNames = requiredExtensions.data();
deviceInfo.enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size());
deviceInfo.pEnabledFeatures        = &features;

vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
```

Notice `VK_KHR_SWAPCHAIN_EXTENSION_NAME` — swapchain support is an extension, not a
core feature. Old or embedded Vulkan devices might not support it.

After creating the device you retrieve the actual queue handles:

```cpp
vkGetDeviceQueue(device, QueueIDs.GraphicsIndex, 0, &graphicsQueue);
vkGetDeviceQueue(device, QueueIDs.PresentIndex,  0, &presentQueue);
```

The `VkQueue` is what you submit command buffers to later.

---

## 8. Swapchain — Image Presentation

A swapchain is a circular buffer of images managed by the OS window system. Your
renderer draws into one image while the display reads another. When you are done, you
"present" your image to the screen and get a new one.

### Querying Swapchain Support

Before creating a swapchain, you must query what the surface supports:

```cpp
struct SwapchainDetails {
    VkSurfaceCapabilitiesKHR          caps;         // min/max images, sizes
    std::vector<VkSurfaceFormatKHR>   formats;      // color space + pixel format
    std::vector<VkPresentModeKHR>     presentModes; // FIFO, Mailbox, Immediate
};
```

- **Capabilities** tell you how many images the swapchain can have, and whether the
  surface has a fixed size or lets you choose.
- **Formats** tell you which pixel formats are supported (usually `VK_FORMAT_B8G8R8A8_SRGB`).
- **Present modes** tell you how frame pacing works.

### Present Modes

| Mode | Behavior |
|------|----------|
| `VK_PRESENT_MODE_FIFO_KHR` | VSync on. Always available. Queue frames, consume one per vblank. |
| `VK_PRESENT_MODE_MAILBOX_KHR` | Triple buffering. No tearing, low latency. Not always available. |
| `VK_PRESENT_MODE_IMMEDIATE_KHR` | No VSync. Tearing possible. Lowest latency. |

SleakEngine supports runtime VSync toggling via `ApplyVSyncChange()`, which recreates
the entire swapchain with the new present mode.

### Creating the Swapchain

```cpp
// From VulkanRenderer.cpp line ~1551:
VkSwapchainCreateInfoKHR info{};
info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
info.surface          = surface;
info.minImageCount    = imageCount;    // typically 2 or 3
info.imageFormat      = format.format; // e.g. VK_FORMAT_B8G8R8A8_SRGB
info.imageColorSpace  = format.colorSpace;
info.imageExtent      = extent;        // window pixel dimensions
info.imageArrayLayers = 1;             // always 1 unless VR stereo
info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
info.presentMode      = mode;

// If graphics and present are different families, share images between them
if (QueueIDs.GraphicsIndex != QueueIDs.PresentIndex) {
    info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices   = indices;
} else {
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
}

info.preTransform   = details.caps.currentTransform; // rotation (e.g. on mobile)
info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
info.clipped        = VK_TRUE;   // clip pixels obscured by other windows
info.oldSwapchain   = VK_NULL_HANDLE;

vkCreateSwapchainKHR(device, &info, nullptr, &swapChain);
```

After creation you enumerate the images the swapchain owns:

```cpp
vkGetSwapchainImagesKHR(device, swapChain, &scImageCount, nullptr);
swapChainImages.resize(scImageCount);
vkGetSwapchainImagesKHR(device, swapChain, &scImageCount, swapChainImages.data());
```

These `VkImage` objects belong to the swapchain — you must not destroy them yourself.

---

## 9. Image Views — How to Read an Image

A `VkImage` is raw GPU memory in some opaque hardware layout. To actually use an image
in a pipeline — whether to render into it or sample it in a shader — you need a
`VkImageView`. The view tells Vulkan:

- Which image this is
- What type (2D, cubemap, array)
- Which mip levels and array layers to expose
- What format to interpret it as
- Which color channels map to which (component swizzle)

```cpp
// From VulkanRenderer.cpp line ~1862 (CreateImageViews):
VkImageViewCreateInfo info{};
info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
info.image    = swapChainImages[i];
info.viewType = VK_IMAGE_VIEW_TYPE_2D;
info.format   = scImageFormat;

info.components = {
    VK_COMPONENT_SWIZZLE_IDENTITY,  // R → R
    VK_COMPONENT_SWIZZLE_IDENTITY,  // G → G
    VK_COMPONENT_SWIZZLE_IDENTITY,  // B → B
    VK_COMPONENT_SWIZZLE_IDENTITY   // A → A
};

info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
info.subresourceRange.baseMipLevel   = 0;
info.subresourceRange.levelCount     = 1;
info.subresourceRange.baseArrayLayer = 0;
info.subresourceRange.layerCount     = 1;

vkCreateImageView(device, &info, nullptr, &swapChainImageViews[i]);
```

The `aspectMask` is `VK_IMAGE_ASPECT_COLOR_BIT` for color images and
`VK_IMAGE_ASPECT_DEPTH_BIT` for depth images. For a depth-stencil format you can have
both aspects.

---

## 10. Render Passes — The Rendering Contract

A render pass is one of Vulkan's most important concepts and also the one that confuses
beginners the most. It is a description of what attachments (render targets) will be
used during rendering, what state they are in before rendering starts, and what state
they should be in when rendering ends.

By knowing this up front, the GPU driver can:
- Allocate tile memory efficiently on mobile GPUs (TBDR architectures)
- Eliminate unnecessary image layout transitions
- Schedule memory transfers ahead of time

### Attachments

An attachment is an image that the GPU reads from or writes to during a render pass.
Typical attachments:

- **Color** — the output image (what you see on screen)
- **Depth/stencil** — the depth buffer
- **Resolve** — the destination for MSAA resolve (only when MSAA is enabled)

For each attachment you specify:

- `format` — pixel format
- `samples` — how many MSAA samples
- `loadOp` — what to do with existing content at pass start (`CLEAR`, `LOAD`, `DONT_CARE`)
- `storeOp` — what to do with content at pass end (`STORE`, `DONT_CARE`)
- `initialLayout` — image layout before the pass
- `finalLayout` — image layout the driver should transition to after the pass

```cpp
// From VulkanRenderer.cpp line ~2417 (CreateRenderPass):
VkAttachmentDescription colorAttachment{};
colorAttachment.format        = scImageFormat;
colorAttachment.samples       = m_msaaSamples;
colorAttachment.loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;   // clear to clearColor
colorAttachment.storeOp       = msaaEnabled
    ? VK_ATTACHMENT_STORE_OP_DONT_CARE   // MSAA: result goes into resolve attachment
    : VK_ATTACHMENT_STORE_OP_STORE;      // no MSAA: store directly to swapchain
colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;     // we will clear it anyway
colorAttachment.finalLayout    = msaaEnabled
    ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // ready to present
```

### Subpasses

A render pass contains one or more **subpasses**. Each subpass references a subset of
the attachments. Adjacent subpasses can share data on-chip without writing to main
memory (useful for deferred rendering on mobile).

For a simple forward renderer you need only one subpass:

```cpp
VkSubpassDescription subpass{};
subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.colorAttachmentCount    = 1;
subpass.pColorAttachments       = &colorRef;       // layout: COLOR_ATTACHMENT_OPTIMAL
subpass.pDepthStencilAttachment = &depthRef;       // layout: DEPTH_STENCIL_ATTACHMENT_OPTIMAL
subpass.pResolveAttachments     = msaaEnabled ? &resolveRef : nullptr;
```

### Subpass Dependencies

Subpass dependencies tell Vulkan which operations must complete before the next step
can begin. Without them you may get rendering artifacts from out-of-order writes:

```cpp
VkSubpassDependency dependency{};
dependency.srcSubpass    = VK_SUBPASS_EXTERNAL; // operations before the pass
dependency.dstSubpass    = 0;                   // our subpass
dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                         | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
dependency.srcAccessMask = 0;
dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                         | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
```

This says: "the color attachment output and early fragment tests stages of any work
outside the pass must complete before our subpass writes to those attachments."

---

## 11. Framebuffers — The Render Target

A `VkFramebuffer` is a collection of `VkImageView` objects that are used as the
attachments during a render pass. It ties the abstract render pass description to
concrete image views.

You create one framebuffer per swapchain image (since each image is a different render
target):

```cpp
VkImageView attachments[] = { swapChainImageViews[i], depthImageView };
// With MSAA:
VkImageView attachments[] = { m_msaaColorImageView, depthImageView, swapChainImageViews[i] };

VkFramebufferCreateInfo fbInfo{};
fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
fbInfo.renderPass      = renderPass;   // must match the render pass exactly
fbInfo.attachmentCount = attachmentCount;
fbInfo.pAttachments    = attachments;
fbInfo.width           = scExtent.width;
fbInfo.height          = scExtent.height;
fbInfo.layers          = 1;

vkCreateFramebuffer(device, &fbInfo, nullptr, &swapChainFramebuffers[i]);
```

The attachment order in `pAttachments` must exactly match the attachment order declared
in the `VkRenderPassCreateInfo` you used.

---

## 12. Shaders and SPIR-V

OpenGL compiled GLSL shaders at runtime in the driver. Vulkan does not. Vulkan uses
**SPIR-V**, an intermediate binary format. You compile your GLSL (or HLSL) to SPIR-V
ahead of time using `glslangValidator` or the DirectXShaderCompiler, and ship the `.spv`
files with your game.

SleakEngine compiles shaders with `scripts/compile_shaders.sh`:

```bash
glslangValidator -V flat_shader.vert -o flat_shader.vert.spv
glslangValidator -V flat_shader.frag -o flat_shader.frag.spv
```

### Loading a Shader

A SPIR-V binary is loaded and wrapped in a `VkShaderModule`:

```cpp
// From VulkanShader.cpp (simplified):
std::vector<char> VulkanShader::ReadFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    size_t size = (size_t)file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

VkShaderModule VulkanShader::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    vkCreateShaderModule(device, &info, nullptr, &module);
    return module;
}
```

The shader module is then described as a pipeline stage:

```cpp
VkPipelineShaderStageCreateInfo vertInfo{};
vertInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
vertInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
vertInfo.module = vertShader;
vertInfo.pName  = "main";   // entry point function name in the SPIR-V
```

Shader modules are used only during pipeline creation. Once the pipeline is built, you
can (and should) destroy the shader modules. SleakEngine keeps them alive because it
recreates pipelines on MSAA changes, but you could free them right after pipeline
creation.

### GLSL in SleakCraft

The Vulkan GLSL shaders live at `Game/assets/shaders/`:
- `flat_shader.vert` / `flat_shader.frag` — standard mesh
- `water_shader.vert` / `water_shader.frag` — water/voxel transparent pass

Each has a corresponding `.spv` compiled binary that must be committed alongside the
source. Always run `scripts/compile_shaders.sh` after editing a Vulkan shader.

---

## 13. Descriptor Set Layouts — What Shaders Declare They Need

Shaders need access to resources: textures, uniform buffers, storage buffers. In
Vulkan, a **descriptor** is a handle to one resource binding. A **descriptor set
layout** declares what bindings a shader expects, without naming actual resources.

Think of it as the function signature for a shader's resource parameters.

```cpp
// From VulkanRenderer.cpp line ~2005 (CreateDescriptorSetLayout):

// Set 0, binding 0: a combined image sampler (texture + sampler in one)
VkDescriptorSetLayoutBinding samplerBinding{};
samplerBinding.binding         = 0;
samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
samplerBinding.descriptorCount = 1;
samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

VkDescriptorSetLayoutCreateInfo layoutInfo{};
layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
layoutInfo.bindingCount = 1;
layoutInfo.pBindings    = &samplerBinding;

vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
```

SleakEngine uses four descriptor set layouts bound to four descriptor set slots:

| Set | Binding | Type | Stage | Purpose |
|-----|---------|------|-------|---------|
| 0 | 0 | `COMBINED_IMAGE_SAMPLER` | Fragment | Current texture |
| 1 | 0 | `UNIFORM_BUFFER` | Vertex | Bone matrices (skeletal anim) |
| 2 | 0 | `UNIFORM_BUFFER` | Fragment | Light data + shadow VP |
| 3 | 0 | `COMBINED_IMAGE_SAMPLER` | Fragment | Shadow map depth texture |

In your GLSL shader these correspond to:

```glsl
layout(set = 0, binding = 0) uniform sampler2D texSampler;
layout(set = 1, binding = 0) uniform BoneData { mat4 bones[128]; };
layout(set = 2, binding = 0) uniform LightData { vec3 lightDir; ... };
layout(set = 3, binding = 0) uniform sampler2D shadowMap;
```

---

## 14. Descriptor Pools and Descriptor Sets — Binding Resources

A **descriptor pool** pre-allocates storage for a fixed number of descriptors. You
specify ahead of time how many descriptors of each type you will allocate:

```cpp
std::vector<VkDescriptorPoolSize> poolSizes = {
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },  // up to 1000 texture slots
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         50   },
};

VkDescriptorPoolCreateInfo poolInfo{};
poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
poolInfo.pPoolSizes    = poolSizes.data();
poolInfo.maxSets       = 1000;  // max total descriptor sets

vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
```

From this pool you **allocate descriptor sets**. Each set is a concrete instance of a
layout — it holds the actual resource references:

```cpp
// From VulkanRenderer.cpp line ~2105:
std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);

VkDescriptorSetAllocateInfo allocInfo{};
allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
allocInfo.descriptorPool     = descriptorPool;
allocInfo.descriptorSetCount = imageCount;
allocInfo.pSetLayouts        = layouts.data();

descriptorSets.resize(imageCount);
vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
```

Why one per swapchain image? Because multiple frames can be in flight simultaneously
(see section 23). If frame 0 is using descriptor set 0 while frame 1 is using
descriptor set 1, neither corrupts the other.

### Writing to a Descriptor Set

Allocating a descriptor set does not point it at any resource yet. You do that with
`vkUpdateDescriptorSets`:

```cpp
// From VulkanRenderer.cpp line ~2190 (WriteTextureDescriptors):
VkDescriptorImageInfo imageInfo{};
imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
imageInfo.imageView   = texture->GetImageView();
imageInfo.sampler     = texture->GetSampler();

VkWriteDescriptorSet write{};
write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
write.dstSet          = sets[i];
write.dstBinding      = 0;           // binding 0 in the layout
write.dstArrayElement = 0;           // first element (if binding is an array)
write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
write.descriptorCount = 1;
write.pImageInfo      = &imageInfo;

vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
```

This is how you change which texture is visible to the shader. SleakEngine calls
`WriteTextureDescriptors` every time a new texture is created, allocating fresh
descriptor sets for it. During rendering, `BindTexture` switches which set is bound.

---

## 15. Pipeline Layout — The Glue Between Pipeline and Descriptors

Before you create the graphics pipeline you need a `VkPipelineLayout`. It describes
the interface between your C++ code and the shader:

- Which descriptor set layouts are used
- Which push constant ranges are available

```cpp
// From VulkanRenderer.cpp line ~2358:

VkPushConstantRange pushConstantRange{};
pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
pushConstantRange.offset     = 0;
pushConstantRange.size       = 128;  // 2x mat4 = WVP matrix + World matrix

std::array<VkDescriptorSetLayout, 4> setLayouts = {
    descriptorSetLayout,             // set 0: texture
    boneDescriptorSetLayout,         // set 1: bone UBO
    m_lightUBODescriptorSetLayout,   // set 2: light UBO
    m_shadowSamplerDescriptorSetLayout  // set 3: shadow map
};

VkPipelineLayoutCreateInfo layoutInfo{};
layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
layoutInfo.setLayoutCount         = static_cast<uint32_t>(setLayouts.size());
layoutInfo.pSetLayouts            = setLayouts.data();
layoutInfo.pushConstantRangeCount = 1;
layoutInfo.pPushConstantRanges    = &pushConstantRange;

vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLay);
```

The pipeline layout is shared across all pipelines in SleakEngine. Skybox, voxel, and
skinned pipelines all use the same `pipelineLay`, which is why they can share the same
descriptor sets.

---

## 16. Graphics Pipeline — The Giant Baked Object

The graphics pipeline is the biggest object in Vulkan. In OpenGL, "state" (blend mode,
depth test, cull mode, shader) could be changed at any point. Vulkan forces you to
declare all of this up front and bake it into one immutable object.

This is the key performance insight: the driver can compile the pipeline to optimal
machine code at creation time. No more "shader stutter" from lazy compilation at draw
time.

A pipeline is composed of many state structs:

### Shader Stages

```cpp
VkPipelineShaderStageCreateInfo shaderStages[] = {
    simpleShader->GetVertexInfo(),   // VK_SHADER_STAGE_VERTEX_BIT
    simpleShader->GetFragInfo()      // VK_SHADER_STAGE_FRAGMENT_BIT
};
```

### Vertex Input

Describes the per-vertex data layout. Must match your vertex buffer structure exactly:

```cpp
// From VulkanRenderer.cpp line ~2236 (Sleak::Vertex is 64 bytes):
VkVertexInputBindingDescription binding{};
binding.binding   = 0;
binding.stride    = sizeof(Vertex);
binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;  // advance per vertex (not per instance)

// Per-attribute: location, format, offset into the vertex struct
attributeDescs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, px) };    // position
attributeDescs[1] = { 0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx) };    // normal
attributeDescs[2] = { 0, 2, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tx) }; // tangent
attributeDescs[3] = { 0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, r) };  // color
attributeDescs[4] = { 0, 4, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, u) };        // UV
attributeDescs[5] = { 0, 5, VK_FORMAT_R32G32B32A32_SINT, offsetof(Vertex, boneIDs) };
attributeDescs[6] = { 0, 6, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, boneWeights) };
```

### Input Assembly

What primitive topology should the GPU use?

```cpp
VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
inputAssembly.primitiveRestartEnable = VK_FALSE;
```

### Viewport and Scissor (Dynamic State)

SleakEngine uses dynamic viewport/scissor so it can change them per-frame without
recreating the pipeline (e.g. for shadow pass at 4096x4096 vs main pass at window size):

```cpp
std::vector<VkDynamicState> dynamicStates = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR
};
VkPipelineDynamicStateCreateInfo dynamicState{};
dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
dynamicState.pDynamicStates    = dynamicStates.data();

// You still declare that you have one viewport and one scissor:
VkPipelineViewportStateCreateInfo viewportState{};
viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
viewportState.viewportCount = 1;
viewportState.scissorCount  = 1;
// pViewports and pScissors are null because they're dynamic
```

### Rasterizer

Controls backface culling, polygon mode, depth bias:

```cpp
VkPipelineRasterizationStateCreateInfo rasterizer{};
rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
rasterizer.depthClampEnable        = VK_FALSE;
rasterizer.rasterizerDiscardEnable = VK_FALSE;
rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
rasterizer.lineWidth               = 1.0f;
rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;       // cull back faces
rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE; // CCW = front
rasterizer.depthBiasEnable         = VK_FALSE;
```

Note: `VK_FRONT_FACE_COUNTER_CLOCKWISE` means triangles with vertices going
counter-clockwise in screen space are front-facing. This is the mathematical convention.
OpenGL uses the same convention; DX11/DX12 uses clockwise by default.

### Depth/Stencil

```cpp
VkPipelineDepthStencilStateCreateInfo depthStencil{};
depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
depthStencil.depthTestEnable       = VK_TRUE;
depthStencil.depthWriteEnable      = VK_TRUE;
depthStencil.depthCompareOp        = VK_COMPARE_OP_LESS;  // pass if new depth < stored depth
depthStencil.depthBoundsTestEnable = VK_FALSE;
depthStencil.stencilTestEnable     = VK_FALSE;
```

### Color Blending

Per-attachment blend configuration. SleakEngine uses standard alpha blending:

```cpp
VkPipelineColorBlendAttachmentState blend{};
blend.blendEnable         = VK_TRUE;
blend.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
blend.colorBlendOp        = VK_BLEND_OP_ADD;
blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
blend.alphaBlendOp        = VK_BLEND_OP_ADD;
```

The formula this implements is:
`finalColor = srcAlpha * srcColor + (1 - srcAlpha) * dstColor`

### Creating the Pipeline

After building all the state structs, you create the pipeline in one call:

```cpp
// From VulkanRenderer.cpp line ~2386:
VkGraphicsPipelineCreateInfo pipelineInfo{};
pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
pipelineInfo.stageCount          = 2;
pipelineInfo.pStages             = shaderStages;
pipelineInfo.pVertexInputState   = &vertexInputInfo;
pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
pipelineInfo.pViewportState      = &viewportInfo;
pipelineInfo.pRasterizationState = &rasterizer;
pipelineInfo.pMultisampleState   = &msaa;
pipelineInfo.pDepthStencilState  = &depthStencil;
pipelineInfo.pColorBlendState    = &colorBlendInfo;
pipelineInfo.pDynamicState       = &dynamicState;
pipelineInfo.layout              = pipelineLay;
pipelineInfo.renderPass          = renderPass;
pipelineInfo.subpass             = 0;
pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
```

The second argument is a `VkPipelineCache` — if provided, the driver can reuse
compilation work across pipeline creations (very useful for faster startup). SleakEngine
passes `VK_NULL_HANDLE` (no cache) for simplicity.

---

## 17. Command Pools and Command Buffers

Commands in Vulkan are not sent immediately to the GPU. They are first recorded into a
command buffer, then submitted as a batch. This allows multi-threading, replaying, and
driver-side optimization.

### Command Pool

A command pool allocates memory for command buffers. It is tied to a specific queue
family index:

```cpp
// From VulkanRenderer.cpp:
VkCommandPoolCreateInfo poolInfo{};
poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.queueFamilyIndex = QueueIDs.GraphicsIndex;
poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

vkCreateCommandPool(device, &poolInfo, nullptr, &commands);
```

`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` allows individual command buffers
from this pool to be reset independently. Without this flag you can only reset the
entire pool at once.

### Command Buffer Allocation

```cpp
// From VulkanRenderer.cpp line ~927:
commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);  // 3

VkCommandBufferAllocateInfo allocInfo{};
allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
allocInfo.commandPool        = commands;
allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
```

`VK_COMMAND_BUFFER_LEVEL_PRIMARY` means this buffer can be submitted directly to a
queue. Secondary command buffers (`VK_COMMAND_BUFFER_LEVEL_SECONDARY`) can only be
called from primary buffers — useful for parallel recording across threads.

### Recording Commands

```cpp
// Reset and begin
vkResetCommandBuffer(command, 0);

VkCommandBufferBeginInfo beginInfo{};
beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
beginInfo.flags = 0;   // or VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT for one-shot

vkBeginCommandBuffer(command, &beginInfo);

// ... record commands (vkCmd*) ...

vkEndCommandBuffer(command);
```

Common recording commands used in SleakEngine:

```cpp
vkCmdBeginRenderPass(command, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
vkCmdSetViewport(command, 0, 1, &viewport);
vkCmdSetScissor(command, 0, 1, &scissor);
vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipelineLay, 0, 1, &descriptorSets[i], 0, nullptr);
vkCmdBindVertexBuffers(command, 0, 1, buffers, offsets);
vkCmdBindIndexBuffer(command, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
vkCmdPushConstants(command, pipelineLay, VK_SHADER_STAGE_VERTEX_BIT, 0, 128, data);
vkCmdDrawIndexed(command, indexCount, 1, 0, 0, 0);
vkCmdEndRenderPass(command);
```

### Submitting

After recording, you submit to the queue for execution:

```cpp
VkSubmitInfo submitInfo{};
submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
submitInfo.waitSemaphoreCount   = 1;
submitInfo.pWaitSemaphores      = &imageAvailableSemaphores[currentFrame];
submitInfo.pWaitDstStageMask    = &waitStage;
submitInfo.commandBufferCount   = 1;
submitInfo.pCommandBuffers      = &command;
submitInfo.signalSemaphoreCount = 1;
submitInfo.pSignalSemaphores    = &renderFinishedSemaphores[currentFrame];

vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
```

The semaphores and fences here are the heart of multi-frame synchronization, explained
in section 22 and 23.

---

## 18. GPU Memory — Understanding Memory Types

Vulkan exposes the GPU's memory system directly. There is no automatic memory
management. You must understand the hardware memory hierarchy.

Every GPU has several **memory heaps** (physical banks) and several **memory types**
(configurations that can access those heaps). The two key properties:

- `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` — VRAM. Fast for the GPU to read, inaccessible
  to the CPU.
- `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` — Can be mapped to CPU address space with
  `vkMapMemory`. Slower for the GPU.
- `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` — CPU writes are immediately visible to GPU
  without explicit flushing.

Finding the right memory type for a buffer:

```cpp
// From VulkanRenderer.cpp line ~1988:
uint32_t VulkanRenderer::FindMemoryType(uint32_t typeFilter,
                                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&                        // GPU says this type is valid
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0; // error
}
```

`typeFilter` is a bitmask from `VkMemoryRequirements.memoryTypeBits` — the GPU tells
you which memory types are compatible with the buffer/image you created.

---

## 19. Buffers — Vertex, Index, Uniform, and Staging

### Creating a Buffer

Every buffer in Vulkan requires two steps: creating the `VkBuffer` object, and
allocating backing `VkDeviceMemory`.

```cpp
VkBufferCreateInfo bufferInfo{};
bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bufferInfo.size        = size;
bufferInfo.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                       | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

VkBuffer buffer;
vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

VkMemoryRequirements req;
vkGetBufferMemoryRequirements(device, buffer, &req);

VkMemoryAllocateInfo allocInfo{};
allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.allocationSize  = req.size;
allocInfo.memoryTypeIndex = FindMemoryType(req.memoryTypeBits,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

VkDeviceMemory memory;
vkAllocateMemory(device, &allocInfo, nullptr, &memory);
vkBindBufferMemory(device, buffer, memory, 0);  // bind at offset 0
```

### The Staging Buffer Pattern

You cannot directly write to a device-local buffer from the CPU (it is in VRAM). The
standard pattern is to use a **staging buffer**:

1. Create a host-visible buffer (staging)
2. Copy your data from CPU to the staging buffer via `vkMapMemory`
3. Record a `vkCmdCopyBuffer` command to copy from staging → device-local
4. Submit and wait, then destroy the staging buffer

```cpp
// From VulkanBuffer.cpp line ~50 (Initialize for a vertex buffer):

// Step 1: staging buffer in host-visible memory
CreateBuffer(size,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    m_stagingBuffer, m_stagingMemory);

// Step 2: map and write data
void* mapped;
vkMapMemory(m_device, m_stagingMemory, 0, size, 0, &mapped);
memcpy(mapped, data, size);
vkUnmapMemory(m_device, m_stagingMemory);

// Step 3: device-local buffer (VRAM)
CreateBuffer(size,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    m_buffer, m_memory);

// Step 4: submit copy command
CopyBuffer(m_stagingBuffer, m_buffer, size);

// Step 5: destroy staging buffer
vkDestroyBuffer(m_device, m_stagingBuffer, nullptr);
vkFreeMemory(m_device, m_stagingMemory, nullptr);
```

SleakEngine batches many staging copies together using `SetBatchingEnabled(true)` to
avoid the overhead of submitting one command buffer per buffer upload. The batch is
flushed asynchronously at the start of each frame.

### Uniform Buffers vs Push Constants

For small per-draw-call data (transformation matrices), SleakEngine uses **push
constants** rather than uniform buffers. Push constants are a small block of data
(guaranteed minimum 128 bytes) embedded directly in the command buffer:

```cpp
// From VulkanRenderer.cpp line ~709:
vkCmdPushConstants(command, pipelineLay,
    VK_SHADER_STAGE_VERTEX_BIT, 0, 128, data);
```

This is faster than a uniform buffer because no buffer binding is needed — the data
flows directly from the CPU command stream into the shader. SleakEngine stores two
`mat4` (WVP + World matrix) = exactly 128 bytes.

For larger data that changes infrequently (bone matrices, light data) uniform buffers
in `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` memory
are used so the CPU can update them without staging.

---

## 20. Images and Samplers — Textures in Vulkan

### Creating an Image

Similar to buffers but with more parameters:

```cpp
// From VulkanRenderer.cpp line ~1893 (CreateDepthResources):
VkImageCreateInfo imageInfo{};
imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
imageInfo.imageType     = VK_IMAGE_TYPE_2D;
imageInfo.extent.width  = width;
imageInfo.extent.height = height;
imageInfo.extent.depth  = 1;
imageInfo.mipLevels     = 1;
imageInfo.arrayLayers   = 1;
imageInfo.format        = format;
imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;  // GPU-optimal memory layout
imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imageInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT;
imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

vkCreateImage(device, &imageInfo, nullptr, &image);
// Then allocate and bind memory exactly as with buffers
```

`VK_IMAGE_TILING_OPTIMAL` means the driver stores the image in a hardware-specific
swizzled layout for optimal cache performance. You cannot map this memory directly —
you always copy data via a staging buffer and `vkCmdCopyBufferToImage`.

### Samplers

A `VkSampler` configures how a texture is read in a shader — filtering, wrapping,
anisotropy:

```cpp
VkSamplerCreateInfo samplerInfo{};
samplerInfo.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
samplerInfo.magFilter        = VK_FILTER_LINEAR;  // bilinear when magnifying
samplerInfo.minFilter        = VK_FILTER_LINEAR;  // bilinear when minifying
samplerInfo.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;  // trilinear
samplerInfo.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.anisotropyEnable = VK_TRUE;
samplerInfo.maxAnisotropy    = 16.0f;
samplerInfo.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
samplerInfo.unnormalizedCoordinates = VK_FALSE;
samplerInfo.compareEnable    = VK_FALSE;
samplerInfo.mipLodBias       = 0.0f;
samplerInfo.minLod           = 0.0f;
samplerInfo.maxLod           = VK_LOD_CLAMP_NONE;

VkSampler sampler;
vkCreateSampler(device, &samplerInfo, nullptr, &sampler);
```

A `COMBINED_IMAGE_SAMPLER` descriptor bundles an image view and sampler together, which
is the most common way to bind textures in Vulkan and what SleakEngine uses.

### Loading a Texture

The flow in `VulkanTexture::LoadFromFile`:

1. Load pixels from disk (stb_image or similar)
2. Create a staging buffer, copy pixels into it
3. Create a `VK_IMAGE_TILING_OPTIMAL` image
4. Transition image layout from `UNDEFINED` → `TRANSFER_DST_OPTIMAL`
5. `vkCmdCopyBufferToImage` from staging to the image
6. Transition image layout from `TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`
7. Create an image view and sampler
8. Write descriptor sets

---

## 21. Image Layout Transitions

This is one of the most confusing aspects of Vulkan for beginners.

A `VkImage` is always in a specific **layout** — a memory organization that is optimal
for a particular use. The GPU needs to know the layout to access the image correctly.
Layouts include:

| Layout | Meaning |
|--------|---------|
| `VK_IMAGE_LAYOUT_UNDEFINED` | Unknown / don't care about existing content |
| `VK_IMAGE_LAYOUT_GENERAL` | Valid for any use, but not optimal for any |
| `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` | Being written as a color render target |
| `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` | Being used as depth/stencil |
| `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` | Being sampled in a shader |
| `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` | Source of a copy operation |
| `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` | Destination of a copy operation |
| `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` | Ready to be presented to the screen |

You transition between layouts using **pipeline barriers**:

```cpp
VkImageMemoryBarrier barrier{};
barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;          // we don't care about old content
barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; // about to copy into it
barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.image               = image;
barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
barrier.subresourceRange.baseMipLevel   = 0;
barrier.subresourceRange.levelCount     = 1;
barrier.subresourceRange.baseArrayLayer = 0;
barrier.subresourceRange.layerCount     = 1;
barrier.srcAccessMask = 0;                         // no prior accesses to synchronize
barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; // copy will write to it

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,   // wait for: nothing (top of pipe)
    VK_PIPELINE_STAGE_TRANSFER_BIT,      // before: the transfer stage
    0, 0, nullptr, 0, nullptr,
    1, &barrier
);
```

The render pass `finalLayout` field also performs an automatic layout transition at pass
end. In `CreateRenderPass`, `colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`
means Vulkan automatically transitions the swapchain image to present layout when the
pass ends — you do not need a manual barrier for this.

---

## 22. Synchronization — Fences, Semaphores, and Pipeline Barriers

Vulkan has three synchronization primitives, each solving a different problem.

### Fences — CPU/GPU synchronization

A `VkFence` lets the CPU wait for a GPU operation to complete:

```cpp
VkFenceCreateInfo fenceInfo{};
fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // pre-signaled: first frame doesn't wait

VkFence fence;
vkCreateFence(device, &fenceInfo, nullptr, &fence);

// CPU waits for GPU to finish frame N:
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

// Reset after waiting so it can be reused:
vkResetFences(device, 1, &fence);
```

In SleakEngine, `inFlightFences[currentFrame]` is waited on at the start of each frame
to ensure the previous use of that frame slot has finished:

```cpp
// From VulkanRenderer.cpp line ~173:
vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
```

### Semaphores — GPU/GPU synchronization

A `VkSemaphore` synchronizes work between GPU operations without the CPU being involved.
SleakEngine uses two semaphores per frame:

- `imageAvailableSemaphores[i]` — signaled by `vkAcquireNextImageKHR` when a swapchain
  image is ready to render into
- `renderFinishedSemaphores[i]` — signaled by `vkQueueSubmit` when rendering is done,
  waited on by `vkQueuePresentKHR` before presenting

```cpp
VkSemaphoreCreateInfo semaphoreInfo{};
semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
```

### Pipeline Barriers — GPU-internal synchronization

`vkCmdPipelineBarrier` (seen in section 21) synchronizes operations within the GPU
command stream. It tells the GPU "the operations I listed in `srcStageMask` must
complete before the operations in `dstStageMask` begin, and the memory accesses in
`srcAccessMask` must be flushed to caches before `dstAccessMask` reads them."

This is needed whenever:
- You write a buffer in one pass and read it in a later pass
- You change an image layout
- You want to ensure a previous compute pass is done before a following render pass

---

## 23. Multi-Frame-in-Flight — Overlapping CPU and GPU Work

Without any special handling, your frame loop would be:

```
CPU: build commands → submit → wait for GPU to finish → build next frame
GPU:                            render                  → idle
```

The GPU sits idle while the CPU builds the next frame. The CPU sits idle while the GPU
renders. Throughput is about 50% of possible.

The solution is to have multiple **frames-in-flight**: while the GPU is rendering frame
N, the CPU is building frame N+1.

SleakEngine uses 3 frames in flight (`MAX_FRAMES_IN_FLIGHT = 3`). This means:

- 3 command buffers (one per frame slot)
- 3 fences (one per frame slot)
- 3 semaphore pairs (one per frame slot)

```cpp
static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
std::vector<VkCommandBuffer> commandBuffers;   // size 3
std::vector<VkSemaphore>     imageAvailableSemaphores; // size 3
std::vector<VkSemaphore>     renderFinishedSemaphores; // size 3
std::vector<VkFence>         inFlightFences;           // size 3
uint32_t                     currentFrame = 0;
```

The frame loop becomes:

```cpp
// Wait for the PREVIOUS use of this frame slot to finish
vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
vkResetFences(device, 1, &inFlightFences[currentFrame]);

// Get a swapchain image to render into
vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
    imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

// Record commands into commandBuffers[currentFrame]
// ...

// Submit: wait for image to be available, signal when done
VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
VkSubmitInfo submitInfo{};
submitInfo.waitSemaphoreCount   = 1;
submitInfo.pWaitSemaphores      = &imageAvailableSemaphores[currentFrame];
submitInfo.pWaitDstStageMask    = &waitStage;
submitInfo.commandBufferCount   = 1;
submitInfo.pCommandBuffers      = &commandBuffers[currentFrame];
submitInfo.signalSemaphoreCount = 1;
submitInfo.pSignalSemaphores    = &renderFinishedSemaphores[currentFrame];

vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);

// Present: wait for rendering to finish
VkPresentInfoKHR presentInfo{};
presentInfo.waitSemaphoreCount = 1;
presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[currentFrame];
presentInfo.swapchainCount     = 1;
presentInfo.pSwapchains        = &swapChain;
presentInfo.pImageIndices      = &imageIndex;

vkQueuePresentKHR(presentQueue, &presentInfo);

currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
```

There is a subtle complication: the swapchain image index and the frame slot index are
independent. The swapchain might give you image 0, 1, 2 in any order, while your frame
slots cycle 0 → 1 → 2 → 0. You need to track which fence guards which swapchain image:

```cpp
// From VulkanRenderer.cpp line ~217:
if (imagesInFlight[CurrentFrameIndex] != VK_NULL_HANDLE) {
    vkWaitForFences(device, 1, &imagesInFlight[CurrentFrameIndex], VK_TRUE, UINT64_MAX);
}
imagesInFlight[CurrentFrameIndex] = inFlightFences[currentFrame];
```

---

## 24. Depth Testing

Depth testing prevents back faces of 3D geometry from appearing in front of nearer
surfaces. You need a depth image the same size as your render target.

SleakEngine creates the depth image in `CreateDepthResources`:

```cpp
// From VulkanRenderer.cpp line ~1891:
depthFormat = FindDepthFormat();  // tries D32, D32_S8, D24_S8 in order

VkImageCreateInfo imageInfo{};
imageInfo.format  = depthFormat;
imageInfo.usage   = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT;
imageInfo.samples = m_msaaSamples;
// ...
```

`FindDepthFormat` queries the physical device for which depth format is supported with
optimal tiling. It prefers `VK_FORMAT_D32_SFLOAT` (full 32-bit float precision).

The `VK_IMAGE_USAGE_SAMPLED_BIT` is needed because the deferred lighting pass reads the
depth buffer in its shader.

The depth image view uses `VK_IMAGE_ASPECT_DEPTH_BIT` as the aspect mask, not
`VK_IMAGE_ASPECT_COLOR_BIT`.

---

## 25. MSAA — Multisampled Anti-Aliasing

MSAA renders each pixel at multiple sample points and averages the results, reducing
jagged edges. In Vulkan you must:

1. Create a multisampled color image (`m_msaaColorImage`) at `N` samples
2. Create the depth image at `N` samples
3. Configure the render pass to use `N` samples for both color and depth
4. Add a resolve attachment (the swapchain image at 1 sample) to the render pass
5. Set `pResolveAttachments` in the subpass
6. Create all pipelines with `rasterizationSamples = N`

```cpp
// From VulkanRenderer.cpp line ~1726 (GetMaxUsableSampleCount):
VkPhysicalDeviceProperties props;
vkGetPhysicalDeviceProperties(physicalDevice, &props);
VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts
                          & props.limits.framebufferDepthSampleCounts;
if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
// ...
```

SleakEngine supports runtime MSAA changes via `ApplyMSAAChange()`. Because MSAA sample
count is baked into the pipeline and render pass, changing it requires destroying and
recreating all pipelines, the render pass, the depth image, the MSAA color image, and
the framebuffers. The order matters — SleakEngine always destroys in reverse creation
order, waits for the device to go idle first, then recreates.

In deferred rendering mode, MSAA is disabled for the GBuffer pass (GBuffer images use
`VK_SAMPLE_COUNT_1_BIT`). MSAA is incompatible with deferred rendering without
extensions like MSAA resolve passes.

---

## 26. Push Constants — Fast Per-Draw Data

Push constants are a small block of data (128 bytes guaranteed) that can be pushed
directly from the command buffer into the shader without a buffer allocation:

```cpp
// Declare in pipeline layout:
VkPushConstantRange range{};
range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
range.offset     = 0;
range.size       = 128;  // 2 * sizeof(mat4)

// Record in command buffer:
vkCmdPushConstants(command, pipelineLay, VK_SHADER_STAGE_VERTEX_BIT, 0, 128, data);
```

In the GLSL vertex shader this is:

```glsl
layout(push_constant) uniform PushConstants {
    mat4 wvp;    // World-View-Projection matrix
    mat4 world;  // World matrix (for normals, shadow)
};
```

SleakEngine uses push constants as the primary per-object data mechanism. The WVP
matrix (64 bytes) is multiplied by the object transform on the CPU and pushed each
draw call.

For the shadow pass there is a special override in `BindConstantBuffer`: it recalculates
the WVP as `LightVP * World` before pushing, so the same geometry pass logic works for
both main rendering and shadow map generation.

---

## 27. The Frame Loop — Putting Everything Together

Here is the complete per-frame sequence in SleakEngine, combining all concepts above:

```
BeginRender():
  1. vkWaitForFences(inFlightFences[currentFrame])     ← wait for last use of this slot
  2. Free staging buffers from that slot (GPU is done with them)
  3. Enable buffer upload batching
  4. vkAcquireNextImageKHR(imageAvailableSemaphores)   ← get next swapchain image
  5. Check imagesInFlight[imageIndex] fence
  6. vkResetFences(inFlightFences[currentFrame])
  7. vkResetCommandBuffer
  8. vkBeginCommandBuffer
  9. [Shadow pass] vkCmdBeginRenderPass(shadowRenderPass)
                   vkCmdBindPipeline(shadowPipeline)
                   ExecuteShadowPass()
                   vkCmdEndRenderPass
 10. [Deferred] vkCmdBeginRenderPass(gbufferRenderPass)
                vkCmdBindPipeline(gbufferPipeline)
  OR
     [Forward]  vkCmdBeginRenderPass(renderPass)
                vkCmdBindPipeline(pipeline)
 11. vkCmdSetViewport, vkCmdSetScissor
 12. vkCmdBindDescriptorSets
 13. Return to game — game records draw calls via RenderCommandQueue

[Game layer calls BindVertexBuffer / BindIndexBuffer / BindConstantBuffer / Draw]

EndRender():
 14. [Deferred lighting pass]
     vkCmdEndRenderPass (gbuffer)
     vkCmdBeginRenderPass(lightingPass)
     ExecuteDeferredLightingPass()
     vkCmdEndRenderPass (lighting)
 15. [Forward transparent pass]
     vkCmdBeginRenderPass(forwardRenderPass)
     ExecuteTransparentPass()
     vkCmdEndRenderPass
 16. ImGui render
     ImGui::Render() → ImGui_ImplVulkan_RenderDrawData(command)
 17. vkCmdEndRenderPass (main)
 18. Flush transfer semaphore
 19. vkEndCommandBuffer
 20. vkQueueSubmit(graphicsQueue, commandBuffer, inFlightFences[currentFrame])
 21. vkQueuePresentKHR(presentQueue)
 22. currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT
```

---

## 28. Swapchain Recreation — Handling Window Resize

When the window is resized, the swapchain becomes out of date. `vkAcquireNextImageKHR`
returns `VK_ERROR_OUT_OF_DATE_KHR` or `VK_SUBOPTIMAL_KHR`:

```cpp
// From VulkanRenderer.cpp line ~208:
result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
    imageAvailableSemaphores[m_semaphoreIndex], VK_NULL_HANDLE, &CurrentFrameIndex);

if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    RecreateSwapChain();
    return;
}
```

`RecreateSwapChain` must:
1. `vkDeviceWaitIdle` — wait for all in-flight work to finish
2. Destroy the old framebuffers, image views, depth image, MSAA image, swapchain
3. Create a new swapchain with the new window dimensions
4. Create new image views, depth resources, MSAA resources, framebuffers
5. Rebuild the GBuffer if deferred was enabled (GBuffer is sized to match swapchain)

Render passes and pipelines do NOT need to be recreated on resize — they are
independent of the window dimensions (the viewport/scissor are dynamic state).

---

## 29. Advanced: Deferred Rendering and the GBuffer

Standard forward rendering: for each object, for each light, compute lighting and
output final color. Cost = O(objects × lights).

Deferred rendering separates geometry and lighting into two passes:

- **Geometry pass (GBuffer)**: render all geometry, output to multiple render targets
  (G-Buffer = Geometry Buffer): albedo, normals, roughness, world position, depth.
- **Lighting pass**: fullscreen quad, read from GBuffer, compute all lights once per
  pixel. Cost = O(pixels × lights).

SleakEngine's GBuffer has 4 color attachments plus depth:

| Attachment | Format | Contents |
|------------|--------|----------|
| RT0 | `RGBA8_UNORM` | AlbedoRGB + AO |
| RT1 | `RGBA16_SFLOAT` | NormalXYZ + Roughness |
| RT2 | `RGBA8_UNORM` | MetalnessRGB + Emissive |
| RT3 | `RGBA32_SFLOAT` | World position XYZ (W unused) |
| Depth | `D32_SFLOAT` | Depth |

The GBuffer render pass has 5 attachments and one subpass that writes all 4 color
attachments. This is configured with `pColorAttachments` pointing to an array of 4
color refs:

```cpp
VkAttachmentReference colorRefs[GBUFFER_COUNT];
for (uint32_t i = 0; i < GBUFFER_COUNT; ++i) {
    colorRefs[i].attachment = i;
    colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
}

VkSubpassDescription subpass{};
subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.colorAttachmentCount = GBUFFER_COUNT;
subpass.pColorAttachments    = colorRefs;
subpass.pDepthStencilAttachment = &depthRef;
```

In the GBuffer fragment shader:

```glsl
layout(location = 0) out vec4 gAlbedoAO;
layout(location = 1) out vec4 gNormalRough;
layout(location = 2) out vec4 gMetalEmit;
layout(location = 3) out vec4 gWorldPos;
```

Each `layout(location = N)` output maps to `colorRefs[N]`.

The lighting pass binds all GBuffer images as `COMBINED_IMAGE_SAMPLER` descriptors and
does one fullscreen draw to compute the final lit image.

After the lighting pass, a separate **forward transparent pass** renders water/particles
that need blending (deferred rendering cannot handle transparency).

---

## 30. Advanced: Shadow Mapping

Shadow mapping renders the scene from the light's point of view into a depth-only
texture. Then in the main pass, each pixel is tested against the shadow map to determine
if it is in shadow.

SleakEngine's shadow map is 4096×4096:

```cpp
static constexpr uint32_t SHADOW_MAP_SIZE = 4096;
```

The shadow render pass has only a depth attachment, no color:

```cpp
VkAttachmentDescription depthAttachment{};
depthAttachment.format        = VK_FORMAT_D32_SFLOAT;
depthAttachment.samples       = VK_SAMPLE_COUNT_1_BIT;
depthAttachment.loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
depthAttachment.storeOp       = VK_ATTACHMENT_STORE_OP_STORE; // keep depth values!
depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
depthAttachment.finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

VkSubpassDescription subpass{};
subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.colorAttachmentCount    = 0;   // no color output
subpass.pDepthStencilAttachment = &depthRef;
```

The shadow pipeline uses a vertex-only shader (no fragment shader needed — just depth):

```cpp
bool VulkanRenderer::CreateShadowPipeline() {
    VulkanShader* shader = new VulkanShader(device);
    shader->compileVertexOnly("assets/shaders/shadow_pass");
    // Pipeline with only vertex stage, no fragment stage
    // ...
    rasterizer.depthBiasEnable = VK_TRUE;  // prevent shadow acne
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor    = 1.75f;
}
```

In `BindConstantBuffer` during the shadow pass, SleakEngine overrides the WVP matrix
with `LightVP * World` so all geometry is rendered from the light's perspective.

The shadow depth image is then bound to descriptor set 3 for the main lighting pass to
compare fragment depth against it.

---

## 31. Advanced: Specialized Pipelines

SleakEngine maintains multiple pipelines that share the same pipeline layout but differ
in shader or rasterizer configuration:

### Skybox Pipeline

```
- Front face: VK_FRONT_FACE_CLOCKWISE  (cube interior faces the camera)
- Depth test: VK_COMPARE_OP_LESS_OR_EQUAL (skybox is at maximum depth)
- Depth write: VK_FALSE (don't overwrite depth with infinity)
- Cull mode: NONE (we are inside the cube)
- Shader: skybox.vert / skybox.frag with samplerCube
```

### Voxel Pipeline

The voxel pipeline uses a different vertex format — a compact 48-byte `VoxelVertex`
instead of the 64-byte standard `Vertex`. When `BindVertexBuffer` detects a voxel
format buffer, it automatically switches to the voxel pipeline:

```cpp
// From VulkanRenderer.cpp line ~652:
bool wantVoxel = buffer->IsVoxelFormat();
if (wantVoxel != m_inVoxelPass && !m_inForwardTransparentPass) {
    if (wantVoxel) BeginVoxelPass();
    else EndVoxelPass();
}
```

### Skinned Pipeline

The skinned pipeline is identical to the standard pipeline but uses a different vertex
shader that reads bone indices and weights and computes a blended transform. It binds
set 1 (bone UBO) with the 128 bone matrices.

### Water / Transparent Pipeline

The water pipeline is used inside `BeginForwardTransparentPass`:
- Depth test: `VK_COMPARE_OP_LESS` (normal)
- Depth write: `VK_FALSE` (transparent: don't occlude things behind it)
- Blending: alpha blending enabled
- Render pass: `m_forwardRenderPass` (not the GBuffer pass)

### Debug Line Pipeline

```
- Primitive topology: VK_PRIMITIVE_TOPOLOGY_LINE_LIST
- Depth test: disabled (lines always visible on top)
- Polygon mode: VK_POLYGON_MODE_LINE
```

---

## 32. Cleanup — Destroying Everything in the Right Order

Vulkan has strict rules: child objects must be destroyed before parent objects. If you
destroy the `VkDevice` before destroying a `VkBuffer` that belongs to it, you have
undefined behavior (likely a crash or GPU hang).

The correct destruction order, derived from SleakEngine's `Cleanup()`:

```
1.  vkDeviceWaitIdle()                          // wait for all GPU work
2.  Destroy ImGui (uses device + descriptor pools)
3.  Destroy descriptor pools (frees descriptor sets too)
4.  Destroy all pipelines
5.  Destroy MSAA color resources (image, view, memory)
6.  Destroy GBuffer resources
7.  Destroy shadow resources
8.  Destroy bone UBO resources
9.  Destroy descriptor set layouts
10. Destroy depth resources (image, view, memory)
11. vkDestroySwapchainKHR
12. Destroy all framebuffers
13. Destroy all swapchain image views
    (do NOT destroy swapchain images — swapchain owns them)
14. Destroy pipeline layout
15. Destroy render pass
16. Destroy sync objects (fences, semaphores)
17. vkFreeCommandBuffers (optional — freed when pool is destroyed)
18. vkDestroyCommandPool
19. vkDestroyDevice                             // logical device
20. Destroy debug messenger (uses instance proc addr)
21. vkDestroySurfaceKHR
22. vkDestroyInstance                           // instance last
```

Key rules from SleakEngine:

- Always call `vkDeviceWaitIdle()` before destroying anything (line 965 in Cleanup).
- Destroying a pool (descriptor pool, command pool) automatically frees all objects
  allocated from it — you don't need to free them individually.
- Swapchain images are owned by the swapchain — never call `vkDestroyImage` on them.
- Image views and samplers must be destroyed before the images they reference.
- Buffers must be destroyed before the device memory they are bound to.

---

## Summary: The Complete Vulkan Object Hierarchy

```
VkInstance
├── VkDebugUtilsMessengerEXT
├── VkSurfaceKHR
└── VkPhysicalDevice (enumerated, not owned)
    └── VkDevice
        ├── VkQueue (retrieved, not owned)
        ├── VkSwapchainKHR
        │   ├── VkImage[] (owned by swapchain)
        │   └── VkImageView[] (created from swapchain images)
        ├── VkImage (created separately: depth, MSAA, GBuffer, shadow)
        │   ├── VkDeviceMemory (allocated for each image)
        │   └── VkImageView
        ├── VkBuffer
        │   └── VkDeviceMemory
        ├── VkSampler
        ├── VkRenderPass
        ├── VkFramebuffer (references VkImageView objects)
        ├── VkDescriptorSetLayout
        ├── VkDescriptorPool
        │   └── VkDescriptorSet[] (allocated from pool)
        ├── VkPipelineLayout
        ├── VkShaderModule
        ├── VkPipeline
        ├── VkCommandPool
        │   └── VkCommandBuffer[]
        ├── VkSemaphore
        └── VkFence
```

Every object lives on the device (except instance-level objects). Every object you
create, you must destroy. Every frame you submit must finish before you destroy
anything it used.

---

## Where to Go Next

Now that you understand the full API, here is what to study in SleakCraft specifically:

| Topic | File | Lines |
|-------|------|-------|
| Full init sequence | `Engine/Engine/src/VulkanRenderer.cpp` | 71–144 |
| Swapchain creation | `VulkanRenderer.cpp` | 1534–1592 |
| Render pass (MSAA) | `VulkanRenderer.cpp` | 2413–2520 |
| Pipeline creation | `VulkanRenderer.cpp` | 2211–2411 |
| Staging buffer pattern | `Engine/Engine/src/VulkanBuffer.cpp` | 50–113 |
| Texture loading | `Engine/Engine/src/VulkanTexture.cpp` | full file |
| Descriptor writes | `VulkanRenderer.cpp` | 2167–2209 |
| Frame loop | `VulkanRenderer.cpp` | 149–550 |
| GBuffer setup | `VulkanRenderer.cpp` | grep `CreateGBuffer` |
| Shadow map setup | `VulkanRenderer.cpp` | grep `CreateShadow` |
| Cleanup order | `VulkanRenderer.cpp` | 959–1150 |

The best way to learn is to trace through a single triangle from a game-layer `Draw()`
call all the way down to `vkQueuePresentKHR`. By the time you can do that without
checking anything, you understand Vulkan.
