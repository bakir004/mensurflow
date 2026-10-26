#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: VkResult " +
                                 std::to_string(result));
    }
}

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [&](const auto& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

// Own every Vulkan resource, including partially constructed state on failure.
struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::array<Buffer, 3> buffers{};
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            for (auto& buffer : buffers) {
                if (buffer.mapped) {
                    vkUnmapMemory(device, buffer.memory);
                }
                vkDestroyBuffer(device, buffer.handle, nullptr);
                vkFreeMemory(device, buffer.memory, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

std::vector<uint32_t> read_shader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::string("Cannot open shader: ") + path);
    }
    const auto bytes = file.tellg();
    if (bytes <= 0 || static_cast<std::size_t>(bytes) % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Invalid SPIR-V file size");
    }
    std::vector<uint32_t> code(static_cast<std::size_t>(bytes) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), bytes)) {
        throw std::runtime_error("Cannot read shader");
    }
    return code;
}

uint32_t dimension(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Dimensions must be positive decimal integers");
    }
    const auto n = std::stoull(value);
    // Keeps rounded dispatch dimensions and shader tile arithmetic in range.
    if (n == 0 || n > std::numeric_limits<uint32_t>::max() - 15u) {
        throw std::invalid_argument("Dimension is zero or too large");
    }
    return static_cast<uint32_t>(n);
}

std::size_t elements(uint32_t rows, uint32_t cols) {
    const uint64_t count = uint64_t(rows) * cols;
    if (count > std::numeric_limits<uint32_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("Matrix exceeds shader indexing or host allocation limits");
    }
    return static_cast<std::size_t>(count);
}

void create_buffer(Context& context, Buffer& buffer, VkPhysicalDevice physical,
                   VkDeviceSize bytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(context.device, &info, nullptr, &buffer.handle), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context.device, buffer.handle, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    const auto required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = properties.memoryTypeCount;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            type = i;
            break;
        }
    }
    if (type == properties.memoryTypeCount) {
        throw std::runtime_error("Example requires host-visible, coherent storage-buffer memory");
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    check(vkAllocateMemory(context.device, &allocation, nullptr, &buffer.memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(context.device, buffer.handle, buffer.memory, 0),
          "vkBindBufferMemory");
    check(vkMapMemory(context.device, buffer.memory, 0, bytes, 0, &buffer.mapped), "vkMapMemory");
}

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 4) {
            throw std::invalid_argument("Usage: vulkan_matmul [M K N]");
        }
        const uint32_t m = argc == 4 ? dimension(argv[1]) : 128;
        const uint32_t k = argc == 4 ? dimension(argv[2]) : 128;
        const uint32_t n = argc == 4 ? dimension(argv[3]) : 128;
        const std::array<std::size_t, 3> counts{elements(m, k), elements(k, n), elements(m, n)};
        Context context;

        uint32_t extension_count = 0;
        check(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr),
              "Enumerate instance extensions");
        std::vector<VkExtensionProperties> extensions(extension_count);
        check(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()),
              "Enumerate instance extensions");
        std::vector<const char*> instance_extensions;
        // Dependency of portability_subset when using the Vulkan 1.0 API.
        if (has_extension(extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        // Required to enumerate MoltenVK devices on current macOS Vulkan loaders.
#ifdef VK_KHR_portability_enumeration
        if (has_extension(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "PRS Vulkan matmul";
        application.apiVersion = VK_API_VERSION_1_0;
        instance_info.pApplicationInfo = &application;
        instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
        instance_info.ppEnabledExtensionNames = instance_extensions.data();
        check(vkCreateInstance(&instance_info, nullptr, &context.instance), "vkCreateInstance");

        uint32_t device_count = 0;
        check(vkEnumeratePhysicalDevices(context.instance, &device_count, nullptr),
              "Enumerate devices");
        std::vector<VkPhysicalDevice> devices(device_count);
        if (device_count == 0) {
            throw std::runtime_error(
                "No Vulkan devices: install/configure a Vulkan driver or MoltenVK");
        }
        check(vkEnumeratePhysicalDevices(context.instance, &device_count, devices.data()),
              "Enumerate devices");
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        uint32_t family = 0;
        VkPhysicalDeviceProperties properties{};
        // Prefer a hardware GPU over a software/CPU Vulkan implementation.
        int best_score = -1;
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties candidate_properties{};
            vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
            const auto& limits = candidate_properties.limits;
            if (limits.maxComputeWorkGroupInvocations < 256 ||
                limits.maxComputeWorkGroupSize[0] < 16 || limits.maxComputeWorkGroupSize[1] < 16 ||
                limits.maxComputeSharedMemorySize < 2 * 16 * 16 * sizeof(float) ||
                limits.maxComputeWorkGroupCount[0] < (n + 15) / 16 ||
                limits.maxComputeWorkGroupCount[1] < (m + 15) / 16 ||
                limits.maxStorageBufferRange <
                    *std::max_element(counts.begin(), counts.end()) * sizeof(float)) {
                continue;
            }
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            const int score =
                candidate_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                : candidate_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                : candidate_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                                            : 1;
            for (uint32_t i = 0; i < family_count; ++i) {
                if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    score > best_score) {
                    best_score = score;
                    physical = candidate;
                    family = i;
                    properties = candidate_properties;
                }
            }
        }
        if (!physical) {
            throw std::runtime_error(
                "No Vulkan compute device supports the requested size and 16x16 tile");
        }
        std::cout << "Device: " << properties.deviceName << '\n';
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            std::cout << "WARNING: software/CPU Vulkan device, not GPU execution.\n";
        }

        extension_count = 0;
        check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr),
              "Enumerate device extensions");
        extensions.resize(extension_count);
        check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count,
                                                   extensions.data()),
              "Enumerate device extensions");
        std::vector<const char*> device_extensions;
        // This extension must be enabled when advertised by a portability driver.
        if (has_extension(extensions, "VK_KHR_portability_subset")) {
            device_extensions.push_back("VK_KHR_portability_subset");
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();
        check(vkCreateDevice(physical, &device_info, nullptr, &context.device), "vkCreateDevice");
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(context.device, family, 0, &queue);

        std::vector<float> a(counts[0]), b(counts[1]), output(counts[2]);
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> random(-1.0f, 1.0f);
        for (auto& value : a)
            value = random(generator);
        for (auto& value : b)
            value = random(generator);
        for (std::size_t i = 0; i < context.buffers.size(); ++i) {
            create_buffer(context, context.buffers[i], physical, counts[i] * sizeof(float));
        }
        std::memcpy(context.buffers[0].mapped, a.data(), a.size() * sizeof(float));
        std::memcpy(context.buffers[1].mapped, b.data(), b.size() * sizeof(float));

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_info.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(context.device, &set_info, nullptr, &context.set_layout),
              "vkCreateDescriptorSetLayout");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &context.set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        check(
            vkCreatePipelineLayout(context.device, &layout_info, nullptr, &context.pipeline_layout),
            "vkCreatePipelineLayout");

        const auto code = read_shader(MATMUL_SHADER_PATH);
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = code.size() * sizeof(uint32_t);
        shader_info.pCode = code.data();
        check(vkCreateShaderModule(context.device, &shader_info, nullptr, &context.shader),
              "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = context.shader;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = context.pipeline_layout;
        check(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                       &context.pipeline),
              "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &context.descriptor_pool),
              "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo set_allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_allocation.descriptorPool = context.descriptor_pool;
        set_allocation.descriptorSetCount = 1;
        set_allocation.pSetLayouts = &context.set_layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(context.device, &set_allocation, &descriptor_set),
              "vkAllocateDescriptorSets");
        std::array<VkDescriptorBufferInfo, 3> buffer_infos{};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (uint32_t i = 0; i < writes.size(); ++i) {
            buffer_infos[i] = {context.buffers[i].handle, 0, counts[i] * sizeof(float)};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(),
                               0, nullptr);

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = family;
        check(
            vkCreateCommandPool(context.device, &command_pool_info, nullptr, &context.command_pool),
            "vkCreateCommandPool");
        VkCommandBufferAllocateInfo command_allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_allocation.commandPool = context.command_pool;
        command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(context.device, &command_allocation, &command),
              "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, context.pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, context.pipeline_layout, 0,
                                1, &descriptor_set, 0, nullptr);
        const std::array<uint32_t, 3> dimensions{m, k, n};
        vkCmdPushConstants(command, context.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(dimensions), dimensions.data());
        vkCmdDispatch(command, (n + 15) / 16, (m + 15) / 16, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(context.device, &fence_info, nullptr, &context.fence), "vkCreateFence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        const auto start = std::chrono::steady_clock::now();
        check(vkQueueSubmit(queue, 1, &submit, context.fence), "vkQueueSubmit");
        check(vkWaitForFences(context.device, 1, &context.fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences");
        const auto end = std::chrono::steady_clock::now();
        std::memcpy(output.data(), context.buffers[2].mapped, output.size() * sizeof(float));

        double max_error = 0.0;
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t col = 0; col < n; ++col) {
                double reference = 0.0;
                for (uint32_t t = 0; t < k; ++t) {
                    reference += double(a[std::size_t(row) * k + t]) * b[std::size_t(t) * n + col];
                }
                const float value = output[std::size_t(row) * n + col];
                const double error = std::abs(value - reference);
                max_error = std::max(max_error, error);
                if (!std::isfinite(value) || error > 1e-4 + 1e-4 * std::abs(reference)) {
                    throw std::runtime_error("GPU result differs from CPU reference at (" +
                                             std::to_string(row) + ", " + std::to_string(col) +
                                             ")");
                }
            }
        }
        std::cout << "A[" << m << ',' << k << "] * B[" << k << ',' << n << "] -> C[" << m << ','
                  << n << "]\nSubmit + wait: "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms (not kernel-only or full end-to-end timing)\n"
                  << "Correctness passed; max absolute error: " << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
