// A tour of mensurflow from a user's point of view: find devices, make tensors, run ops on the
// CPU or a GPU, read results back, and take a gradient.
#include "mensurflow/mensurflow.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mensurflow;

namespace {

void print(const char* label, const Tensor& t) {
    std::cout << label << " " << to_string(t.shape()) << " on " << to_string(t.device()) << ": ";
    for (float v : t.to_vector<float>()) {
        std::cout << v << ' ';
    }
    std::cout << '\n';
}

bool has_gpu() {
    for (Device d : list_devices()) {
        if (d.type != DeviceType::CPU) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    // 1. Which devices can this build use? CPU is always there; CUDA or Metal appear only when
    //    compiled in and the hardware is present.
    std::cout << "devices:";
    for (Device d : list_devices()) {
        std::cout << ' ' << to_string(d);
    }
    std::cout << "\ndefault device: " << to_string(default_device()) << "\n\n";

    // 2. Tensors. Data is copied from host vectors; tensors live on the CPU unless told otherwise.
    const Tensor a = Tensor::from_vector<float>({1, 2, 3, 4}, {4});
    const Tensor b = Tensor::from_vector<float>({10, 20, 30, 40}, {4});
    const Tensor ones = Tensor::ones({4});

    // 3. Ops. Every op takes an Execution as its last argument; Auto is the default.
    std::cout << "dot(a, b) = " << dot(a, b).item<float>() << '\n';
    print("add(a, b)", add(a, b));
    print("relu(sub(a, 2))", relu(sub(a, Tensor::full({4}, 2.0))));
    std::cout << "sum(a) = " << sum(a).item<float>() << ", max(b) = " << max(b).item<float>()
              << "\n\n";

    // 4. Matrices: views like transpose cost nothing; matmul reads them in place.
    const Tensor m = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
    print("m", m);
    print("m @ m^T", matmul(m, m.transpose(0, 1)));
    print("row sums", sum(m, 1));
    std::cout << "argmax per row:"; // int64 indices, one per row
    for (std::int64_t i : argmax(m).to_vector<std::int64_t>()) {
        std::cout << ' ' << i;
    }
    std::cout << "\n\n";

    // 5. Choosing where an op runs. Data stays where it was created; Execution::GPU copies the
    //    inputs to the GPU, runs there, and copies the result back.
    if (has_gpu()) {
        std::cout << "dot on GPU = " << dot(a, b, Execution::GPU).item<float>() << '\n';

        // Keeping data on the GPU avoids those copies.
        const Device gpu = default_device();
        const Tensor ga = a.to(gpu);
        const Tensor gb = b.to(gpu);
        print("add on GPU, data on GPU", add(ga, gb, Execution::GPU));
    } else {
        std::cout << "no GPU in this build; Execution::GPU would throw\n";
    }
    std::cout << '\n';

    // 6. Autograd: mark leaves, build a scalar, call backward(), read grad().
    Tensor w = Tensor::from_vector<float>({0.5f, -1.0f, 2.0f}, {3});
    w.set_requires_grad();
    const Tensor x = Tensor::from_vector<float>({1, 2, 3}, {3});
    const Tensor loss = sum(square(mul(w, x))); // d/dw = 2 * w * x^2
    loss.backward();
    std::cout << "loss = " << loss.item<float>() << '\n';
    print("dloss/dw", w.grad());
    std::cout << '\n';

    // 7. Mistakes are caught before any kernel runs, with a message that says what to fix.
    try {
        dot(a, Tensor::ones({3}));
    } catch (const std::invalid_argument& e) {
        std::cout << "error: " << e.what() << '\n';
    }
    return 0;
}
