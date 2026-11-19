// The least a user has to write to use the auto-tuner: wrap the computation in an Algorithm, tune
// it once, run it with the tuner. Everything else is left at its default.
//
//   client_tune_simple      (results go to mensurflow_tuning.txt; the second run reuses them)
#include "mensurflow/mensurflow.hpp"

#include <iostream>
#include <vector>

using namespace mensurflow;

int main() {
    // The computation, written as usual.
    const Algorithm network([](const std::vector<Tensor>& in) {
        return std::vector<Tensor>{matmul(relu(matmul(in[0], in[1])), in[2])};
    });
    const std::vector<Tensor> inputs = {uniform({256, 512}), uniform({512, 1024}),
                                        uniform({1024, 256})};

    Tuner tuner;                 // reads mensurflow_tuning.txt, if there is one
    tuner.tune(network, inputs); // finds the best settings for these inputs and saves them
    const Tensor out = network.run(inputs, tuner)[0]; // runs with them

    std::cout << "output " << to_string(out.shape()) << '\n';
    return 0;
}
