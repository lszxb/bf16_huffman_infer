#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace bf16_huffman_infer {

// Defines the operators
TORCH_LIBRARY(bf16_huffman_infer, m) {
    m.def("gemv_bf16(Tensor A, Tensor X, Tensor(a!) Y) -> ()");
    m.def("gemv_bf16_huffman(Tensor A_rem, Tensor A_exp, Tensor X, Tensor(a!) Y, Tensor offsets, Tensor LUT1, Tensor LUT2, Tensor LUT3, Tensor LUT4, Tensor code_lengths, Tensor reorder_indices) -> ()");
    m.def("huffman_encode(Tensor data, Tensor LUT, Tensor(a!) output, Tensor(b!) output_lengths) -> ()");
    m.def("huffman_decode(Tensor A_rem, Tensor A_exp, Tensor(a!) Y, Tensor offsets, Tensor LUT1, Tensor LUT2, Tensor LUT3, Tensor LUT4, Tensor code_lengths) -> ()");
}

}


