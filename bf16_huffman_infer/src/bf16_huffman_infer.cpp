#include <Python.h>
#include <torch/csrc/stable/library.h>

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
STABLE_TORCH_LIBRARY(bf16_huffman_infer, m) {
    m.def("gemv_bf16_huffman(Tensor A_rem, Tensor A_exp, Tensor X, Tensor(a!) Y, Tensor offsets, Tensor LUT1, Tensor LUT2, Tensor LUT3, Tensor LUT4, Tensor code_lengths) -> ()");
    m.def("huffman_encode(Tensor data, Tensor LUT, Tensor(a!) output, Tensor(b!) output_lengths) -> ()");
    m.def("huffman_decode(Tensor A_rem, Tensor A_exp, Tensor(a!) Y, Tensor offsets, Tensor LUT1, Tensor LUT2, Tensor LUT3, Tensor LUT4, Tensor code_lengths) -> ()");
    
    m.def("gemv_bf16_ans(Tensor A_rem, Tensor A_exp, Tensor X, Tensor(a!) Y, Tensor offsets, Tensor LUT) -> ()");
    m.def("ans_encode(Tensor data, Tensor freq, Tensor cum, Tensor(a!) output, Tensor(b!) output_lengths) -> ()");
    m.def("ans_decode(Tensor A_rem, Tensor A_exp, Tensor(a!) Y, Tensor offsets, Tensor LUT) -> ()");
}

}


