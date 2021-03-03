/**
 * Function definitions for a Python Module for NextDoor's sampling implementations.
 * To generate a Python Module, user needs to do following:
 * * Include this file at the end of your sampling implementation.
 * * Define the macro INIT_FUNC_NAME that should be of the form init<PythonModuleName>
 * * Define a string that is the module name as const char* moduleName = "<PythonModuleName>"
 * * Generate .so with the same name as the Python Module Name, i.e., PythonModuleName.so
 * 
 * The Python Module contains following functions:
 * * initSampling(graph): It takes a path to graph that is stored in a binary edge-list format.
 * * sample(): This will perform the sampling
 * * finalSamples(): This will return the final samples in the form of a Python List.
 * * finalSampleLength(): This will return the size of each sample.
 * * finalSamplesArray(): This avoids building a list and gives a pointer to the final samples array of NextDoor.
 *                        This pointer can then be converted to a numpyarray. 
 */

const char* docString = "The Python Module contains following functions:\n"
" * initSampling(graph): It takes a path to graph that is stored in a binary edge-list format.\n"
" * sample(): This will perform the sampling\n"
" * finalSamples(): This will return the final samples in the form of a Python List.\n"
" * finalSampleLength(): This will return the size of each sample.\n"
" * finalSamplesArray(): This avoids building a list and gives a pointer to the final samples array of NextDoor.\n"
"                        This pointer can then be converted to a numpyarray. \n";

#ifdef PYTHON_2 
#include <Python.h>
#endif

#ifdef PYTHON_3 
#include <Python.h>
#endif

#include "nextdoor.cu"

static CSR* graph_csr = nullptr;
static Graph graph;
static GPUCSRPartition gpuCSRPartition;

static PyObject * initSampling(PyObject *self, PyObject *args)
{
  char* graph_file;
  if (!PyArg_ParseTuple(args, "s", &graph_file)) {
    return NULL;
  }
  graph_csr = loadGraph(graph, graph_file, "edge-list", "binary");
  assert(graph_csr != nullptr);
  gpuCSRPartition = transferCSRToGPU(graph_csr);
  allocNextDoorDataOnGPU(graph_csr, nextDoorData);
  Py_RETURN_NONE;
}

static PyObject* sample(PyObject *self, PyObject *args)
{
  doSampleParallelSampling(graph_csr, gpuCSRPartition, nextDoorData);
  Py_RETURN_NONE;
}

static PyObject* finalSamples(PyObject *self, PyObject *args)
{
  std::vector<VertexID_t>& final = getFinalSamples(nextDoorData);
  PyObject* result = PyList_New(0);
  int i;

  for (i = 0; i < final.size(); ++i)
  {
      PyList_Append(result, PyLong_FromLong(final[i]));
  }

  return result;
}

static PyObject* finalSampleLength(PyObject *self, PyObject *args)
{
  std::vector<VertexID_t>& final = getFinalSamples(nextDoorData);

  return Py_BuildValue("i", final.size());
}

extern "C" int* finalSamplesArray()
{
  return &getFinalSamples(nextDoorData)[0];
}

static PyMethodDef ModuleMethods[] = {
  {"initSampling",  initSampling, METH_VARARGS, "initSampling(graph): Initialize NextDoor. Takes a path to graph that is stored in a binary edge-list format."},
  {"sample",  sample, METH_VARARGS, "sample(): This will perform the sampling on GPU"},
  {"finalSamples", finalSamples, METH_VARARGS, "finalSamples(): This will return the final samples in the form of a Python List."},
  {"finalSampleLength", finalSampleLength, METH_VARARGS, "finalSampleLength(): This will return the size of each sample."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


#ifdef PYTHON_2
extern "C"
PyMODINIT_FUNC INIT_FUNC_NAME ()
{
    Py_InitModule(moduleName,ModuleMethods);
}
#endif

#ifdef PYTHON_3
static struct PyModuleDef SamplingModule = {
  PyModuleDef_HEAD_INIT,
  moduleName,
  NULL,
  -1,
  ModuleMethods
};
extern "C"
PyMODINIT_FUNC INIT_FUNC_NAME(void)
{
  PyObject *m;

  m = PyModule_Create(&SamplingModule);
  if (m == NULL)
    return NULL;
  

  return m;
}
#endif