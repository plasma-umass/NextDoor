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
  graph_csr = loadGraph(graph, graph_file, "adj-list", "text");
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

static PyMethodDef NextDoorMethods[] = {
  {"initSampling",  initSampling, METH_VARARGS, "Initialize Sampling"},
  {"sample",  sample, METH_VARARGS, "Do Sampling on GPU"},
  {"finalSamples", finalSamples, METH_VARARGS, "Final Samples"},
  {"finalSampleLength", finalSampleLength, METH_VARARGS, "Length of Final Samples Array"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

#ifdef PYTHON_2
extern "C"
PyMODINIT_FUNC initNextDoor ()
{
    Py_InitModule("NextDoor",NextDoorMethods);
}
#endif

#ifdef PYTHON_3
static struct PyModuleDef NextDoorModule = {
  PyModuleDef_HEAD_INIT,
  "NextDoor",
  NULL,
  -1,
  NextDoorMethods
};
extern "C"
PyMODINIT_FUNC PyInit_NextDoor(void)
{
  PyObject *m;

  m = PyModule_Create(&NextDoorModule);
  if (m == NULL)
    return NULL;
  

  return m;
}
#endif