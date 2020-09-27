#include <Python.h>

#include "nextdoor.cu"

static CSR* graph_csr = nullptr;
static Graph graph;
static GPUCSRPartition gpuCSRPartition;
static NextDoorData nextDoorData;

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
  doSampling(gpuCSRPartition, nextDoorData);
  Py_RETURN_NONE;
}

static PyObject* finalSamples(PyObject *self, PyObject *args)
{
  std::vector<VertexID_t>& final = getFinalSamples(nextDoorData);
  PyObject* result = PyList_New(0);
  int i;

  for (i = 0; i < final.size(); ++i)
  {
      PyList_Append(result, PyInt_FromLong(final[i]));
  }

  return result;
}

static PyMethodDef NextDoorMethods[] = {
  {"initSampling",  initSampling, METH_VARARGS, "Initialize Sampling"},
  {"sample",  sample, METH_VARARGS, "Do Sampling on GPU"},
  {"finalSamples", finalSamples, METH_VARARGS, "Final Samples"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initNextDoor ()
{
    Py_InitModule("NextDoor",NextDoorMethods);
}
