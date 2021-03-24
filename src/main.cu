#include "nextdoor.cu"

#include <anyoption.h>

template<class SampleType, typename App>
int appMain(int argc, char* argv[], bool (*checkResults) (NextDoorData<SampleType, App>& nextDoorData))
{ 
  AnyOption *opt = new AnyOption();
  opt->addUsage("usage: ");
  opt->addUsage("");
  opt->addUsage("-h --help              Prints this help");
  opt->addUsage("-g --graph-file        File containing graph");
  opt->addUsage("-t --graph-type <type> Format of graph file: 'adj-list' or 'edge-list'");
  opt->addUsage("-f --format <format>   Format of graph file: 'binary' or 'text'");
  opt->addUsage("-n --nruns             Number of runs");
  opt->addUsage("-c --check-results     Check results using an algorithm");
  opt->addUsage("-p --print-samples     Print Samples");
  opt->addUsage("-k --kernel-type       Type of Kernel: 'SampleParallel' or 'TransitParallel'");
  opt->addUsage("-l --load-balancing    Enable Load Balancing");

  opt->setFlag  ("help",           'h');
  opt->setOption("graph-file",     'g');
  opt->setOption("graph-type",     't');
  opt->setOption("graph-format",   'f');
  opt->setOption("nruns",          'n');
  opt->setOption("kernel-type",    'k');
  opt->setFlag  ("print-samples",  'p');
  opt->setFlag  ("check-results",  'c');
  opt->setFlag  ("load-balancing", 'l');

  opt->processCommandArgs(argc, argv);

  if (!opt->hasOptions()) {
    opt->printUsage();
    delete opt;
    return 0;
  }

  char* graph_file = opt->getValue('g');
  char* graph_type = opt->getValue('t');
  char* graph_format = opt->getValue('f');

  if (graph_file == nullptr || graph_type == nullptr || 
      graph_format == nullptr) {
    opt->printUsage();
    delete opt;
    return 0;
  }

  nextdoor<SampleType, App>(opt->getValue('g'), opt->getValue('t'), opt->getValue('f'), 
                  atoi(opt->getValue('n')), opt->getFlag("check-results"), opt->getFlag("print-samples"),
                  opt->getValue('k'), opt->getFlag('l'), checkResults);

  return 0;
}