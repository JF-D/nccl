#ifndef NCCL_BINDING_H_
#define NCCL_BINDING_H_

#include <vector>
#include <string>

#include "comm.h"
#include "nccl.h"
#include "xml.h"
#include "topo.h"
#include "enqueue.h"
#include "transport.h"

class Communicator {
public:
  Communicator(std::vector<std::vector<int>> groups, std::vector<std::vector<int>> groupsRank,
      int nRanks, int nNodes, std::string topofile);

  float broadcast(size_t nBytes, int root);

  float reduce(size_t nBytes, int root);

  float allreduce(size_t nBytes);

  float allgather(size_t nBytes);

  float reducescatter(size_t nBytes);

  struct ncclComm* comm;
};


#endif
