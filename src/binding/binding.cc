#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <iostream>
#include "binding.h"

namespace py = pybind11;

Communicator::Communicator(std::vector<std::vector<int>> groups,
    std::vector<std::vector<int>> groupsRank, int nRanks, int nNodes, std::string topofile) {
  struct ncclXml* xml;
  ncclCalloc(&xml, 1);
  ncclTopoGetXmlFromFile(topofile.c_str(), xml, 1);


  struct ncclGraphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float speedIntra;
    float speedInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int collNetSupport;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
  } *graphData;
  ncclCalloc(&graphData, groups.size());

  struct ncclTopoGraph ringGraph;
  struct ncclTopoGraph treeGraph;
  struct ncclTopoGraph collNetGraph;

  int minCompCap, maxCompCap;
  for (int i = 0; i < groups.size(); i++) {
    auto& group = groups[i];
    auto& group_rank = groupsRank[i];

    ncclCalloc(&comm, 1);
    comm->rank = 0;
    comm->nNodes = nNodes;
    comm->nRanks = nRanks;

    struct ncclTopoSystem *system;
    ncclTopoGetSystemFromXml(xml, &system);
    memcpy(&comm->topo, &system, sizeof(system));
    ncclCalloc(&comm->peerInfo, group.size() + 1);

    for (int g = comm->topo->nodes[GPU].count - 1; g >= 0; g--) {
      bool find = false;
      for (int j = 0; j < group.size(); j++) {
        if (comm->topo->nodes[GPU].nodes[g].gpu.rank == group[j]) {
          comm->topo->nodes[GPU].nodes[g].gpu.rank = group_rank[j];
          comm->peerInfo[group_rank[j]].rank = group_rank[j];
          comm->peerInfo[group_rank[j]].cudaDev = comm->topo->nodes[GPU].nodes[g].gpu.dev;
          comm->peerInfo[group_rank[j]].gdrSupport = comm->topo->nodes[GPU].nodes[g].gpu.gdrSupport;
          comm->peerInfo[group_rank[j]].hostHash = 0x10b5976510b59765;
          comm->peerInfo[group_rank[j]].pidHash = 0x12984781327;
          comm->peerInfo[group_rank[j]].shmDev = 0x13;
          comm->peerInfo[group_rank[j]].busId = comm->topo->nodes[GPU].nodes[g].id;

          minCompCap = std::min(minCompCap, comm->topo->nodes[GPU].nodes[g].gpu.cudaCompCap);
          maxCompCap = std::max(maxCompCap, comm->topo->nodes[GPU].nodes[g].gpu.cudaCompCap);
          find = true;
          break;
        }
      }
      if (!find) {
        ncclTopoRemoveNode(comm->topo, GPU, g);
      }
    }

    ncclTopoComputePaths(comm->topo, comm->peerInfo);
    ncclTopoTrimSystem(comm->topo, comm);
    ncclTopoComputePaths(comm->topo, comm->peerInfo);
    ncclTopoSearchInit(comm->topo);
    ncclTopoPrint(comm->topo);

    ringGraph.id = 0;
    ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
    ringGraph.crossNic = 2;
    ringGraph.collNet = 0;
    ringGraph.minChannels = 1;
    ringGraph.maxChannels = MAXCHANNELS / 2;
    ncclTopoCompute(comm->topo, &ringGraph);
    ncclTopoPrintGraph(comm->topo, &ringGraph);

    treeGraph.id = 1;
    treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    treeGraph.crossNic = 2;
    treeGraph.collNet = 0;
    treeGraph.minChannels = 1;
    treeGraph.maxChannels = ringGraph.nChannels;
    ncclTopoCompute(comm->topo, &treeGraph);
    ncclTopoPrintGraph(comm->topo, &treeGraph);

    collNetGraph.id = 2;
    collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
    collNetGraph.collNet = 1;
    collNetGraph.crossNic = 2;
    collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
    ncclTopoCompute(comm->topo, &collNetGraph);
    ncclTopoPrintGraph(comm->topo, &collNetGraph);

    comm->collNetSupport = 0;

    graphData[i].tree.pattern = treeGraph.pattern;
    graphData[i].tree.nChannels = treeGraph.nChannels;
    graphData[i].tree.sameChannels = treeGraph.sameChannels;
    graphData[i].tree.speedIntra = treeGraph.speedIntra;
    graphData[i].tree.speedInter = treeGraph.speedInter;
    graphData[i].tree.typeIntra = treeGraph.typeIntra;
    graphData[i].tree.typeInter = treeGraph.typeInter;
    graphData[i].ring.pattern = ringGraph.pattern;
    graphData[i].ring.nChannels = ringGraph.nChannels;
    graphData[i].ring.sameChannels = ringGraph.sameChannels;
    graphData[i].ring.speedIntra = ringGraph.speedIntra;
    graphData[i].ring.speedInter = ringGraph.speedInter;
    graphData[i].ring.typeIntra = ringGraph.typeIntra;
    graphData[i].ring.typeInter = ringGraph.typeInter;
    graphData[i].collNet.pattern = collNetGraph.pattern;
    graphData[i].collNet.nChannels = collNetGraph.nChannels;
    graphData[i].collNet.sameChannels = collNetGraph.sameChannels;
    graphData[i].collNet.speedIntra = collNetGraph.speedIntra;
    graphData[i].collNet.speedInter = collNetGraph.speedInter;
    graphData[i].collNet.typeIntra = collNetGraph.typeIntra;
    graphData[i].collNet.typeInter = collNetGraph.typeInter;
    graphData[i].collNetSupport = comm->collNetSupport;

    comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  }

  int nChannelsOrig = comm->nChannels;
  for (int i = 0; i < groups.size(); i++) {
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(graphData[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(graphData[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(graphData[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(graphData[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.typeIntra = std::min(graphData[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::min(graphData[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(graphData[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(graphData[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(graphData[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(graphData[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.typeIntra = std::min(graphData[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::min(graphData[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(graphData[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(graphData[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.speedIntra = std::min(graphData[i].collNet.speedIntra, collNetGraph.speedIntra);
    collNetGraph.speedInter = std::min(graphData[i].collNet.speedInter, collNetGraph.speedInter);
    collNetGraph.typeIntra = std::min(graphData[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::min(graphData[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(graphData[i].collNetSupport, comm->collNetSupport);
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i = 0; i < comm->nChannels; i++)
      memcpy(comm->channels + comm->nChannels + i, comm->channels + nChannelsOrig + i, sizeof(struct ncclChannel));
  }

  ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph);
}

float getAlgoInfo(struct ncclInfo* info, int collNetTypeSupport, int numPipeOps) {
  // Ring Simple
  float time;
  ncclTopoGetAlgoTime(info, 1, 2, numPipeOps, &time);
  if (time >= 0) return time * 1e-3;

  float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  info->algorithm = -1;
  info->protocol = -1;
  if (info->comm->nRanks == 1) return 0;
  int nAlgos = NCCL_NUM_ALGORITHMS;
  for (int a=0; a<nAlgos; a++) {
    if (a == NCCL_ALGO_COLLNET && collNetTypeSupport != 1) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      float time;
      ncclTopoGetAlgoTime(info, a, p, numPipeOps, &time);
      if (time >= 0 && time < minTime) {
        info->algorithm = a;
        info->protocol = p;
        minTime = time;
      }
    }
  }
  if (info->algorithm == -1 || info->protocol == -1) {
    WARN("Error : no algorithm/protocol available");
    return 0;
  }
  return minTime * 1e-3;
}

float Communicator::broadcast(size_t nBytes, int root){
  struct ncclInfo info = {ncclFuncBroadcast, "Broadcast",
                          nullptr, nullptr, nBytes, ncclChar, ncclSum, root, comm, nullptr, /* Args */
                          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS};
  info.nBytes = nBytes;

  return getAlgoInfo(&info, 0, 1);
}

float Communicator::reduce(size_t nBytes, int root){
  struct ncclInfo info = {ncclFuncReduce, "Reduce",
                          nullptr, nullptr, nBytes, ncclChar, ncclSum, root, comm, nullptr, /* Args */
                          REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS};
  info.nBytes = nBytes;

  return getAlgoInfo(&info, 0, 1);
}


float Communicator::allreduce(size_t nBytes){
  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                          nullptr, nullptr, nBytes, ncclChar, ncclSum, 0, comm, nullptr, /* Args */
                          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  info.nBytes = nBytes;

  return getAlgoInfo(&info, 0, 1);
}

float Communicator::allgather(size_t nBytes){
  struct ncclInfo info = {ncclFuncAllGather, "AllGather",
                          nullptr, nullptr, nBytes, ncclChar, ncclSum, 0, comm, nullptr, /* Args */
                          ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  info.nBytes = nBytes;

  return getAlgoInfo(&info, 0, 1);
}

float Communicator::reducescatter(size_t nBytes){
  struct ncclInfo info = {ncclFuncReduceScatter, "ReduceScattere",
                          nullptr, nullptr, nBytes, ncclChar, ncclSum, 0, comm, nullptr, /* Args */
                          REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS};
  info.nBytes = nBytes;

  return getAlgoInfo(&info, 0, 1);
}


PYBIND11_MODULE(binding, m) {
  py::class_<Communicator> communicator(m, "Communicator");
  communicator.def(py::init<std::vector<std::vector<int>>,
      std::vector<std::vector<int>>, int, int, std::string>())
      .def("broadcast", &Communicator::broadcast)
      .def("reduce", &Communicator::reduce)
      .def("allreduce", &Communicator::allreduce)
      .def("allgather", &Communicator::allgather)
      .def("reduce_scatter", &Communicator::reducescatter);
}