import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll


def alltoall_FLASH(num_nodes, gpus_per_node, protocol):
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)


    with MSCCLProgram("FLASH_all_to_all", topology, collective, 1, protocol=protocol):

        for n1 in range(num_nodes):
            for step in range(1, num_nodes):
                # server i sends to server i+step
                n2 = (n1 + step) % num_nodes
                for g1 in range(gpus_per_node):
                    rank1 = n1 * gpus_per_node + g1
                    rank2 = n2 * gpus_per_node + g1
                    c = chunk(rank1, Buffer.input, n2 * gpus_per_node, gpus_per_node)    # one big transfer
                    c = c.copy(rank2, f'temp-{step%2}', ch=((n1+n2) % gpus_per_node)*2+(rank1%2)+2)
                    for g2 in range(gpus_per_node):
                        c = chunk(rank2, f'temp-{step%2}', g2, 1)
                        c = c.copy(c.get_dst_rank(), Buffer.output, c.get_dst_index())


        # intrinsic alltoall
        for n1 in range(num_nodes):
            for g1 in range(gpus_per_node):
                rank1 = n1 * gpus_per_node + g1
                for g2 in range(gpus_per_node):
                    c = chunk(rank1, Buffer.input, n1 * gpus_per_node + g2)
                    c = c.copy(c.get_dst_rank(), Buffer.output, c.get_dst_index())

        XML() # Prints the XML
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('gpus_per_node', type=int, help ='gpus per node')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()


alltoall_FLASH(args.num_nodes, args.gpus_per_node, args.protocol)
