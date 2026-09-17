import numpy as np

def optimal_irsa(slot_selection):
    """
    Simulate the IRSA process for a given slot selection and return the observation for each node.
    The observation can be:
    - 0: Collision (more than one node transmitting in the same slot)
    - 1: No transmission (node did not transmitted in the slot)
    - 2: Successful decoding (node transmitted in the slot and was successfully decoded)
    """
    num_nodes, num_slots = slot_selection.shape
    decoded_values = np.zeros(num_nodes)

    # check for a slot with exactly one node transmitting
    sum_slots = slot_selection.sum(axis=0)

    while np.any(sum_slots == 1):
        # find the index of the slot with exactly one node transmitting
        idx = np.where(sum_slots == 1)[0][0]

        # Find the node that is transmitting in that slot
        node_idx = np.where(slot_selection[:, idx] == 1)[0][0]
        
        # decode the value for that node
        decoded_values[node_idx] = 1  # Placeholder for actual value decoding logic
        
        # remove the decoded node from the slot selection
        slot_selection[node_idx, :] = 0
        
        # update the sum of slots
        sum_slots = slot_selection.sum(axis=0)

    # Check for collisions 
    if np.any(sum_slots > 1):
        obs = np.zeros(num_nodes) # Collision
    else:
        obs = np.ones(num_nodes) # No transmission with correct decoding (IMPLICIT INFORMATION)

    obs[decoded_values == 1] = 2 # Successful decoding
    return obs



def imperfect_irsa(slot_selection, erasure_prob=0.0, sic_efficiency=0.95):
    """
    Simulate the IRSA process for a given slot selection and return the observation for each node.
    The observation can be:
    - 0: Collision (more than one node transmitting in the same slot)
    - 1: No transmission (node did not transmitted in the slot)
    - 2: Successful decoding (node transmitted in the slot and was successfully decoded)
    
    This function simulates an imperfect IRSA process where there is a small probability
    of erasure and the SIC process is not perfect. 
    - erasure_prob: probability that a transmitted packet is erased and not received
    - sic_efficiency (gamma): w_l = gamma ** (l-1) where l-1 is the number of decoded packets
      in the same slot and w_l is the probability of successful decoding after l-1 packets 
      have been decoded in the same slot.
    """
    num_nodes, num_slots = slot_selection.shape
    decoded_values = np.zeros(num_nodes)

    # Simulate the erasure of packets
    erasures = np.random.rand(num_nodes, num_slots) < erasure_prob
    slot_selection[erasures] = 0

    # check for a slot with exactly one node transmitting
    sum_slots = slot_selection.sum(axis=0)
    
    # Simulate the imperfect SIC process
    w_l = np.array([sic_efficiency ** (l-1) for l in sum_slots])

    # Collision detetected 
    c_d = False

    while np.any(sum_slots == 1):
        # find the index of the slot with exactly one node transmitting
        idx = np.where(sum_slots == 1)[0][0]

        # Find the node that is transmitting in that slot
        node_idx = np.where(slot_selection[:, idx] == 1)[0][0]
        
        # decode the value for that node
        if np.random.rand() < w_l[idx]: # Successful decoding with probability w_l
            decoded_values[node_idx] = 1  # Placeholder for actual value decoding logic
        else:
            c_d = True
        
        # remove the decoded node from the slot selection
        slot_selection[node_idx, :] = 0
        
        # update the sum of slots
        sum_slots = slot_selection.sum(axis=0)

    # Check for collisions 
    if np.any(sum_slots > 1) or c_d: # If there is a slot with more than one node transmitting and no successful decoding, then it is a collision
        obs = np.zeros(num_nodes) # Collision
    else:
        obs = np.ones(num_nodes) # No transmission with correct decoding (IMPLICIT INFORMATION)

    obs[decoded_values == 1] = 2 # Successful decoding
    return obs


degree_distribution = {
    # https://ieeexplore.ieee.org/document/7736044
    "bcsa": {
        "replicas": [3,8],
        "replicas_prob": [0.86, 0.14]
    },
    "bcsa2": {
        "replicas": [3,8],
        "replicas_prob": [0.8, 0.2]
    },
    # https://ieeexplore.ieee.org/document/5668922 
    "liva_g_0.8548": {
        "replicas": [2,4],
        "replicas_prob": [0.5102, 0.4898]
    },
    "liva_g_0.898": {
        "replicas": [2,3,8],
        "replicas_prob": [0.5631, 0.0436, 0.3933]
    },
    "liva_g_0.915":
    {
        "replicas": [2,3,6],
        "replicas_prob": [0.5465, 0.1623, 0.2912]
    },
    "liva_g_0.938": 
    {
        "replicas": [2,3,8],
        "replicas_prob": [0.5, 0.28, 0.22]
    },
    "custom": 
    {
        "replicas": [2,3,4,6,8],
        "replicas_prob": [0.2, 0.2, 0.2, 0.2, 0.2]
    },
    "chat": 
    {
        "replicas": [2,3,4],
        "replicas_prob": [0.4, 0.4, 0.2]
    },
    "lambda3":
    {
        "replicas": [3,4],
        "replicas_prob": [0.86, 0.14]
    },
    "x3":
    {
        "replicas": [3],
        "replicas_prob": [1.0]
    },
    "x4":
    {
        "replicas": [4],
        "replicas_prob": [1.0]
    }
}


import numpy as np
from collections import deque

def optimal_irsa_fast(slot_selection):
    num_nodes, num_slots = slot_selection.shape

    # Copy to avoid modifying input
    C = slot_selection.copy()

    decoded = np.zeros(num_nodes, dtype=bool)

    # Precompute slot degrees
    slot_degree = C.sum(axis=0)

    # Build adjacency (much faster access)
    node_to_slots = [np.where(C[i])[0] for i in range(num_nodes)]
    slot_to_nodes = [np.where(C[:, j])[0] for j in range(num_slots)]

    # Initialize queue with singleton slots
    queue = deque(np.where(slot_degree == 1)[0])

    while queue:
        slot = queue.popleft()

        if slot_degree[slot] != 1:
            continue  # might have changed

        # Find the only active node in this slot
        for node in slot_to_nodes[slot]:
            if not decoded[node]:
                node_idx = node
                break
        else:
            continue

        # Decode node
        decoded[node_idx] = True

        # Remove all replicas of this node
        for s in node_to_slots[node_idx]:
            if slot_degree[s] > 0:
                slot_degree[s] -= 1

                if slot_degree[s] == 1:
                    queue.append(s)

    # Build output
    obs = np.zeros(num_nodes)

    if np.any(slot_degree > 1):
        obs[:] = 0  # collision
    else:
        obs[:] = 1  # no transmission / resolved

    obs[decoded] = 2
    return obs


import numpy as np

def build_csr_from_binary(C):
    """Build CSR-like structures for nodes->slots and slots->nodes."""
    num_nodes, num_slots = C.shape

    # Node -> slots
    node_deg = C.sum(axis=1).astype(np.int32)
    node_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    node_ptr[1:] = np.cumsum(node_deg)

    node_edges = np.zeros(node_ptr[-1], dtype=np.int32)

    cursor = np.zeros(num_nodes, dtype=np.int32)
    for j in range(num_slots):
        rows = np.where(C[:, j])[0]
        for r in rows:
            idx = node_ptr[r] + cursor[r]
            node_edges[idx] = j
            cursor[r] += 1

    # Slot -> nodes
    slot_deg = C.sum(axis=0).astype(np.int32)
    slot_ptr = np.zeros(num_slots + 1, dtype=np.int32)
    slot_ptr[1:] = np.cumsum(slot_deg)

    slot_edges = np.zeros(slot_ptr[-1], dtype=np.int32)

    cursor[:] = 0
    for j in range(num_slots):
        rows = np.where(C[:, j])[0]
        for r in rows:
            idx = slot_ptr[j] + cursor[j]
            slot_edges[idx] = r
            cursor[j] += 1

    return node_ptr, node_edges, slot_ptr, slot_edges


def irsa_peeling_cstyle(node_ptr, node_edges, slot_ptr, slot_edges, num_nodes, num_slots):
    decoded = np.zeros(num_nodes, dtype=np.uint8)
    slot_degree = np.diff(slot_ptr).copy()

    # Preallocate queue (fixed-size array, no Python deque)
    queue = np.zeros(num_slots, dtype=np.int32)
    q_head = 0
    q_tail = 0

    # Initialize queue with singleton slots
    for s in range(num_slots):
        if slot_degree[s] == 1:
            queue[q_tail] = s
            q_tail += 1

    while q_head < q_tail:
        slot = queue[q_head]
        q_head += 1

        if slot_degree[slot] != 1:
            continue

        # Find active node in slot
        start = slot_ptr[slot]
        end = slot_ptr[slot + 1]

        node = -1
        for i in range(start, end):
            n = slot_edges[i]
            if decoded[n] == 0:
                node = n
                break

        if node == -1:
            continue

        decoded[node] = 1

        # Remove all replicas of this node
        n_start = node_ptr[node]
        n_end = node_ptr[node + 1]

        for i in range(n_start, n_end):
            s = node_edges[i]
            if slot_degree[s] > 0:
                slot_degree[s] -= 1
                if slot_degree[s] == 1:
                    queue[q_tail] = s
                    q_tail += 1

    # Build observation
    obs = np.zeros(num_nodes, dtype=np.int32)

    if np.any(slot_degree > 1):
        obs[:] = 0
    else:
        obs[:] = 1

    obs[decoded == 1] = 2
    return obs

from numba import njit

njit_function = njit(irsa_peeling_cstyle)

def irsa_fast(slot_selection):
    node_ptr, node_edges, slot_ptr, slot_edges = build_csr_from_binary(slot_selection)
    return njit_function(node_ptr, node_edges, slot_ptr, slot_edges, *slot_selection.shape)