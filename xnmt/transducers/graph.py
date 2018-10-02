from typing import List, Tuple
import numbers

import numpy as np
import dynet as dy

from xnmt import expression_seqs
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.param_collections import ParamManager
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.transducers import base as transducers
from xnmt.persistence import serializable_init, Serializable, Ref, bare


class GraphTransducer(transducers.SeqTransducer):
  """
  A generic graph network transducer, following the survey of Battaglia et al. (2018).
  TODO: (except that it doesn't have the graph global state, for now)
  
  The input should always contain three components:
  - A list of node vectors (length N).
  - A list of edge vectors (length E).
  - An adjacency list, with (src, trg) indices. Indices correspond to nodes in the node
  list, the tuple position correspond to the edge vector in the edge list. (length E)
  These inputs are not explicitly lists but dynet tensors.
  
  This class is a generic interface with signatures for three mandatory functions:
  - edge_update: calculate new hidden vectors for edges.
  - node_edge_aggregate: for each node, aggregate its edges into a single vector.
  - node_update: calculate new hidden vectors for nodes, using node_edge_aggregate.
  
  Unlike Battaglia et al. (2018) and more like Kearnes et al. (2016), nodes and edges are
  updated in parallel instead of sequentially. (Daniel: I think this makes more sense)
  Note that the functions above always take all three inputs. A final function takes the
  result from the three functions above and output an updated graph, containing new vectors
  for nodes and edges and *the same* adjacency list. This allows us to chain Graph Networks
  easily.
  """

  def transduce(self, graph: Tuple['expression_seqs.ExpressionSequence',
                                   'expression_seqs.ExpressionSequence',
                                   'expression_seqs.ExpressionSequence']
  ) -> Tuple['expression_seqs.ExpressionSequence',
             'expression_seqs.ExpressionSequence',
             'expression_seqs.ExpressionSequence']:
    """
    Encode the graph, generating a new one.
    """
    raise NotImplementedError("GraphTransducer.transduce() must be implemented by Transducer sub-classes")

  def edge_update(self, graph: Tuple['expression_seqs.ExpressionSequence',
                                     'expression_seqs.ExpressionSequence',
                                     'expression_seqs.ExpressionSequence']
  ) -> 'expression_seqs.ExpressionSequence':
    """
    Update edge hidden vectors
    """
    raise NotImplementedError("GraphTransducer.edge_update() must be implemented by Transducer sub-classes")

  def node_edge_aggregate(self, graph: Tuple['expression_seqs.ExpressionSequence',
                                             'expression_seqs.ExpressionSequence',
                                             'expression_seqs.ExpressionSequence']
  ) -> 'expression_seqs.ExpressionSequence':
    """
    Aggregate edge hidden vectors per node
    """
    raise NotImplementedError("GraphTransducer.node_edge_aggregate() must be implemented by Transducer sub-classes")
  
  def node_update(self, graph: Tuple['expression_seqs.ExpressionSequence',
                                     'expression_seqs.ExpressionSequence',
                                     'expression_seqs.ExpressionSequence']
  ) -> 'expression_seqs.ExpressionSequence':
    """
    Update node hidden vectors
    """
    raise NotImplementedError("GraphTransducer.node_update() must be implemented by Transducer sub-classes")

  
class GraphMLPTransducer(GraphTransducer, Serializable):
  """
  A graph network transducer with multilayer perceptrons as update functions and simple
  (element-wise) edge aggregation (Sanchez-Gonzalez et al. 2018).
  """
  yaml_tag = '!GraphMLPTransducer'

  @serializable_init
  def __init__(self,
               layers=1,
               node_hidden_dim=Ref("exp_global.default_layer_dim"),
               edge_hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               activation='relu',
               output_type='nodes_edges'):
    self.num_layers = layers
    self.node_hidden_dim = node_hidden_dim
    self.edge_hidden_dim = edge_hidden_dim
    if activation == 'relu':
      self.activation = dy.rectify
    elif activation == 'tanh':
      self.activation = dy.tanh
    self.output_type = output_type
    model = ParamManager.my_params(self)

    # Edge update MLP
    edge_update_dim = edge_hidden_dim + (node_hidden_dim * 2)
    self.edge_W = model.add_parameters(dim=(edge_hidden_dim, edge_update_dim),
                                       init=param_init.initializer((edge_hidden_dim, edge_update_dim)))
    self.edge_b = model.add_parameters(dim=(edge_hidden_dim,),
                                       init=bias_init.initializer((edge_hidden_dim,)))

    # Node update MLP
    node_update_dim = node_hidden_dim + edge_hidden_dim
    self.node_W = model.add_parameters(dim=(node_hidden_dim, node_update_dim),
                                       init=param_init.initializer((node_hidden_dim, node_update_dim)))
    self.node_b = model.add_parameters(dim=(node_hidden_dim,),
                                       init=bias_init.initializer((node_hidden_dim,)))

  def transduce(self, graph: Tuple['expression_seqs.ExpressionSequence',
                                   'expression_seqs.ExpressionSequence',
                                   List[numbers.Integral],
                                   List[numbers.Integral]]
                                   #'expression_seqs.ExpressionSequence']
  ) -> Tuple['expression_seqs.ExpressionSequence',
             'expression_seqs.ExpressionSequence',
             'expression_seqs.ExpressionSequence']:

    # Make everything a tensor
    nodes, edges, src_adj, trg_adj = graph
    nodes = nodes.as_tensor()
    edges = edges.as_tensor()

    for i in range(self.num_layers):
      # Get updated nodes and edges
      new_edges = self.edge_update(nodes, edges, src_adj, trg_adj)
      node_aggs = self.node_edge_aggregate(nodes, edges, src_adj, trg_adj)
      new_nodes = self.node_update(nodes, node_aggs)
      # Update
      nodes = new_nodes
      edges = new_edges      

    # Build a new ExpressionSequence
    self.nodes_ret = expression_seqs.ExpressionSequence(expr_tensor=nodes)
    # For now we just return nodes (as in Kearnes et al 2016)
    # TODO: return edges as well, requires a new attender    
    #return (new_nodes, new_edges, graph[2], graph[3])
    if self.output_type == "nodes_only":
      return self.nodes_ret
    else:
      self.edges_ret = expression_seqs.ExpressionSequence(expr_tensor=edges)
      return (self.nodes_ret, self.edges_ret)

  def edge_update(self,
                  nodes,
                  edges,
                  src_adj,
                  trg_adj) -> 'expression_seqs.ExpressionSequence':
    """
    Update edge vectors using a MLP.
    """
    edge_W = dy.parameter(self.edge_W)
    edge_b = dy.parameter(self.edge_b)

    # For each *edge*, get its source and target node vectors.
    # Then, append these to the current edge vector.
    # This forms the input to the MLP, which outputs a new edge vector.
    src_nodes = dy.select_cols(nodes, src_adj)
    trg_nodes = dy.select_cols(nodes, trg_adj)
    input_expr = dy.concatenate([edges, src_nodes, trg_nodes], d=0)
    ret = dy.affine_transform([edge_b, edge_W, input_expr])
    return self.activation(ret)

  def node_edge_aggregate(self,
                          nodes,
                          edges,
                          src_adj,
                          trg_adj) -> 'expression_seqs.ExpressionSequence':
    """
    Aggregate edge hidden vectors per node
    """
    nodes = dy.transpose(nodes)
    
    # For each *node*, get the edges that point to it.
    # We probs need a for loop here because number of edges per node can vary.
    # Vectorise this will prove tricky...

    node_aggs = []
    for i, node in enumerate(nodes):
      # TODO: consider source nodes as well
      edge_indices = np.argwhere(trg_adj == i)
      node_agg = dy.sum_dim(dy.select_cols(edges, edge_indices), [1])
      node_aggs.append(node_agg)
    return expression_seqs.ExpressionSequence(node_aggs)
  
  def node_update(self,
                  nodes,
                  node_aggs) -> 'expression_seqs.ExpressionSequence':
    """
    Update node hidden vectors
    """
    node_aggs = node_aggs.as_tensor()
    node_W = dy.parameter(self.node_W)
    node_b = dy.parameter(self.node_b)

    # For each *node*, get it corresponding aggregated vector,
    # concatenate both and pass through an MLP
    input_expr = dy.concatenate([nodes, node_aggs], d=0)
    ret = dy.affine_transform([node_b, node_W, input_expr])
    return self.activation(ret)

  def get_final_states(self):
    """
    This just returns the first node as a dummy final state. A Graph Encoder
    should *always* be used with a "NoBridge".
    TODO: enforce this constraint in code
    TODO: add some flexibility here? Sentences still have a final word.
    """
    return [transducers.FinalTransducerState(main_expr=self.nodes_ret[0])]

