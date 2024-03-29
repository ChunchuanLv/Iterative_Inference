import torch
from torch.nn import Dropout, Linear
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List,Dict

def wrap_f(f,name = 0):

    def wrap(f,x,name):
        out = {}
        out[0] = f(x)
        out[name] = out[0]
        return out

    return lambda x: wrap(f,x,name)

class ComputationNode(object):

    def __init__(self,output_name:str,input_names:List[str],forward ,backward,extra_input_names:List[str]=[],requires_grad = True):

        self.output_name = output_name
        self.input_names = input_names
        self.extra_input_names = extra_input_names
        self.forward_f = forward
        self.backward_f = backward
        self.requires_grad = requires_grad
        self.states = None

    def forward(self,input_dict):

        if self.states: return self.states[0]
        inputs = [input_dict[name] for name in self.input_names] + [input_dict[name] for name in self.extra_input_names]
  #      print ("forwarding\n",self.output_name)
     #   for name in self.input_names: print (self.output_name,name,input_dict[name].size())
        all_output  = self.forward_f(*inputs)

        if not isinstance(all_output,List) : all_output = [all_output]

        self.states = all_output + inputs

        return all_output[0]

    def backward(self,grad_output:torch.Tensor):
        if not self.requires_grad: return {}

        assert self.states is not None
     #   print ("backwarding ",self.output_name,self.states)
    #    print ("grad_output ",grad_output.size())
    #    for s in self.states: print (s.size())
        grad_inputs = self.backward_f(grad_output, *self.states)

        if isinstance(grad_inputs,List) is False: grad_inputs = [grad_inputs]

        assert len(grad_inputs) == len(self.input_names)


        grad_dict = {}

        # allowing same duplicates names in input
        for name, grad in zip(self.input_names,grad_inputs ):
            if name in grad_dict:
                grad_dict[name] = grad_dict[name]+ grad
            else:
                grad_dict[name] =  grad
        return grad_dict

    def __hash__(self):
        return hash(self.output_name)

    def __str__(self):
        return self.output_name


    def __repr__(self):

        return self.output_name +"  :=  "+str(self.forward)+" ( "+" , ".join(self.input_names)+" )"

class ComputationGraph:


    def __init__(self,output_names:[str],input_names:List[str],extra_input_names:List[str]=[]):
        if isinstance(output_names,str): output_names = [output_names]
        self.output_names = output_names
        self.input_names = input_names
        self.extra_input_names = extra_input_names
        self.G = nx.DiGraph()
        for name in input_names+extra_input_names:
            self.G.add_node(name)

    def add_node(self,node:ComputationNode):

        self.G.add_node(node.output_name,data=node)
        for name in node.input_names:
            self.G.add_edge(name,node.output_name)
        for name in node.extra_input_names:
            self.G.add_edge(name,node.output_name)


    def forward(self, inputs: List[torch.Tensor],extra_inputs:List[torch.Tensor] = []):

        assert len(inputs) ==len(self.input_names)


        execution_order = self.get_forward_order()
        values = {**dict(zip(self.input_names,inputs )), **dict(zip(self.extra_input_names,extra_inputs ))}
        for node_name in execution_order:


            node = self.G.nodes[node_name]["data"]

            values[node_name] = node.forward(values)
        if len(self.output_names) == 1:
            return values[self.output_names[0]]
        return [values[name] for name in self.output_names]


    def get_tensor_by_name(self,node_name):

        node = self.G.nodes[node_name]["data"]

        return node.states[0]


    def backward(self, grads = None):



        execution_order = self.get_backward_order()

        if grads is None:
            assert len(self.output_names) == 1
            total_grads_dict  = { self.output_names[0]:1 }
        elif isinstance(grads,torch.Tensor):

            total_grads_dict  = { self.output_names[0]:grads }
        else:
            total_grads_dict = {}
            for i,grad in enumerate(grads):
                total_grads_dict[self.output_names[i]] = grad

        for node_name in execution_order:

            node = self.G.nodes[node_name]["data"]
            if node_name not in total_grads_dict: continue
            grads_dict = node.backward(total_grads_dict[node_name])
            for name, grad in grads_dict.items():
                if name in total_grads_dict:
                    total_grads_dict[name] = total_grads_dict[name]+ grad
                else:
                    total_grads_dict[name] =  grad

        return [ total_grads_dict[name]  for name in self.input_names]

    def get_forward_order(self)->List[ComputationNode]:

        return [str(x) for x in nx.topological_sort(self.G) if str(x)  not in self.input_names + self.extra_input_names ]

    def get_backward_order(self)->List[ComputationNode]:

        return list(reversed(self.get_forward_order()))

def main():
    '''
    node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 score_dim: int,

    :return:
    '''

    f =  nn.functional.softplus
    b = lambda grad, output, input: nn.functional.sigmoid(input) * grad

    g = ComputationGraph("score",["x"],["y"])

    m1 = ComputationNode("m1",["x"],f,b )

    m2 = ComputationNode("m2",["m1","m1"],lambda x,y: x + y,  lambda grad, output, input, input_:  [grad,grad])

    m3 = ComputationNode("score",["m2","x","y"], lambda m2,x,y: (m2+x+y).sum(),   lambda grad, output, m2, x,y: [grad*torch.ones_like(m2),grad*torch.ones_like(x),0])

    g.add_node(m1)
    g.add_node(m2)
    g.add_node(m3)

    print (g.G.nodes)
    print (g.G.edges)

    x = torch.rand(5,5,requires_grad=True)

    y = torch.rand(5,5,requires_grad=True)

    score = g.forward([x],[y])

    x_grad = g.backward()[0]

    score.backward()

    print ("x.grad",x.grad)
    print ("x_grad",x_grad)

    print ("diff",(x_grad-x.grad).pow(2).sum())

if __name__ == "__main__":
    main()
