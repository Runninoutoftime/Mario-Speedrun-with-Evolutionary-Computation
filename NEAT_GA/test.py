import pickle
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

node_attrs = {
    'shape': 'circle',
    'fontsize': '9',
    'height': '0.2',
    'width': '0.2'}

dot = graphviz.Digraph(format='svg', node_attr=node_attrs)

dot.render("Digraph.gv.svg", view=True)