# -*- coding: utf-8 -*-
"""
OOBN reader
"""
import numpy as np
import pandas as pd
import lark

import pybn
from pybn.util import sep

GRAMMAR = r"""
    oobn_class: "class" name properties [comment]
    name: CNAME

    properties: "{" property_type* "}"

    ?property_type: property
                  | class_property

    property: name "=" value ";" [comment]
    ?class_property: node | potential

    node: "node" name properties
    potential: "potential" "(" name ["|" parents] ")" properties
    parents: name*

    ?value: string 
          | number 
          | tuple

    string: ESCAPED_STRING
    number: SIGNED_NUMBER
    tuple: "(" value* (comment value)* ")" [comment]

    comment: "%" /.+/ NEWLINE

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.NEWLINE
    %import common.WS
    %ignore WS
"""

class BasicTransformer(lark.Transformer):
    """Transform lark.Tree into basic, Python native, objects."""
    
    def oobn_class(self, items):
        """The oobn_class is the root element of an OOBN file."""
        name, properties, comment = items
        #for idx, i in enumerate(items):
        #    print(repr(i)[:25])
        
        oobn_obj = {
            'name': name
        }

        oobn_obj.update(properties)

        return oobn_obj

    def name(self, tokens):
        """A name is essentially an unquoted string."""
        # tokens: list of Token; Token is a subclass of string        
        return str(tokens[0])
    
    def properties(self, items):
        """
        Properties of the root element or one of its contained subclassess.
        """
        properties = []
        nodes = {}
        potentials = {}
        
        for i in items:
            if isinstance(i, tuple):
                properties.append(i)
            
            elif i['type'] == 'DiscreteNetworkNode':
                nodes[i['name']] = i

            elif i['type'] == 'CPT':
                # i['properties']['parents'] = i['parents']
                potentials[i['name']] = i

        if nodes:
            properties.append(('nodes', nodes))

        if potentials:
            properties.append(('potentials', potentials))
                
        return dict(properties)
            
    def node(self, items):
        """Subclass of the root element.

        Contains a list of states for a particular node.
        """
        name, properties = items
        node = {
            'type': 'DiscreteNetworkNode',
            'name': name,
        }
        node.update(properties)

        return node
    
    def potential(self, items):
        """Subclass of the root element.

        Contains a node's
         * the prior/conditional probability table (CPT)
         * experience table
        """

        parents = []
        
        if len(items) == 2:
            name, properties = items            
        elif len(items) == 3:
            name, parents, properties = items
        
        potential = {
            'type': 'CPT',
            'name': name,
            'parents': parents,
        }

        potential.update(properties)

        return potential
                
    def property(self, items):
        """A property consists of a simple key, value pair."""
        name, value = items[:2]
        return (name, value)
    
    def parents(self, items):
        """Represents the parents of a node in the DAG."""
        return list(items)
    
    def string(self, items):
        """Quoted string."""
        # Remove the quotes ...
        return str(items[0][1:-1])

    def number(self, items):
        """Number."""
        return float(items[0])
    
    def tuple(self, items):
        """Tuple."""
        return [i for i in items if i is not None]
    
    def comment(self, items):
        """Comment. Ignored."""
        return None


def _parse(filename):
    with open(filename, 'r') as fp:
        oobn_data = fp.read()

    parser = lark.Lark(GRAMMAR, parser='lalr', start='oobn_class')
    tree = parser.parse(oobn_data)
    
    return tree

def _transform(tree):
    transformer = BasicTransformer()
    return transformer.transform(tree)

def _create_structure(tree):
    # dict, indexed by node name
    nodes = {}

    # list of tuples
    edges = []

    # Iterate over the list of nodes
    for name in tree['nodes']:
        node = tree['nodes'][name]
        potential = tree['potentials'][name]
        node_parents = potential['parents']

        for parent in node_parents:
            edges.append((parent, name))

        node_states = node['states']
        parent_states = {}

        # print('-' * 80)
        # print(f'processing: {name}')        
        # print(f'  states: {node_states}')
        # print(f'  parents: {node_parents}')

        for parent in node_parents:
            parent_states[parent] = tree['nodes'][parent]['states']

        variable_states = {
            name: node_states
        }

        variable_states.update(parent_states)

        # Get the data for the prior/conditional probability distribution.
        data = potential['data']
        data = np.array(data)

        # If there are parents, it's a CPT
        if len(node_parents) > 0:
            # Prepare the indces for the dataframe
            index = pd.MultiIndex.from_product(
                parent_states.values(),
                names=parent_states.keys()
            )
            columns = pd.Index(node_states, name=name)
            data = data.reshape(-1, len(columns))
            df = pd.DataFrame(data, index=index, columns=columns)

            # print('-' * 80)
            # print(name)
            # print('-' * 80)
            # print(df.head())
            # print()
            # print(df.stack().head())
            # print()

            cpt = pybn.CPT(
                df.stack(),
                conditioned_variables=[name],
                # variable_states=variable_states
            )

            # print(pybn.CPT._index_from_variable_states(variable_states))
            # print(df.variable_states)

        # Else, it's a probability table
        else:
            # index = pd.Index(node_states, name=name)
            # columns = pd.Index([''])
            # df = pd.DataFrame(data, index=index, columns=columns)
            # df = df.transpose()
            cpt = pybn.CPT(
                data,
                conditioned_variables=[name],
                variable_states=variable_states
            )

        # Get the data for the dataframe
        nodes[name] = {
            'type': node['type'],
            'RV': name,
            'name': name,
            'states': node_states,
            'parents': node_parents,
            'CPT': cpt
        }

    network = {
        'name': tree['name'],
        'nodes': nodes,
        'edges': edges,
    }

    return network

def _create_bn(structure):
    """Create a BayesianNetwork from a previously created structure."""
    nodes = []

    for name, node_properties in structure['nodes'].items():
        RV = node_properties['RV']
        states = node_properties['states']
        description = ''

        cpt = node_properties['CPT']
        
        if None in cpt.index.names:
            cpt.index = cpt.index.droplevel()

        constructor = getattr(pybn, node_properties['type'])

        n = constructor(RV, name, states, description, cpt)
        nodes.append(n)

    edges = structure['edges']

    # for name, node_properties in structure['nodes'].items():
    #     for parent in node_properties['parents']:
    #         nodes[parent].add_child(nodes[name])

    return pybn.BayesianNetwork(structure['name'], nodes, edges)

def read(filename):
    """Parse the OOBN file and transform it into a sensible dictionary."""
    # Parse the OOBN file
    parsed = _parse(filename)

    # Transform the parsed tree into native objects
    transformed = _transform(parsed)

    # The transformed tree still has separate keys for nodes' states and their
    # parents and PDs/CPDs. Let's merge the structure to have all relevant data
    # together and let's transform the PDs/CPDs into Pandas DataFrames.
    structure = _create_structure(transformed)

    return _create_bn(structure)


