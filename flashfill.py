
import re
import logging
import heapq


from pathlib import Path
from itertools import product
from collections import namedtuple

class Token:
    def __init__(self, name, pattern):
        self.name = name
        self.pattern = pattern


# Define a dictionary of Token objects with their corresponding patterns
tok_list = {
 
    # Alphanumeric token 
    "AlphNumTok": Token("AlphNumTok", "[a-zA-Z0-9]+" ),

    # Alphanumeric with spaces token
    "AlphNumWsTok": Token( "AlphNumWsTok","[a-zA-Z0-9 ]+" ),
    
    # Alphabetic token
    "AlphTok": Token( "[a-zA-Z]+" ,"AlphTok"),

    # Backslash token 
    "BackSlashTok": Token("\\\\" , "BackSlashTok"),

    # End token
    "EndTok": Token( "\$" , "EndTok"),

    # Hyphen token
    "HyphenTok": Token( "-", "HyphenTok"),

    # Lowercase token
    "LowerTok": Token( "LowerTok" ,"[a-z]+" ),

    # Colon token
    "ColonTok": Token(":" , "ColonTok"),

    # Comma token 
    "CommaTok": Token( "," , "CommaTok"),
    
    # Dash token
    "DashTok": Token( "-" , "DashTok"),

    # Dot token 
    "DotTok": Token(  "\\.","DotTok"),

    # # Non-alphanumeric token 
    # "NonAlphNumTok": Token( "[^A-Z0-9]+" , "NonAlphNumTok"),

     # Non-alphanumeric token 
    "NonAlphNumTok": Token( "[^a-zA-Z0-9]+" , "NonAlphNumTok"),

    # Non-alphanumeric with spaces token
    "NonAlphNumWsTok": Token( "[^a-zA-Z0-9 ]+" , "NonAlphNumWsTok"),

    # Non-alphabetic token 
    "NonAlphTok": Token("NonAlphTok", "[^a-zA-Z]+"),

    # Non-lowercase token 
    "NonLowerTok": Token("NonLowerTok", "[^a-z]+"),

    # Non-numeric token
    "NonNumTok": Token("NonNumTok", "[^\\d]+"),

    # Non-uppercase token
    "NonUpperTok": Token("NonUpperTok", "[^A-Z]+"),

    # Numeric token
    "NumTok": Token("NumTok", "\\d+"),

    # Right angle bracket token
    "RightAngleTok": Token("RightAngleTok", ">"),

    # Right parenthesis token 
    "RightParenTok": Token( "\\)" , "RightParenTok"),

    # Right square bracket token 
    "RightSquareTok": Token("RightSquareTok", ">"),
    
    # Left angle bracket token 
    "LeftAngleTok": Token("LeftAngleTok", "<"),

    # Left parenthesis token 
    "LeftParenTok": Token("LeftParenTok", "\\("),

    # Left square bracket token
    "LeftSquareTok": Token("LeftSquareTok", "<"),

    # Underscore token 
    "UnderscoreTok": Token("UnderscoreTok", "_"),

    # Semicolon token 
    "SemicolonTok": Token("SemicolonTok", ";"),

    # Slash token
    "SlashTok": Token("SlashTok", "\\/"),

    # Start token 
    "StartTok": Token( "^" , "StartTok"),

    # Uppercase token
    "UpperTok": Token( "[A-Z]+" , "UpperTok"),

    # Whitespace token
    "WsTok": Token("WsTok", " "), 
}

if not tok_list:
      raise ValueError('Token list is  empty') 

class Token:
    def __init__(self, name, pattern):
        self.name = name
        self.pattern = pattern

# Class representing a Node in the Directed Acyclic Graph (DAG)
class Node:
    def __init__(self, node_id):
        self.id = node_id


class Edge:
    def __init__(self, src, dst, tok_list, position):
        self.src = src
        self.dst = dst
        self.tok_list = tok_list
        self.position = position  


class ProgramError(Exception):
    pass

# Class representing a mapping of expression patterns between edges
class Map:
    def __init__(self):
        self.data = {}

class intersect:
    def __init__ (self,d1, d2):
  
        result = set(d1.keys()).intersection(d2.keys())
        return result if result else None
    
class keys:
    def keys(dict, order):
        keys = dict.keys()  
        
        # Sort keys
        if order == "asc":
            keys = sorted(keys)  
        elif order == "desc":
            keys = sorted(keys, reverse=True)
      
        return keys

class Unify:
    def __init__(self):
        self.substitution = {}

    # def unify_var(self, var, x):
    #     if var in self.substitution:
    #         return self.unify(self.substitution[var], x)
    #     elif x in self.substitution:
    #         return self.unify(var, self.substitution[x])
    #     elif self.occur_check(var, x):
    #         return None
    #     else:
    #         self.substitution[var] = x
    #         return self.substitution

    def unify(self, d1, d2):
        if d1 == d2:
            return self.substitution
        elif isinstance(d1, str) and d1.isalpha():
            return self.unify_var(d1, d2)
        elif isinstance(d2, str) and d2.isalpha():
            return self.unify_var(d2, d1)
        elif isinstance(d1, list) and isinstance(d2, list) and len(d1) == len(d2):
            for d1i, d2i in zip(d1, d2):
                result = self.unify(d1i, d2i)
                if result is None:
                    return None
            return result
        else:
            return None

    def occur_check(self, var, x):
        if var == x:
            return True
        elif isinstance(x, list):
            return any(self.occur_check(var, xi) for xi in x)
        else:
            return False


   
class TokenInfo:
    def __init__(self, name, type):

        self.name = name
        self.type = type  
      
def prod(*iterables):

    if not iterables:
        return [()]
    else:
        result = []
        for item in iterables[0]:
            for rest in prod(*iterables[1:]):

                result.append((item,) + rest)
        return result


def validate_inputs(input_str):
   if not input_str: 
      raise ValueError('Invalid input')
   

# Class representing a Directed Acyclic Graph (DAG) with methods for finding paths, intersection, and unification

class DAG:
    def __init__(self, input_str, output_str, tok_list ):
     ##Node creation
        self.nodes = [Node(i) for i in range(len(input_str) + 1)]
       
        self.edges = []
        for i in range(len(input_str)):
            for j in range(i + 1, len(input_str) + 1):
                edge_tok_list = []
                New_tok_list = []
                ##Append the matching tokens in to tok list for future use
                for token_name, token in tok_list.items():
                    edge_tok_list.append(token.pattern)

                edge_position = (i, j - 1)  # Adjust position calculation
                edge = Edge(self.nodes[i], self.nodes[j], edge_tok_list, edge_position)
                self.edges.append(edge)


    # def __init__(self, input_str, output_str, tok_list ):
   
    #     self.nodes = [Node(i) for i in range(len(input_str) + 1)]
       
    #     self.edges = []
    #     for i in range(len(input_str)):
    #         for j in range(i , len(input_str) ):
    #             print("i")
    #             edge_tok_list = [tok_list[token_name].pattern for token_name in tok_list]
    #             edge = Edge(self.nodes[i], self.nodes[j], edge_tok_list, edge_position)
    #             self.edges.append(edge)


    def construct_edge(edge,nodes):
        edge = Edge()
        
        for i in range(1, len(nodes)):
            edge.set_source(nodes[i])
            edge.set_destination(nodes[j]) 

            #token_matches = match_token_patterns(input_str, tok_list)
            
            edge.append(edge)

    def get_edge(self, edge):

        src, dst = edge
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                return edge
        return None

    def find_path(self, src, dst):
        if src and dst is not None:

            if src == dst:
                return [src]

            for edge in self.edges:
                if edge.src == src:
                    remaining = self.find_path(edge.dst, dst)
                    if remaining:
                        return [src] + remaining

            return None
        

        else:
            return None
    def is_valid_dag(dag):
    # Check for acyclicity using depth-first search (DFS)
        visited = set()
        stack = set()

        def is_cyclic(node):

            visited.add(node)
            stack.add(node)

            for neighbor in dag.edges.get(node, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in stack:
                    return True

            stack.remove(node)
            return False

        for node in dag.nodes:
            if node not in visited:
                if is_cyclic(node):
                    return False  # Graph is cyclic

        # Check for connectivity
        for node in dag.nodes:
            if not any(node in neighbors for neighbors in dag.edges.values()):
                return False  # Node is not connected

        # Check for unique edges
        all_edges = [edge for neighbors in dag.edges.values() for edge in neighbors]
        if len(all_edges) != len(set(all_edges)):
            return False  # Duplicate edges found

        # Check for single source
        sources = dag.nodes - set(neighbor for neighbors in dag.edges.values() for neighbor in neighbors)
        if len(sources) != 1:
            return False  # Multiple sources found

        return True

    def intersect1(d1, d2):
  
        result = set(d1.keys()).intersection(d2.keys())
        return result if result else None

    def intersect(self, d1, d2):

        if d1 and d2 is not None:
            nodes = set()
            source = [d1.source, d2.source]
            target = [d1.target, d2.target]
            edges = set()
            mapping = Map()

            for ((d1_source, d1_target), (d2_source, d2_target)) in product(keys(d1.W), keys(d2.W)):
                
                result = intersect({d1_source: d1_target}, {d2_source: d2_target})

                if not result is not None:
                    n1, n2 = [d1_source , d2_source], [d1_target , d2_target]
                    nodes.add(n1)
                    nodes.add(n2)
                    edges.add((n1, n2))

                    mapping[n1 , n2] = result

            return DAG(nodes, source, target, edges, mapping)
        else:
            return 0

    def unify(self, d1, d2):
        nodes = set()
        source = [d1.source, d2.source]
        target = [d1.target, d2.target]
        edges = set()
        mapping = Map()
        if d1_source and d2_source is not None:
            if  d1_target and d2_target is not None:
                for ((d1_source, d1_target), (d2_source, d2_target)) in product(keys(d1.W), keys(d2.W)):
                    result = unify({d1_source: d1_target}, {d2_source: d2_target})
                    print("Unify")
                    if result is not None:
                        n1, n2 = [d1_source, d2_source], [d1_target, d2_target]
                        nodes.add(n1)
                        nodes.add(n2)
                        edges.add((n1, n2))
                        mapping[n1, n2] = result

                return DAG(nodes, source, target, edges, mapping) 


def match_token_patterns(input_str, token_list):
   matches = []  
   
   for token in token_list:
      pattern = re.compile(token.pattern)   
      current_matches = pattern.findall(input_str)
      
      matches.append((token.name, current_matches))
      
   return matches

def check_token_list(prog_list):

    ####check if the program list is empty
    if not prog_list:
        print("Token list is empty. No program to generate.")
        return False
    ##check if program list has programs
    else:

        print("Token list is not empty. Generating program.")
        return True

def print_DAG(dag, token_matches):
    print("\nEdges:")
    for edge in dag.edges:
        position = edge.position
        print(f" {edge.src.id} -> {edge.dst.id}, tok_list: {edge.tok_list}, Position: {position}")

        for token_pattern in edge.tok_list:
            
            token_name = [name for name, token in tok_list.items() if token.pattern == token_pattern][0]

            if token_name in token_matches and position in token_matches[token_name]:
                matches = token_matches[token_name][position]

                #if matches is not None:
                print("token name")
                print(f"   {token_name}: {matches}")

        # for token_pattern in edge.tok_list:
        #     matching_tokens = [name for name, token in tok_list.items() if token.pattern == token_pattern]

        #     if matching_tokens:
        #         token_name = matching_tokens[0]
                
        #         if token_name in token_matches and position in token_matches[token_name]:
        #             matches = token_matches[token_name][position]
        #             print("Token Name:")
        #             print(f"   {token_name}: {matches}")
    
  
    print("DAGs are created succesfully")


###Function to generate program from dags
def get_program(input_str, output_str, tok_list):
    dag = DAG(input_str, output_str, tok_list)

    # Get path
    path = dag.find_path(dag.nodes[0], dag.nodes[-1])
    if not path:
        return "No program found"

    # Build token matches dynamically
    token_matches = {}
    new_tok_list = []
    for i, node_id in enumerate(path):
      
        position_start = i
        position_end = i + len(path) - 1
        position = (position_start, position_end)

        for token_name, token in tok_list.items():
            pattern = re.compile(token.pattern)
            substring = input_str[node_id.id:node_id.id + 1]
            #Find matches
            matches = [match.group(0) for match in pattern.finditer(substring)]

            if token_name not in token_matches:
                new_tok_list.append(token_name)
                token_matches[token_name] = {}

            token_matches[token_name][position] = matches
    
    prog_list = []


    # Build program
    #program = "Concatenate(\n"
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        dag.get_edge(edge).position = (i, i + len(path) - 1)  # Update edge position dynamically
        tok_list = dag.get_edge(edge).tok_list
        program = f"   {tok_list},\n"
        
        prog_list.append(program)
    prog_check = check_token_list(prog_list)
    
    return prog_list, token_matches
# Program class 
class Program:
    
    def __init__(self):
        self.statements = []
        
    def add_statement(self, statement):
        self.statements.append(statement)
        
# Function to check empty   
def check_empty(program):
    
    # Validate input
    if not isinstance(program, Program):
        raise TypeError('Invalid program object')
        
    # Check length   
    if len(program.statements) == 0:
        print('Program is empty')
        return True
    
    print('Program has statements')
    return False

    prog = Program()
    prog.add_statement('load_input()')
    prog.add_statement('tokenize(input)')

    is_empty = check_empty(prog)
    print(is_empty)

    # Add statement
    prog.add_statement('output = transform(tokens)')  

    is_empty = check_empty(prog)
    print(is_empty)


#give rank to the program
def ranking_prog(prog_list):
    if prog_list is not None:
        for i in prog_list:
            if any(tok in i for tok in ["StartTok", "EndTok"]):
                prog1 = i
                exit
    return prog1



def execute_program(input_str, program):
   
    result = input_str
    for token_pattern in program:
     
        result += f" {token_pattern}"

    return result

def check_program(input_str, program, expected_output):
    # Execute the program on the input string
    result = execute_program(input_str, program)

    # Compare the result with the expected output
    if result == expected_output:
        print("Verification Passed: Program produced the expected output.")
    else:
        print("Verification Failed: Program did not produce the expected output.")

        print(f"Expected Output: {expected_output}")
        print(f"Actual Output: {result}")


##Main function fo the program.

def main():
    #Iput-Output examples
    input1 = "BTR KRNL WK CORN 15Z"
    output1 = "15Z"

    input2 = "CAMP DRY DBL NDL 3.6 OZ"
    output2 = "3.6 OZ"

    input3 = "KRNL HAT KEA UDA 3.8 OZ"
    output3 = " "

    dag_list = []
    prog_list = []

    for input_str, output_str in [(input1, output1), (input2, output2)]:
        if not input_str:
            raise ValueError('Invalid input')
        if not output_str:
            raise ValueError('Invalid output')

        print(f"Input: {input_str}")
        print(f"Output: {output_str}")

        prog_list, token_matches = get_program(input_str, output_str, tok_list)

        dag = DAG(input_str, output_str, tok_list)
        dag_list.append(dag)
        print_DAG(dag, token_matches)

    Final_prog = ranking_prog(prog_list)

    if Final_prog is not None:
        Final_prog = prog_list[0]

    print("Program::", "Substr(", input3, ",", Final_prog, ")")


if __name__ == "__main__":
    main()

