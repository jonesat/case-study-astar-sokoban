
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2022-03-27  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban
import math
import copy
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (1, 'Me', 'Coder') ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_suite():
    """
    Inputs: None
    Outputs: None
    This function tests specific situations that can cause issues with the sokoban solver
    """
    test_results = []
    test_warehouse = sokoban.Warehouse()
    number_of_tests=7
    x = sokoban.Warehouse()
    ################### Test 1 #####################
    message=" This is a string representation of a warehouse where the worker cannot move, it should fail"
    # The warehouse looks like the following
    # ###
    # #@#
    # ###
    test_warehouse_1 = "###\n#@#\n###"
    x.from_string(test_warehouse_1)
    s = ["Down"]
    out = check_elem_action_seq(x, s)
    Expected_Answer = "Impossible"

    test1_flag = False
    if out=="Impossible":
    	test1_flag = True	
    else:
        print(f"The expected answer was: {Expected_Answer}")
    test_results.append(test1_flag)
    print(f"{message}\ntest 1 passed? {test1_flag}")
    ################################################	


    #################### Test 2 ####################
    message= "box to box test - This should cause two boxes to collide and fail."
    ######
    # .  #
    #    #
    #*@  #
    #  $ #
    #    #
    ######
    test_warehouse_2 = "###### \n# .  #\n #    #\n #*@  #\n #  $ #\n #    #\n ######"
    y = sokoban.Warehouse()
    y.from_string(test_warehouse_2)
    s = ["Down","Down","Right","Up","Right","Up","Left","Left"]
    out = check_elem_action_seq(y, s)
    Expected_Answer = "Impossible"

    test2_flag = False
    if out==Expected_Answer:
        test2_flag = True	
    else:
        print(f"The expected answer was: {Expected_Answer}")
    test_results.append(test2_flag)
    print(f"{message}\ntest 2 passed? {test2_flag}")
    ################################################

    #################### Test 3 ####################
    message="box to wall test - this test cause a box to collide with a wall and should fail"

    y.from_string(test_warehouse_2)
    s=["Down","Right","Right"]
    out = check_elem_action_seq(y, s)
    Expected_Answer = "Impossible"

    test3_flag = False
    if out==Expected_Answer:
        test3_flag = True	
    else:
        print(f"The expected answer was: {Expected_Answer}")
    test_results.append(test3_flag)
    print(f"{message}\ntest 3 passed? {test3_flag}")
    ################################################

    #################### Test 4 ####################
    message = "box to wall with goal test - This test attempt to push a box against a wall it shouldn't fail as there is a target along that wall therefore there is no deadlock."

    y.from_string(test_warehouse_2)
    s = ["Down","Down","Right","Up","Up","Up"]

    out = check_elem_action_seq(y, s)
    Expected_Answer="Impossible"

    test4_flag = False
    if out!=Expected_Answer:
        test4_flag = True	
    else:
        print(f"The expected answer was not {Expected_Answer} but a warehouse")
    test_results.append(test4_flag)
    print(f"{message}\ntest 4 passed? {test4_flag}")
    ################################################

    #################### Test 5 ####################
    message = "box to wall with goal test - This test attempt to push a box against a wall that doesn't have a target along it. There will be no way to recover as the box is deadlocked. It should fail."
    ######
    # .  #
    #    #
    #*@  #
    #  $ #
    #    #
    ######
    y.from_string(test_warehouse_2)
    s=["Right","Down"]
    out = check_elem_action_seq(y, s)
    Expected_Answer = "Impossible"
    print(out)
    test5_flag = False
    if out==Expected_Answer:
        test5_flag = True
    else:
        print(f"The expected answer was: {Expected_Answer}")
    test_results.append(test5_flag)
    print(f"{message}\ntest 5 passed? {test5_flag}")
    ################################################

    #################### Test 6 ####################
    message="These actions push a box to a corner with no target resulting in an "'Impossible'" action sequence."
    ######
    # .  #
    #    #
    #*@  #
    #  $ #
    #    #
    ######
    test_warehouse_3 = "######\n # .  #\n #    #\n #*@  #\n #  $ #\n #    #\n ######"
    z = sokoban.Warehouse()
    z.from_string(test_warehouse_3)
    s = ["Down","Right","Up","Right","Down","Down"]
    out = check_elem_action_seq(z, s)
    Expected_Answer = "Impossible"
    
    test6_flag = False
    if out==Expected_Answer:
        test6_flag = True	
    else:
        print(f"The expected answer was: {Expected_Answer} - This one technically passes as it is being prevented from moving into the corner")
    test_results.append(test6_flag)
    print(f"{message}\ntest 6 passed? {test6_flag}")
    ################################################

    #################### Test 7 ####################
    message="These actions push a box in to a corner with a target which is a legal sequence of actions."

    z.from_string(test_warehouse_3)
    s = ["Down","Left","Down","Right","Right"]
    out = check_elem_action_seq(z, s)
    Expected_Answer = "Impossible"

    test7_flag = False
    if out!=Expected_Answer:
        test7_flag = True	
    else:
        print(f"The expected answer was not {Expected_Answer} but a warehouse")
    test_results.append(test7_flag)
    print(f"{message}\ntest 7 passed? {test7_flag}")

    ################################################

    ################# Final Results ################

    print(test_results)
    if len(test_results)==number_of_tests and all(test_results):
        print(f"This test suite passed all {number_of_tests} tests")
        return True
    else: 
        print(f"This test suite failed on {['Test '+str(i+1) for i,v in enumerate(test_results) if not v]}")
        return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    def __init__(self, *,warehouse,goal=None):
       
        self.walls = warehouse.walls        
        self.initial = (warehouse.worker,tuple(warehouse.boxes))
        self.goal = warehouse.targets
        self.weights = warehouse.weights
        self.metric_types = ["Euclidean","Manhattan"]
        self.unit_direction = {'Up':(0,-1),'Down':(0,1),'Right':(1,0),'Left':(-1,0)}
    
    
    ## Auxiliary Functions
    def Add_tuple(self,tuple1:tuple,tuple2:tuple):
        '''
        Add tuples together as though they were vectors
        inputs two tuples
        returns a new tuple
        '''
        new_tuple = tuple(map(sum, zip(tuple1,tuple2)))
        return new_tuple
    
    def Check_if_A_in_B(self,A,B):
        """
        Check to see if element A falls in the iterable B
        """
        # print(f"Check if {A} is in {B}")
        if A in B:
            return True
        else:
            return False
            
    def Check_if_stuck_on_edge(self,new_box_position,boxes):
        """
        This functions takes in the new position of a box as a tuple 
        It also takes in a list of other box tuples.
        This function attempts to determine if the new position of the box will:
        1. be a corner, trapping the box
        2. be an edge with no way to move off of the edge
        If either of these two scenarios could occur by moving into the input box position this function returns true.
        """
        x_edge = self.Check_if_at_edge(new_box_position,1,self.walls+boxes)
        y_edge = self.Check_if_at_edge(new_box_position,0,self.walls+boxes)
        goal_neighbour_x = self.Check_if_at_edge(new_box_position,0,self.goal)
        goal_neighbour_y = self.Check_if_at_edge(new_box_position,1,self.goal)
        
        if x_edge and y_edge and new_box_position not in self.goal:
            # print("Box wants to move into a corner")
            return True
        elif x_edge and not y_edge and new_box_position not in self.goal:
            # print("Box wants to move into deadlocked column - look up and down")
            return self.Check_if_aisle_deadlocked(0,new_box_position,self.walls+boxes)
        elif y_edge and not x_edge and not self.goal:
            # print("Box wants to move into deadlocked row - look left and right")
            return self.Check_if_aisle_deadlocked(1,new_box_position,self.walls+boxes)
        else:
            return False
                    
    def Check_if_aisle_deadlocked(self,axis:int,new_box_position:tuple,obstacles:list):
        """
        Make sure that if the new box position is along an edge that it has a target to move towards.
        Inputs: the tuple representing the boxes new position
        Inputs: an integer representing the axis to check, 0 for along x-columns, 1 for along y-rows.
        Inputs: obstacles, the objects that could deadlock a box in the aisle
        """
        if axis==0:
            actions = list(self.unit_direction.keys())[2:]
        elif axis==1:
            actions = list(self.unit_direction.keys())[:2]
        # off_axis = {0:1,1:0}
        aisle_obstacles= {}
        
        for action in actions:
            looking = new_box_position
            aisle_obstacles[action]=True
            while looking not in obstacles:
                looking = self.Add_tuple(looking,self.unit_direction[action])
                if looking in self.goal:
                    # if there is a goal in the same aisle then we can safely move along the aisle
                    aisle_obstacles[action]=False
                    break
                elif not self.Check_if_at_edge(looking,axis,obstacles):
                    # If by continuing to move along the aisle we can move off an aisle we will.
                    # print(f"No longer on an edge at {looking}")
                    aisle_obstacles[action]=False
                    break                    
        if sum(aisle_obstacles.values())==2:
            # print("This box is deadlocked in axis {0}")
            return True
        else:
            return False
    
    def Check_if_at_edge(self,new_box_position,axis,neighbours):
        """
        Check to see if the new box position is along an edge.
        Inputs: the tuple representing the boxes new position
        Inputs: an integer representing the axis to check, 0 for along x-columns, 1 for along y-rows.
        Inputs: neighbours, the objects that constitute a boundary a game element is potentially colliding with
        """
        if axis==0:
            actions = list(self.unit_direction.keys())[2:]
        elif axis==1:
            actions = list(self.unit_direction.keys())[:2]
        edge_flag = any(self.Add_tuple(new_box_position,self.unit_direction[action]) in neighbours for action in actions) 
        return edge_flag            
        
    def Move_element(self,element,action):
        """
        This function uses a unit vector representing a direction to update the location of a game element, worker or box.
        It does this by adding the game elements position as a tuple to a tuple representing the unit vector corresponding to the direction they are moving in.
        This function returns the new position as tuple.
        """
        return self.Add_tuple(element,self.unit_direction[action])
        
    def Push_which_box(self,boxes,new_worker_position):
        """
        This function finds the box that a worker has collided with.
        It does this by making a generator containing any box, of which 
        there will be only one, that has the same position as the workers new
        position.
        The inputs are the an iterable containing boxes  and the workers new position as a tuple
        This function returns the first instance of boxes that occupy the space
        a worker is trying to move to.        
        """
        return next(box for box in boxes if box == new_worker_position)
    
    def movement_metric(self,type,x1,x2,weight):
        """
        This function takes in a box and a goal as tuples the weight as the cost of moving the box.
        This function calculates the number of squares between the two input tuples and multiplies them by (1+weight) representing the cost of moving the worker and box that many squares.
        Then the manhattan or euclidean distances is calculated and returned.
        """
        z = tuple(map(lambda x ,y: abs(x - y)*(1+weight), x1, x2))
        # assert type in self.metric_types, f"Invalid Metric Type please choose one of: {[chr(10)+type for type in self.metric_types]}"
        if type =="Euclidean":
            f = math.sqrt(sum(v**2 for v in z))
        if type =="Manhattan":
            f = sum(v for v in z)
        return f
        
    ## Mandatory Functions
        
    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        # print(f"goal_test goal: {self.goal}")
        # print(f"goal_test boxes: {state[1]}")
        
        if all(box in self.goal for box in state[1]):
            print(f"The following spots have boxes {[box for box in state[1] if box in self.goal]}")
            print(f"All boxes are in a spot")
            return True
        else:
            return False
    
    def actions(self, state):
        """
        Return the list of legal actions that can be executed in the given state        
        """
        worker = state[0]
        boxes = state[1]
        valid_moves =[]
        
        for direction in self.unit_direction.keys():
            new_worker_position = self.Move_element(worker,direction)
            worker_in_walls = self.Check_if_A_in_B(new_worker_position,self.walls)
            if not worker_in_walls:
                valid_moves.append(direction)                                        
        return valid_moves

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        """
        
        worker = state[0]
        boxes = list(state[1])
        
        assert action in self.actions(state), f"This action, {action} is not supposed to be available as it is not in the list of available actions"
        
        new_worker_position = self.Move_element(worker,action)
        worker_in_walls = self.Check_if_A_in_B(new_worker_position,self.walls)
        worker_in_box = self.Check_if_A_in_B(new_worker_position,boxes)
        if not worker_in_walls:
            if not worker_in_box:
                worker = new_worker_position
            else:
                pushed_box = self.Push_which_box(boxes,new_worker_position)
                pushed_box_index = boxes.index(pushed_box)
                new_box_position = self.Move_element(pushed_box,action)
                
                box_in_walls = self.Check_if_A_in_B(new_box_position,self.walls)
                
                leftover_boxes = copy.deepcopy(boxes)
                leftover_boxes.remove(pushed_box)
                # print(f"These are the leftover boxes {leftover_boxes}")
                box_in_boxes = self.Check_if_A_in_B(new_box_position,leftover_boxes)
                
                if not box_in_walls and not box_in_boxes:
                    if not self.Check_if_stuck_on_edge(new_box_position,leftover_boxes):                        
                        # print(f"The box can move into position {new_box_position} as it is not a corner or corner without a goal")
                        boxes[pushed_box_index]=new_box_position
                        worker = new_worker_position
        return (worker,tuple(boxes))
                
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        old_worker = state1[0]
        old_boxes = state1[1]
        new_worker = state2[0]
        new_boxes = state2[1]
        
                    
        if old_worker!=new_worker:            
            did_boxes_stay_inplace = [box1==box2 for box1,box2 in zip(old_boxes,new_boxes)]
            if False in did_boxes_stay_inplace:
                box_moved_flag = True
                moved_box_index = did_boxes_stay_inplace.index(False)
                c += self.movement_metric(self.metric_types[1], old_boxes[moved_box_index],new_boxes[moved_box_index],self.weights[moved_box_index])
                return c
            else:
                return c+1
        else:
            return c
        
    def h(self,n):
        '''
        The heuristic here will be the sum of the largest manhattan or euclidean distance for each box.
        The idea here is that to reach the goal state each box has to be moved, so we need the largest distance that all boxes need to travel to reach the goal.
    
        Current Strategy:
        1. Computer all manhattan distances in a matrix for all box-target pairs - have to do this to find the largest admissible heuristic
        2. Select the largest mahattan distance for a single box target pair
        3. For the box-target pair remove the row and column of metrics to prevent other boxes or targets interacting with the same goal
        4. Select the next largest metric and again reduce the matrix of metrics by removing all rows/columns that correspond to metrics involving the current largest box-target pair
        ''' 
        
        # double check that the state and the goal involve the same number of pancakes
        
        # gather relevant variables
        targets = copy.deepcopy(self.goal)
        boxes = list(n.state[1])
        assert len(boxes)>=len(self.goal), "You don't have enough boxes to solve this puzzle"
        weights = self.weights

        # Initialise empty matrix to calculate all manhattan (or euclidean) distances
        METRICS=np.empty([len(targets),len(boxes)])

        # Have the option to get either distance metric
        # metric_types = ["Euclidean","Manhattan"] # Added this to problem definition

        # loop over i and j of matrix and get a metric for each pair of boxes and targets
        for i in range(np.shape(METRICS)[0]):
            for j in range(np.shape(METRICS)[1]):
                movement_cost = self.movement_metric(self.metric_types[1],targets[i],boxes[j],weights[j])
                # print(f"The {types[1]} distance between target: {target_dict[i]} and box: {boxes_dict[j]} is {out}" )
                METRICS[i,j]=movement_cost

        # Initialise Sum    
        total = 0

        # While you still have targets uncovered 
        while len(boxes)>0:
            # print(METRICS)
            max_manhattan = np.amax(METRICS)
            total += max_manhattan #accumulate the max manhattan distance
            result = np.where(METRICS ==max_manhattan) # find the location of the maximum manhattan distance
            max_target,max_box=[i[0] for i in result] # assign the indices to easily accessed variables
            # print(f"The maximum manhattan distance is {max_manhattan} and occurs at i={max_target}, j={max_box} - the total manhattan distance is now {total}")
            
            # remove box-target pairs from list, the list should perhaps be a copy of inputs to the function so as to not mess with  problem.boxes, problem.goal
            del targets[max_target] 
            del boxes[max_box]
            # remove rows and columns that correspond to the box-target pair so they cannot be included in future manhattan distances (no double dipping to get higher maximums)
            METRICS=np.delete(METRICS,max_target,axis=0)
            METRICS=np.delete(METRICS,max_box,axis=1)
        return total

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    warehouse = warehouse.copy()
    problem = SokobanPuzzle(warehouse=warehouse)
    # worker = warehouse.worker
    # boxes = warehouse.boxes
    # print(action_seq)
    valid_action = True
    i = 0
    print(action_seq)
    print(warehouse.__str__())
    while valid_action and i < len(action_seq):
        print(warehouse.__str__())
        action = action_seq[i]
        i+=1        
        print(f"Check check_elem_action_seq - Action: {action}")
        new_worker_position = problem.Move_element(warehouse.worker,action)
        worker_in_walls = problem.Check_if_A_in_B(new_worker_position,warehouse.walls)
        worker_in_box = problem.Check_if_A_in_B(new_worker_position,warehouse.boxes)
        if not worker_in_walls:
            if not worker_in_box:
                print(f"The warehouse worker has moved from {warehouse.worker} to {new_worker_position}")
                warehouse.worker = new_worker_position
                # return warehouse.__str__()
            else:
                print(f"the worker has moved into a box in {new_worker_position}")
                pushed_box = problem.Push_which_box(warehouse.boxes,new_worker_position)
                pushed_box_index = warehouse.boxes.index(pushed_box)
                new_box_position = problem.Move_element(pushed_box,action)
                
                box_in_walls = problem.Check_if_A_in_B(new_box_position,warehouse.walls)
                # leftover_boxes = copy.deepcopy(boxes).remove(pushed_box)
                leftover_boxes = copy.deepcopy(warehouse.boxes)
                leftover_boxes.remove(pushed_box)
                print(f"The leftover boxes are {leftover_boxes}")
                
                box_in_boxes = problem.Check_if_A_in_B(new_box_position,leftover_boxes)
                
                if not box_in_walls and not box_in_boxes:
                    print(f"The box can move into {new_box_position} as it is not moving into any boxes or into walls")
                    if not problem.Check_if_stuck_on_edge(new_box_position,leftover_boxes):
                        print(f"Before: {warehouse.boxes}")
                        warehouse.boxes[pushed_box_index]=new_box_position
                        print(f"After: {warehouse.boxes}")
                        warehouse.worker = new_worker_position
                    else:
                        print("The box moved into a corner")
                        print(warehouse.__str__())
                else:
                    print(f"The box wants to move into {new_box_position} but it is moving into a box or a wall")
                    print(warehouse.__str__())
                    valid_action=False
        else:
            print(f"The worker is trying to move into {new_worker_position} which is in a wall")
            print(warehouse.__str__())
            valid_action=False

    # print("exited")
    if not valid_action:
        return "Impossible"
    else:
        return warehouse.__str__()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    problem = SokobanPuzzle(warehouse=warehouse)
    solution = search.astar_graph_search(problem)
    print(f"The solution is {solution}")
    print(f"The path cost is {solution.path_cost} for a depth of {solution.depth} and the number of states traversed is {len(solution.path())}")
    
    if solution is not None:                
        S = [i.action for i in solution.path()][1:]
        C = solution.path_cost
        # d = solution.depth
        return S,C

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ =="__main__":
    import os,sys
    os.system('cls')
    location = sys.path[0]
    os.chdir(location)    

    test_suite()
    pre="warehouses/"
    in_string = pre+"warehouse_8a.txt"
    x = sokoban.Warehouse()
    x.load_warehouse(in_string)
    s,c=solve_weighted_sokoban(x)
    out = check_elem_action_seq(x,s)
    

# if __name__ =="__main__":
# 
#     from datetime import datetime
#     from datetime import timedelta
#     import time
#     import os,sys
#     from openpyxl import Workbook
#     os.system('cls')
#     location = sys.path[0]
#     os.chdir(location)
#     # print(location)
# 
# 
#     results=[]
#     pre="warehouses/"
#     warehouses=os.listdir("warehouses")
#     exclusions = ['warehouse_03_impossible.txt','warehouse_07.txt','warehouse_33.txt','warehouse_09.txt', 'warehouse_35.txt','warehouse_13.txt','warehouse_17.txt','warehouse_19.txt','warehouse_27.txt','warehouse_39.txt','warehouse_47.txt','warehouse_49.txt','warehouse_53.txt','warehouse_57.txt','warehouse_59.txt','warehouse_5n.txt','warehouse_61.txt','warehouse_63.txt','warehouse_65.txt','warehouse_69.txt','warehouse_6n.txt','warehouse_71.txt','warehouse_73.txt','warehouse_75.txt']
#     test_warehouse = [wh for wh in warehouses if wh not in exclusions]
# 
# 
# 
#     for wh in test_warehouse:
# 
#         print("#################################################")
#         in_string = pre+wh
#         print(in_string)
#         x = sokoban.Warehouse()
#         x.load_warehouse(in_string)
#         t0 = time.time()
#         s,c,d=solve_weighted_sokoban(x)
#         t1 = time.time()
#         t_total = t1-t0
#         print('A* Solver took {:.6f} seconds'.format(t_total))
#         print(f"Depth = {d}")
#         results.append([in_string, t_total, d,c])
# 
#     filename = "warehouses.xlsx"
# 
# 
#     workbook = Workbook()
#     sheet = workbook.active
#     sheet.append(["Warehouse Name","Run Time","Solution Depth","Path Cost"])
#     for i,j  in enumerate(results):
#         sheet[f"A{i+1}"] = j[0]
#         sheet[f"B{i+1}"] = j[1]
#         sheet[f"C{i+1}"] = j[2]
#         sheet[f"D{i+1}"] = j[3]
# 
#     workbook.save(filename=filename)

        
# 
# if __name__ =="__main__":
#     import os,sys
#     os.system('cls')
#     location = sys.path[0]
#     os.chdir(location)    
#     # print(location)
#     pre="warehouses/"
#     warehouses=os.listdir("warehouses")
#     # w_str = "\n #####\n #  .#\n #$  #\n #@  #\n #####"
#     # for i in warehouses[6:7]:
#     in_string = pre+"warehouse_8a.txt"
#     # print(in_string)
#     x = sokoban.Warehouse()
#     # x.from_string(w_str)
#     x.load_warehouse(in_string)
#     s,c=solve_weighted_sokoban(x)
#     out = check_elem_action_seq(x,s)
# 
#     # solutions_depths={}
#     # solution_run_times={}
#     # solution_path_costs={}
#     # warehouse_object = sokoban.Warehouse()
#     # warehouse_names = []
#     # for warehouse in warehouse_names:
#     #     x.load_warehouse(warehouse)
#     #     time_start = ???
#     #     S,C = solve_weighted_sokoban
#     #     time_end =???
#     #     solutions_depths[warehouse]=solution.depth
#     #     solution_run_times[warehouse]=time_end-timestart
#     #     solution_path_costs[warehouse]=solution.path_cost
# 
#         ### make a graph of solution depth vs run time
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
