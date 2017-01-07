import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import itertools

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.epsilon_c = 1.0
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        
        self.deadline_states = ['no_rash', 'faster', 'hurry']
        self.cross_traffic_states = ['cross_traffic', 'no_cross_traffic']
        #self.generate_accident_states()
        
    def generate_accident_states(self):
        # Major accident
        self.__generate_safety_states(-3.0, 'any', 'forward', 'any', 'red', 'forward')
        self.__generate_safety_states(-3.0, 'forward', 'any', 'any', 'red', 'forward')
        self.__generate_safety_states(-3.0, 'any', 'forward', 'any', 'red', 'left')
        self.__generate_safety_states(-3.0, 'forward', 'any', 'any', 'red', 'left')
        self.__generate_safety_states(-3.0, 'any', 'any', 'right', 'red', 'left')
        # Accident
        self.__generate_safety_states(-2.0, 'any', 'any', 'right', 'green', 'left')
        self.__generate_safety_states(-2.0, 'any', 'any', 'forward', 'green', 'left')
        self.__generate_safety_states(-2.0, 'forward', 'any', 'any', 'green', 'right')

    def __generate_safety_states(self, new_value, left, right, incoming, light, action):
        states = itertools.product(self.valid_actions, self.__all_states(left), self.__all_states(right), self.__all_states(incoming), [light], self.deadline_states)
        for state in states:
            self.createQ(state)
            self.Q[state][action] = new_value

    def __all_states(self, state):
        if state == 'any':
            all_states = self.valid_actions
        else:
            all_states = [ state ]
        return all_states
    
    def generate_direction_state(self):
        #any
        pass

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            #self.epsilon = 1.0 - self.epsilon_c / (600.0 - self.epsilon_c)
            self.epsilon = self.epsilon - 0.003
            self.epsilon_c += 1

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline
        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, self.cross_traffic(inputs['left'], inputs['right']), inputs['oncoming'], inputs['light'], self.position_state(deadline))

        return state

    def cross_traffic(self, left, right):
        if left == 'forward' or right == 'forward':
            return 'cross_traffic'
        return 'no_cross_traffic'
    
    def position_state(self, deadline):
        if deadline > 20:
            return 'no_rash'
        if deadline < 20 and deadline > 10:
            return 'faster'
        return 'hurry'

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        stateQ = self.Q[state]

#        actions = {k: v for (k, v) in stateQ.items() if v == 0.0 }.keys()
#        if len(actions) > 0:
#            action = random.choice(list(actions))
#            return (action, 0.0)

        max_v = max(stateQ.values())
        actions = {k: v for (k, v) in stateQ.items() if v == max_v}.keys()
        action = random.choice(list(actions))
        return (action, max_v)


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if state not in self.Q:
            actionsQ = { x: 30.0 for x in self.valid_actions}
            self.Q[state] = actionsQ
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        if not self.learning:
            action = self.valid_actions[random.randint(0,3)]
        else:
            chooseRandom = (random.random() < self.epsilon)
            if chooseRandom:
                action = self.valid_actions[random.randint(0,3)]
            else:
                maxAction, maxQ = self.get_maxQ(state)
                action = maxAction

        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
 
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        maxQ = None
        if self.Q[state][action] == 30.0:
            self.Q[state][action] = reward
            return

        all_states_generator = self.generate_all_states()
        for next_state in all_states_generator:
            self.createQ(next_state)
            maxAction, local_max = self.get_maxQ(next_state)
            if maxQ is None or local_max > maxQ:
                maxQ = local_max
        new_Q = (1.0 - self.alpha) * self.Q[state][action] + self.alpha * (reward + maxQ)
        self.Q[state][action] = new_Q
        return

    def generate_all_states(self):
        #state = (waypoint, inputs['left'], inputs['right'], inputs['oncoming'], inputs['light'])
        all_waypoint_states = self.valid_actions[1:]
        all_oncoming_states = self.valid_actions
        all_light_states = ['green', 'red']
#
        return itertools.product(all_waypoint_states, self.cross_traffic_states, all_oncoming_states, all_light_states, self.deadline_states)


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha = 0.8, epsilon = 1)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.01, log_metrics = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10, tolerance = 0.05)


if __name__ == '__main__':
    run()
