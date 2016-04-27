import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# import matplotlib.pyplot as plt
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    INITIAL_EPSILON = 1.0
    MIN_EPSILON = 0.05

    ACTION_EXPLORATION = 1
    ACTION_EXPLOITATION = 2

    def __init__(self, env):
        """
        inputs at each time step:
            Next waypoint location, relative to its current location and heading,
            Intersection state (traffic light and presence of cars), and,
            Current deadline value (time steps remaining),
        """
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        self.state_labels = None  # q matrix 'rows' (states)
        self.state_visits = {}  # count of how many times the agent has visited the state
        self.state_action_visits = {}  # count of how many times the agent has visited the state_action pair

        self.action_labels = None  # q matrix 'columns' (actions)

        #####
        self.using_preloaded_q_matrix = False

        self.q_matrix = None  # q matrix

        # self.q_matrix = {'red_right_obstructed': [0, 0, 0, 2.468524155631328],
        #             'red_left_free': [1.3314034749672143, 0, -0.718325782192847, 0.9564762285714286],
        #             'green_left_free': [1.2711882500674918, 0.8746975800730244, 2.3671708581445214, 0.9531292773079366],
        #             'red_right_free': [1.2284549333333334, -0.7112824024181088, -0.7308284470551725, 2.398533066773934],
        #             'green_right_free': [1.4830812448016857, 0.8296, 0.8738025291305571, 2.417800947120226],
        #             'green_forward_free': [1.3980889598626929, 2.3746681687246927, 0.9544188216062917,
        #                                    0.7964770040379209],
        #             'red_forward_obstructed': [1.3488113257700967, -0.6879899054653977, -0.715812034664014,
        #                                        0.78579290343571]}
        #####

        self.action = None  # previous action taken
        self.reward = 0  # previous reward received
        if self.using_preloaded_q_matrix:
            self.learning_rate = 0.0  # using pre-loaded q-matrix therefore
        else:
            self.learning_rate = 0.5  # determines to what extend the newly acquired information will override the old informatino

        self.discount_factor = 0.2  # determines the importance of future rewards

        self.epsilon = LearningAgent.INITIAL_EPSILON  # exploration variable

        self.total_runs = 0
        self.deadline = 0

        self.steps_taken = 0  # count how many steps we've taken
        self.accumulated_rewards = 0  # sum of rewards for a given run
        self.accumulated_random_counts = 0  # count how many times we make a random choice for a given run
        self.accumulated_penalties = []

        self.recorded_deadlines = []
        self.recorded_rewards = []  # rewards for each move
        self.recorded_random_choice = []
        self.recorded_run_rewards = []  # normalised self.accumulated_rewards for each run
        self.recorded_run_random_choices = []
        self.recorded_run_abs_rewards = []
        self.recorded_run_steps_taken = []
        self.recorded_run_accumulated_penalties = []


        self.create_state_action_matrix()

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward = 0
        self.state = None
        self.action = None
        self.deadline = 0

        self.epsilon = (1.0/float(self.total_runs+1))
        #self.learning_rate = (1.0 / float(self.total_runs + 1))

        self.steps_taken = 0
        self.accumulated_rewards = 0
        self.accumulated_random_counts = 0
        self.accumulated_penalties = []

        if self.total_runs >= 100:
            self.print_q_matrix()

    def create_state_action_matrix(self):
        """

        """
        # state label: <light>_<next_waypoint>
        #self.state_labels = ['green_left', 'green_right', 'green_forward',
        #                     'red_left', 'red_right', 'red_forward']

        self.state_labels = []

        self.action_labels = [None, 'forward', 'left', 'right']

        self.q_matrix = {}

        for state_label in self.state_labels:
            self.q_matrix[state_label] = [0 for _ in range(len(self.action_labels))]

    def get_state_key(self, next_waypoint, deadline, light, oncoming, right, left):
        state_label = "{}_{}".format(light, next_waypoint)

        # if oncoming is not None:
        #     state_label += "_{}".format("oncoming")
        # else:
        #     state_label += "_{}".format("nooncoming")

        # if right is not None:
        #     state_label += "_{}".format("right")
        # else:
        #     state_label += "_{}".format("noright")

        state_label += "_{}".format(self.is_next_waypoint_obstructed(next_waypoint, light, oncoming, right, left))

        if state_label not in self.state_labels:
            self.state_labels.append(state_label)
            self.state_visits[state_label] = 1.0
            self.q_matrix[state_label] = [0 for _ in range(len(self.action_labels))]
        else:
            self.state_visits[state_label] += 1.0

        return state_label

    def is_next_waypoint_obstructed(self, next_waypoint, light, oncoming, right, left):
        next_waypoint_state = "free"

        if next_waypoint == 'right':
            if light == 'red' and left == 'forward':
                next_waypoint_state = "obstructed"
        elif next_waypoint == 'forward':
            if light == 'red':
                next_waypoint_state = "obstructed"
        elif next_waypoint == 'left':
            if 'light' == 'red' or (oncoming == 'forward' or oncoming == 'right'):
                next_waypoint_state = "obstructed"

        return next_waypoint_state

    def get_q(self, state, action):
        return self.q_matrix[state][self.get_action_label_index(action)]

    def set_q(self, state, action, q_value):
        self.q_matrix[state][self.get_action_label_index(action)] = q_value

    def get_max_q(self, state):
        return max(self.q_matrix[state])

    def get_action_label_index(self, action):
        for i in range(len(self.action_labels)):
            if action == self.action_labels[i]:
                return i

        return -1

    def get_action_for_state(self, state):
        #return self.get_action_for_state_random(state=state)

        #return self.get_action_for_state_q_learning_basic(state=state)

        return self.get_action_for_state_q_learning(state=state)

    def get_action_for_state_random(self, state):
        """
        Randomly choose action (our 'control scenario')
        :param state: ignored
        :return: {"action": chosen action, "is_random": Bool indicating if action was randomly selected}
        """
        return {"action": random.choice(self.action_labels), "is_random": True}

    def get_action_for_state_q_learning_basic(self, state):
        """
        Get action for the parameter state
        :param state:
        :return: {"action": chosen action, "is_random": Bool indicating if action was randomly selected}
        """
        epsilon = 0.1

        action = None
        action_randomly_selected = False

        # To explore or exploit our knowledge? Probability of epsilon

        if random.random() < epsilon:
            action_randomly_selected = True
            action = random.choice(self.action_labels)
        else:
            q_values = [self.q_matrix[state][self.get_action_label_index(a)] for a in self.action_labels]
            max_q_value = max(q_values)
            max_action_index = q_values.index(max_q_value)
            action = self.action_labels[max_action_index]

        return {"action": action, "is_random": action_randomly_selected}

    def get_action_for_state_q_learning(self, state):
        """
        Get action for the parameter state
        :param state:
        :return: {"action": chosen action, "is_random": Bool indicating if action was randomly selected}
        """
        epsilon = self.epsilon
        #epsilon = max(1.0/(self.state_visits[state]), 0.1)

        action = None
        action_randomly_selected = False

        # To explore or exploit our knowledge? Probability of epsilon

        if random.random() < epsilon:
            # duplicates take precedence
            action_labels_set = set(self.q_matrix[state])
            action_randomly_selected = True

            if len(action_labels_set) == len(self.q_matrix[state]):
                # zero values take precedence
                idx_zeros = [idx for idx in range(0, len(self.q_matrix[state]))
                             if self.q_matrix[state].count(self.q_matrix[state][idx]) == 0.0]

                if len(idx_zeros) > 0:
                    action = self.action_labels[random.choice(idx_zeros)]
                else:
                    action = random.choice(self.action_labels)
            else:
                idx_duplicates = [idx for idx in range(0, len(self.q_matrix[state]))
                                  if self.q_matrix[state].count(self.q_matrix[state][idx]) > 1]
                action = self.action_labels[random.choice(idx_duplicates)]
        else:
            q_values = [self.q_matrix[state][self.get_action_label_index(a)] for a in self.action_labels]
            max_q_value = max(q_values)
            count = q_values.count(max_q_value)
            if count > 1:
                top_q_values = [i for i in range(len(self.action_labels)) if
                                self.q_matrix[state][i] == max_q_value]
                action = self.action_labels[random.choice(top_q_values)]
                action_randomly_selected = True
            else:
                max_action_index = q_values.index(max_q_value)
                action = self.action_labels[max_action_index]

        return {"action": action, "is_random": action_randomly_selected}

    def update_q_matrix(self, previous_state, previous_action, previous_reward, current_state):
        """
        Reference for formula from https://e.wikipedia.org/wiki/Q-learning
        :param previous_state:
        :param previous_action:
        :param previous_reward:
        :param current_state:
        :return:
        """

        old_value = self.get_q(state=previous_state, action=previous_action)

        expected_future_reward = self.get_max_q(state=current_state)

        state_action = "{}_{}".format(previous_state, previous_action)

        if state_action not in self.state_action_visits:
            self.state_action_visits[state_action] = 1.0
        else:
            self.state_action_visits[state_action] += 1.0

        learning_rate = 1.0 / self.state_action_visits[state_action]
        #learning_rate = self.learning_rate

        q_value = old_value + \
                  learning_rate * \
                  (previous_reward + self.discount_factor * expected_future_reward - old_value)

        self.set_q(state=previous_state, action=previous_action, q_value=q_value)

    def print_q_matrix(self):
        print("\n===Q Matrix===\n")

        header = "".rjust(24, ' ')
        for action in self.action_labels:
            if action is None:
                header += "n".ljust(4, ' ')
            else:
                header += "{}".format(action[0].ljust(4, ' '))

            header += "\t"

        print header

        for state, action_rewards in self.q_matrix.iteritems():
            row = "{}".format(state.rjust(20, ' '))
            row += "\t"
            for action_reward in action_rewards:
                row += str(round(action_reward, 2)).ljust(4, ' ')
                row += "\t"

            print row

        print("\n")

        matrix = np.zeros((len(self.state_labels), len(self.action_labels)))
        row = 0
        col = 0
        for state, action_rewards in self.q_matrix.iteritems():
            for action_reward in action_rewards:
                #matrix[row][col] = action_reward
                matrix[(row, col)] = action_reward
                col += 1

            col = 0
            row += 1

        matrix = matrix/matrix.max()

        #self.plot_matrix(matrix)

    def update(self, t):
        """
        Task is to learn the road rules (as outlined in https://discussions.udacity.com/t/intention-of-the-project/163473)

        state:
            - next_waypoint {forward, left, right}
            - light {green, red}

            - oncoming
            - left
            - right

            - deadline
        """
        self.steps_taken += 1

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        self.deadline = self.env.get_deadline(self)

        # {'light': 'green', 'oncoming': None, 'right': None, 'left': None}
        inputs = self.env.sense(self)

        # TODO: Update state
        previous_state = self.state
        self.state = self.get_state_key(next_waypoint=self.next_waypoint,
                                        deadline=self.deadline,
                                        light=inputs["light"],
                                        oncoming=inputs["oncoming"],
                                        right=inputs["right"],
                                        left=inputs["left"])

        # TODO: Learn policy based on state, action, reward
        if previous_state is not None:
            # update_q_matrix(self, previous_state, previous_action, previous_reward, current_state):
            self.update_q_matrix(
                previous_state=previous_state,
                previous_action=self.action,
                previous_reward=self.reward,
                current_state=self.state)
        
        # TODO: Select action according to your policy
        next_action = self.get_action_for_state(state=self.state)
        self.recorded_random_choice.append(next_action["is_random"])
        self.action = next_action["action"]

        if next_action["is_random"]:
            self.accumulated_random_counts += 1

        # Execute action and get reward
        self.reward = self.env.act(self, self.action)
        self.recorded_rewards.append(self.reward)

        self.accumulated_rewards += self.reward

        if self.reward < 0:
            self.accumulated_penalties.append({
                "state": self.state,
                "action": self.action
            })

        # if self.reward <= 0.5 and not next_action["is_random"]:
        #     print "bad move? \nstate: {} action: {}\ninputs: {}".format(self.state, self.action, inputs)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, " \
        #       "reward = {}, state = {}"\
        #     .format(deadline, inputs, self.action, self.reward, self.state)  # [debug]

        # self.print_q_matrix()

    def on_finished(self):
        self.total_runs += 1.0
        self.recorded_deadlines.append(self.deadline)

        self.recorded_run_abs_rewards.append(self.accumulated_rewards)
        self.recorded_run_steps_taken.append(self.steps_taken)

        self.recorded_run_rewards.append(float(self.accumulated_rewards)/float(self.steps_taken))
        self.recorded_run_random_choices.append(float(self.accumulated_random_counts)/float(self.steps_taken))
        #self.recorded_run_random_choices.append(float(self.accumulated_random_counts))

        self.recorded_run_accumulated_penalties.append(self.accumulated_penalties)

        print "\n=== ===\n"
        print "\n"
        print "LearningAgent.on_finished(): deadline = {} ({})".format(self.deadline,
                                                                       (float(self.deadline)/float(self.total_runs)))
        #self.print_q_matrix()

        if self.total_runs >= 100:
            self.print_q_matrix()

            print ""

            print "deadlines_array = {}".format(self.recorded_deadlines)
            print "random_choice_array = {}".format(self.recorded_random_choice)
            print "rewards_array = {}".format(self.recorded_rewards)
            print "state_labels = {}".format(self.state_labels)
            print "action_labels = {}".format(self.action_labels)
            print "q_matrix = {}".format(self.q_matrix)
            print "recorded_run_rewards = {}".format(self.recorded_run_rewards)
            print "recorded_run_random_choices = {}".format(self.recorded_run_random_choices)

            print "recorded_run_abs_rewards = {}".format(self.recorded_run_abs_rewards)
            print "recorded_run_steps_taken = {}".format(self.recorded_run_steps_taken)

            print "recorded_run_accumulated_penalties = {}".format(self.recorded_run_accumulated_penalties)

            run_penalties = [len(penalties) for penalties in self.recorded_run_accumulated_penalties]
            print "run_penalties = {}".format(run_penalties)

            state_visits_labels = []
            state_visits_count = []
            for key, value in self.state_visits.iteritems():
                state_visits_labels.append(key)
                state_visits_count.append(int(value))

            print "state_visits_labels = {}".format(state_visits_labels)
            print "state_visits_count = {}".format(state_visits_count)



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    if a.using_preloaded_q_matrix:
        sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
        sim.run(n_trials=10)  # press Esc or close pygame window to quit
    else:
        sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
        sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()

    #print max([1,2,1,1])
    #t = {"a":2, "b":2, "c":1}
    #print max(t)

    #t2 = {"a": 2, "b": 2, "c": 1}

    # print max([9,2,3,4,5,6])
