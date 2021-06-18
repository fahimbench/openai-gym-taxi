import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class TaxiEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        num_states = 10000
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 8
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        #DEBUT
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx1 in range(len(locs) + 1):  # +1 pour l'état du passager1 dans la voiture
                    for dest_idx1 in range(len(locs)):
                        for pass_idx2 in range(len(locs) + 1):  # +1 pour l'état du passager2 dans la voiture
                            for dest_idx2 in range(len(locs)):
                                state = self.encode(row, col, pass_idx1, dest_idx1,pass_idx2, dest_idx2)
                                if (pass_idx1 < 4 and pass_idx2 < 4 and pass_idx1 != dest_idx1 and pass_idx1 != pass_idx2 and pass_idx1 != dest_idx2 and
                                dest_idx1 != dest_idx2 and dest_idx1 != pass_idx2 and pass_idx2 != dest_idx2):
                                    initial_state_distrib[state] += 1
                                for action in range(num_actions):
                                    # defaults
                                    new_row, new_col, new_pass_idx1, new_pass_idx2 = row, col, pass_idx1, pass_idx2
                                    reward = -1  # default reward when there is no pickup/dropoff
                                    done = False
                                    taxi_loc = (row, col)

                                    if action == 0:
                                        new_row = min(row + 1, max_row)
                                    elif action == 1:
                                        new_row = max(row - 1, 0)
                                    if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                        new_col = min(col + 1, max_col)
                                    elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                        new_col = max(col - 1, 0)
                                    elif action == 4:  # pickup
                                        if pass_idx1 < 4 and taxi_loc == locs[pass_idx1] and locs[pass_idx1] != locs[dest_idx1]:
                                            new_pass_idx1 = 4
                                        else:
                                            reward = -10
                                    elif action == 5: #dropoff
                                        if taxi_loc == locs[dest_idx1] and pass_idx1 == 4:
                                            new_pass_idx1 = dest_idx1
#                                             reward = 40
                                        else:
                                            reward = -10
                                    elif action == 6:
                                        if pass_idx2 < 4 and taxi_loc == locs[pass_idx2] and locs[pass_idx2] != locs[dest_idx2]:
                                            new_pass_idx2 = 4
                                        else:
                                            reward = -10
                                    elif action == 7:
                                        if taxi_loc == locs[dest_idx2] and pass_idx2 == 4:
                                            new_pass_idx2 = dest_idx2
#                                             reward = 40
                                        else:
                                            reward = -10

                                    if(new_pass_idx1 < 4 and new_pass_idx2 < 4 and locs[new_pass_idx1] == locs[dest_idx1] and locs[new_pass_idx2] == locs[dest_idx2]):
                                        reward = 20
                                        done = True

                                    new_state = self.encode(
                                        new_row, new_col, new_pass_idx1, dest_idx1, new_pass_idx2, dest_idx2)
                                    P[state][action].append(
                                        (1.0, new_state, reward, done))

        #FIN
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc1, dest_idx1, pass_loc2, dest_idx2):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc1
        i *= 4
        i += dest_idx1
        i *= 5
        i += pass_loc2
        i *= 4
        i += dest_idx2
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx1, dest_idx1, pass_idx2, dest_idx2  = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        #si les 2 passagers sont dans le taxi
        if pass_idx1 == 4 and pass_idx2 == 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)
        #si seulement le passager 1 est dans le taxi
        elif pass_idx1 == 4 and pass_idx2 != 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[pass_idx2]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'green', bold=True)
        #si seulement le passager 2 est dans le taxi
        elif pass_idx1 != 4 and pass_idx2 == 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'cyan', highlight=True)
            pi, pj = self.locs[pass_idx1]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # no passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'red', highlight=True)
            pi, pj = self.locs[pass_idx1]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
            pi, pj = self.locs[pass_idx2]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'green', bold=True)

        di, dj = self.locs[dest_idx1]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')

        di, dj = self.locs[dest_idx2]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'red')

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West","Pickup P1", "Dropoff P1", "Pickup P2", "Pickup P2"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
