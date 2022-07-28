
from collections import defaultdict, namedtuple

Task = namedtuple('Task', ['start', 'finish'])
Rotation = namedtuple('Rotation', ['tasks', 'cost', 'duration'])

# class for nice set printing
class frozenset(frozenset):
    __repr__ = lambda self: '{%s}' % str.join(', ', map(str, sorted(self)))


class GangSchedulingProblem:
    """Gang Scheduling problem """
    def __init__(self, input_file):
        file_contents = [tuple(map(int, line.split())) for line in input_file]
        nr_tasks, time_limit = file_contents.pop(0)

        self.time_limit = time_limit
        self.tasks = [Task(*t) for t in file_contents[:nr_tasks]]
        self.transition_costs = defaultdict(lambda: float('inf'))
        self.possible_transitions = [list() for task in self.tasks]
        for i, j, cost in file_contents[nr_tasks:]:
            # Reindex to start from 0, not from 1 as in the instance
            self.transition_costs[i - 1, j - 1] = cost
            self.possible_transitions[i - 1].append(j - 1)

    def generate_rotations(self, from_rotation=()):
        if from_rotation:
            candidates = self.possible_transitions[from_rotation[-1]]
        else:
            candidates = range(len(self.tasks))

        for task in candidates:
            rotation = from_rotation + (task,)
            start_time  = self.tasks[rotation[0]].start
            finish_time = self.tasks[rotation[-1]].finish
            duration = finish_time - start_time
            if duration > self.time_limit:
                continue
            cost = sum(self.transition_costs[t] for t in zip(rotation, rotation[1:]))
            yield Rotation(frozenset(rotation), cost, duration)

            for r in self.generate_rotations(rotation):
                yield r


def main():
    problem_file = open("gsp_50tasks.txt")

    gsp = GangSchedulingProblem(problem_file)

    print(f'Problem size: %d' % len(gsp.tasks)) 
    print(f'Time limit: %d  ' % gsp.time_limit) 
    print(f'Transitions: %d ' % len(gsp.transition_costs)) 

    for i in gsp.generate_rotations():
        print(i)

if __name__ == '__main__':
    main()