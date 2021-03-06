# -*- coding: utf-8 -*-
from config.log import TASK_RESULT_KEY, EPISODE_RESULT_KEY
import logging
import time
from sets import Set

episode_result_logger = logging.getLogger(EPISODE_RESULT_KEY)
task_result_logger = logging.getLogger(TASK_RESULT_KEY)


class ResultLogger(object):
    def __init__(self):
        self.steps = 0
        self.start_time = time.time()

        self.episode = 0
        self.task = 1

        episode_result_logger.info("task,episode,step,time,agents")
        task_result_logger.info("task,success,failure,agents")

        self.agents = Set()

    def initialize(self):
        self.start_time = time.time()
        self.steps = 0
        self.episode += 1

    def add_agent(self, agent):
        self.agents.add(agent)

    def step(self):
        self.steps += 1

    def report(self, success, failure, finished):
        agent_ids = ':'.join([hex(id(agent)) for agent in self.agents])
        elapsed_time = time.time() - self.start_time
        if finished:
            task_result_logger.info('{}, {}, {}, {}'.format(self.task, success, failure, agent_ids))
            self.task += 1
            self.episode = 0

        episode_result_logger.info('{}, {}, {}, {}, {}'.format(self.task, self.episode, self.steps, elapsed_time, agent_ids))
