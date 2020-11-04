import gym
from gym import spaces
import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
import math

import os


def get_truncated_normal(mean=0, std=1, low=0, high=10):
    return truncnorm(
        (low - mean) / std, (high - mean) / std, loc=mean, scale=std)


class Solar():
    def __init__(self):
        samples = np.array(range(0, 24 * 100, 1)) / 100
        self.dist = get_truncated_normal(mean=12, std=24 ** 1 / 10, low=0, high=24)
        max_sample = max([self.dist.pdf(x) for x in samples])
        self.shift = 1 / max_sample

    def solar_output(self, hour):
        return self.dist.pdf(hour) * self.shift


class FooEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(FooEnv, self).__init__()
        self.seed()

        self.data = data

        self.battery_capacity = np.random.uniform(10, 10 * 100)  # kWh
        self.bias = self.battery_capacity / (10 * 100)

        self.battery_charge = 0
        self.i = 1

        self.battery_rate = get_truncated_normal(mean=self.bias * 5 * 100 + 5, std=np.sqrt(5 * 100),
                                                 low=5, high=5 * 100).ppf(np.random.uniform())
        # self.battery_rate = np.random.uniform(5, 5*100) # kW

        self.solar_sqft = get_truncated_normal(mean=self.bias * 1000 * 50 + 100, std=np.sqrt(1000 * 50), low=100,
                                               high=1000 * 50).ppf(np.random.uniform())
        # self.solar_sqft = np.random.uniform(100, 1000*50) #sqft

        # https://www.researchgate.net/figure/Average-daily-energy-consumption-during-the-weekdays-and-the-variation-throughout-the_fig2_326358349

        self.solar = Solar()
        self.solar_rate = lambda hour: self.solar_sqft * 15 * 1 / 1000 * self.solar.solar_output(hour)  # 15 w/sqft * sqft * kw/w = kW
        # TODO replace this with a function of solar outpuit

        consumption_LUT_base = {  # kWh
            0: 9.5,
            0.5: 8,
            1: 7,
            1.5: 5.5,
            2: 5,
            2.5: 4.5,
            3: 4.25,
            3.5: 4,
            4: 4.25,
            4.5: 4.5,
            5: 4.25,
            5.5: 4,
            6: 5,
            6.5: 5.5,
            7: 6,
            7.5: 6.75,
            8: 6,
            8.5: 5.25,
            9: 5,
            9.5: 5,
            10: 5,
            10.5: 5.25,
            11: 5.1,
            11.5: 5,
            12: 5,
            12.5: 5,
            13: 4.8,
            13.5: 4.75,
            14: 4.6,
            14.5: 4.75,
            15: 5,
            15.5: 5.15,
            16: 5.3,
            16.5: 5.75,
            17: 5.9,
            17.5: 6,
            18: 7,
            18.5: 8,
            19: 9,
            19.5: 10,
            20: 10,
            20.5: 9.9,
            21: 9.5,
            21.5: 9.25,
            22: 10,
            22.5: 11,
            23: 11,
            23.5: 10.25,
            24: 9.5,
        }
        self.consumption_LUT = {}
        for k,v in consumption_LUT_base.items():
            self.consumption_LUT[k] = v*self.bias*5 # TODO this is uniform, but it should probably be normal with ...
                                                    # positive skew and mode around default values
        self.action_space = spaces.Discrete(3)  # discharge, idle, store

        self.low = np.array(
            [-100000, 0, 10, 0, 5, 0, 0], dtype=np.float32
        )
        self.high = np.array(
            [100000, 200, 10 * 100, 10 * 100, 5 * 100, 10000000, 289], dtype=np.float32
        )
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)
        # profit, price, capacity, charge, battery_rate, solar_rate, i

        self.charge_hist = []
        self.price_hist = []
        self.action_hist = []
        self.profit_hist = []

        set_params = True
        if set_params:
            self.battery_capacity = 14*3 # 3 tesla powerwalls
            self.battery_rate = 5*3 # in parallel
            self.solar_sqft = 1500
            self.solar_rate = lambda i: self.solar_sqft * 15 * 1 / 1000 * self.solar.solar_output(i/5*60)
            self.consumption_LUT = consumption_LUT_base
                             # profit, price, capacity, charge, brate,  solar rate, i
            self.low = np.array([-100000, 0,   14*3-1,   0,     14.99,       0,    0], dtype=np.float32)
            self.high = np.array([100000, 200, 14*3+1,   14*3,  15.01, 1500*15/1000,   289], dtype=np.float32)
            self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)


    def step(self, action):
        # Execute one time step within the environment
        total_profit, price, capacity, charge, battery_rate, curr_solar_rate, i = self.state
        # calculate profit
        # get new price
        # set new charge
        # set new i

        delta_t = 5 / 60  # in units hours

        profit = 0
        new_charge = charge

        self.charge_hist.append(charge)
        self.price_hist.append(price)
        self.action_hist.append(action)

        # Calclulate consumption
        hour_low = math.floor(i/6)/2 # 12.5, for example
        hour_high = math.ceil(i/6)/2 # 13, for example

        low_prop = 1
        if hour_low == hour_high: # Div By 0 case
            low_prop = 1
        else:
            low_prop = (i/12 - hour_high)/(hour_low-hour_high)

        consumption = self.consumption_LUT[hour_low]*low_prop + self.consumption_LUT[hour_high]*(1-low_prop)
        surplus = -1*consumption

        #Calucluate solar panel charge delivery
        if action == 0:  # discharge121
            new_charge = max(0, charge - battery_rate * delta_t) # can't undercharge
            discharge = charge - new_charge # add solar panel
            surplus += curr_solar_rate * delta_t + discharge
           # print(charge, new_charge, discharge, battery_rate, delta_t, "\n\n")

        elif action == 1:  # idle
            new_charge = charge
            surplus += curr_solar_rate * delta_t

        elif action == 2:  # charge

            solar_charge_rate = min(curr_solar_rate,  battery_rate)

            grid_charge_rate = battery_rate-solar_charge_rate
            surplus -= grid_charge_rate * delta_t

            new_charge = min(self.battery_capacity, charge + battery_rate * delta_t)

            extra_discharge_rate = max(0, curr_solar_rate - solar_charge_rate)  # find extra energy
            surplus += extra_discharge_rate * delta_t
        else:
            print("ACTION IS UNKNOWN, EXPECTED INT IN RANGE [0,2] FOUND ", action)


        profit = surplus * price / 1000  # price is in $/MWh, discharge is in kWh

        # set up next state
        new_price = self.day_data[i]
        total_profit += profit
        self.profit_hist.append(profit)

        self.state = (total_profit, new_price, capacity, new_charge, battery_rate, self.solar_rate(i+1), i + 1)

        # for debugging
        self.last_action = "Error"
        if action == 0:
            self.last_action = "discharging"
        if action == 1:
            self.last_action = "idling"
        if action == 2:
            self.last_action = "charging"

        # check done state
        done = bool(i == len(self.day_data)-1)
        if done:
            print("Rendering")
            self.render()

        # add penalties
        # action penalty
        action_changes = 0
        if len(self.action_hist) < 6:
            action_changes = 0
        else:
            action_changes = sum([1 if bool(self.action_hist[-6+i] is not self.action_hist[-5+i]) else 0 for i in range(6)])
            if action_changes < 2:
                action_changes = 0
        action_penalty = action_changes * 1 # hyperparameter-able

        # capacity penalty
        capacity_penalty = 0
        if abs(i - 144) < 20:
            if self.battery_charge / self.battery_capacity < .3:
                capacity_penalty = 5

        reward = total_profit + action_penalty
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        days = pd.unique(self.data['Date'])
        nodes = pd.unique(self.data['Node'])
        day = np.random.choice(days)
        node = np.random.choice(nodes)
        self.node = node
        self.day = day
        print(f"Resetting env to {node} on {day}")

        query_df = self.data[(self.data["Date"]==day) & (self.data["Node"] == node)]
        self.day_data = query_df.reset_index()['Value']
        start_interval = query_df.reset_index()["Interval Number"][0]



        self.state = (0, self.day_data[0], self.battery_capacity, 0, self.battery_rate, self.solar_rate(start_interval),
                      start_interval)
        self.charge_hist = []
        self.price_hist = []
        self.action_hist = []
        self.profit_hist = []
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment
        total_profit, price, capacity, charge, battery_rate, solar_rate, i = self.state
        print(f"--------------{self.node}, {self.day}--------------")
        #print(f"{capacity}kWh battery is at {charge / capacity}% after {self.last_action}")
        print(f"${total_profit}")
        #print(f"Battery charge rate is {battery_rate} and current solar output is {solar_rate}")
        if np.random.random() > .95 and os.path.exists('stop.txt'): # my way of asyncronously stopping :)
            smoothed_action_hist = [sum(self.action_hist[i:i+14])/(14*3) - 2 for i in range(len(self.action_hist)-14)]
            # smoothed_action_hist += self.action_hist[-7:]
            plt.plot(range(len(self.charge_hist)), np.array(self.charge_hist)/max(self.charge_hist), label="BCharge")
            plt.plot(range(5), [capacity/max(self.charge_hist)]*5, label = "BCap")
            plt.plot(range(len(self.charge_hist)), np.array(self.price_hist)/(sum(self.price_hist)/len(self.price_hist)), label="Price")
            plt.plot(range(len(smoothed_action_hist)), smoothed_action_hist, label="Action")
            plt.plot(range(len(self.profit_hist)), self.profit_hist, label="ProfitDelta")
            plt.legend()
            plt.show()

        return None