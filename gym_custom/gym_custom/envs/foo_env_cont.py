import gym
from gym import spaces
import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
import math

import os



class FooEnvCont(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data=None):
        if data is None:
               data = np.load(r'E:\Projects\ProjectX\large_data_cleaned.npy') #pd.read_csv(r'E:\Projects\ProjectX\cleaned_data2020_v2.csv')
        super(FooEnvCont, self).__init__()
        self.seed()

        self.d = 0
        self.data = data

        self.battery_capacity = np.random.uniform(10, 10 * 100)  # kWh
        self.bias = self.battery_capacity / (10 * 100)

        self.battery_charge = 0
        self.i = 1


        self.min_action = -1.0
        self.max_action = 1.0

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )


        self.price_energy_cumulative = 0
        self.charge_hist = []
        self.price_hist = []
        self.action_hist = []
        self.profit_hist = []
        self.default_hist = []
        self.default_solar_hist = []
        self.reward_hist = []
        self.solar_hist = []

        self.dummy_hist = []

        self.reward = 0

        self.battery_capacity = 14*3 # 3 tesla powerwalls
        self.battery_rate = 5*3 # in parallel

                             # battery_profit, price, rolling average, last_hour_average, last_hour_action_avg,, last_half_hour_action_avg, charge_prop, i, i_prop
        self.low = np.array([-1000, -50, -50, -50,  0, 0,    0,    0, 0], dtype=np.float32)
        self.high = np.array([1000, 150, 150, 150, 2, 2,  1,  276, 1], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)


    def step(self, action):
        # Execute one time step within the environment
        # battery_profit, price, rolling average, last_hour_average, last_hour_action_avg,, last_half_hour_action_avg, charge_prop, i, i_prop
        battery_profit, price, rolling_average, last_hour_average, last_hour_action_avg, last_half_hour_action_avg, charge_prop, i, i_prop = self.state

        capacity = self.battery_capacity
        charge = charge_prop * self.battery_capacity

        # calculate profit
        # get new price
        # set new charge
        # set new i

        delta_t = 5 / 60  # in units hours

        new_charge = 0

        self.charge_hist.append(charge)
        self.price_hist.append(price)
        self.action_hist.append(action)


       # print(surplus_energy*price/1000, curr_solar_rate*price/1000)

        lost_energy_penalty = 0

        price_energy = 0
        # net flow into/out of battery time price, ignoring solar panels
        action = action[0]
        if charge == 0 and action <- .1: # discharge
            lost_energy_penalty += abs(action)
        if charge == self.battery_capacity and action>.1:
            lost_energy_penalty += abs(action) #charge

        delta = self.battery_rate * delta_t * action
        new_charge = max(min(charge + delta, self.battery_capacity), 0)
        discharge = charge - new_charge  # add solar panel

        price_energy = discharge * price / 1000

        #Calucluate solar panel charge delivery

        # set up next state

        # check done state
        done = bool(i == len(self.day_data)-1)



        # add penalties
        # action penalty
        action_changes = 0
        if len(self.action_hist) < 6:
            action_changes = 0
        else:
            action_changes = sum([np.abs(self.action_hist[-6+i] - self.action_hist[-5+i]) if bool(np.abs(self.action_hist[-6+i] - self.action_hist[-5+i])>.2) else 0 for i in range(6)])
        try:
            _ = len(action_changes)
            action_changes = action_changes[0]
        except:
            pass
        action_penalty = action_changes
        # lost energy penalty

        #print(action, price_energy)
        self.price_energy_cumulative += price_energy
        battery_profit = self.price_energy_cumulative
        self.reward += price_energy

        self.dummy_hist.append(self.price_energy_cumulative)
        #self.reward += action_penalty*-1/50
       # reward += capacity_penalty*profit*.1*-1
        #self.reward += lost_energy_penalty*delta_t*1/100
       # print(total_profit, profit, action_penalty, capacity_penalty, lost_energy_penalty)
       # print(total_profit, profit,  action_penalty, capacity_penalty, lost_energy_penalty)
        #rint(self.battery_charge, self.battery_capacity, curr_solar_rate)

        self.reward_hist.append(self.reward)

        if done:
            average_last_price = sum(sorted(self.price_hist[:-5])[1:4])/3
            average_last_price *= .8 # punish it a little
            self.reward += new_charge*average_last_price/1000 #discharge the rest of the battery when the day ends at curr price

            new_charge = 0
            # battery_profit, price, rolling average, last_hour_average, last_hour_action_avg,, last_half_hour_action_avg, charge_prop, i, i_prop
            self.state = [battery_profit, price, rolling_average, last_hour_average, last_hour_action_avg, last_half_hour_action_avg,
                                           charge_prop, i, i_prop]

            self.battery_charge = 0
            self.reward_hist[-1] = self.reward
            # self.render()
        else:
            new_price = self.day_data[i + 1]

            rolling_average = np.mean(self.day_data[:i + 1])
            if i >= 11:
                last_hour_average = np.mean(self.day_data[-11 + i: i + 1])
                last_hour_action_avg = np.mean(self.action_hist[-11+i: i+1])
            else:
                last_hour_average = rolling_average
                last_hour_action_avg = np.mean(self.action_hist[:i + 1])

            if i>= 5:
                last_half_hour_action_avg = np.mean(self.action_hist[-5:i + 1])
            else:
                last_half_hour_action_avg = np.mean(self.action_hist[:i + 1])

            charge_prop = new_charge / self.battery_capacity
            # battery_profit, price, rolling average, last_hour_average, last_hour_action_avg,, last_half_hour_action_avg, charge_prop, i, i_prop

            self.state = [battery_profit, self.day_data[i + 1], rolling_average, last_hour_average, last_hour_action_avg,
                          last_half_hour_action_avg, charge_prop, i+1, (i+1)/len(self.day_data)]

        return np.array(self.state), self.reward, done, {'p': battery_profit}

    def reset(self):
        # Reset the state of the environment to an initial state


        self.node = "NPY"
        self.day = np.random.choice(range(len(self.data)))
        self.reward = 0
        self.day_data = self.data[self.day]
        start_interval = 0
        # battery_profit, price, rolling average, last_hour_average, last_hour_action_avg,, last_half_hour_action_avg, charge_prop, i, i_prop

        self.state = [0, self.day_data[0], self.day_data[0], self.day_data[0], 1,
                          1, 0, 0, 0/len(self.day_data)]

        self.charge_hist = []
        self.price_hist = []
        self.action_hist = []
        self.profit_hist = []
        self.default_hist = []
        self.default_solar_hist = []
        self.reward_hist = []
        self.solar_hist = []
        self.dummy_hist = []
        self.price_energy_cumulative = 0

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment
        # profit, price, rolling average, last_hour_average, charge_prop,  solar rate, i
        battery_profit, price, rolling_average, last_hour_average, last_hour_action_avg, last_half_hour_action_avg, charge_prop, i, i_prop = self.state


        #print(f"{capacity}kWh battery is at {charge / capacity}% after {self.last_action}")
        print(f"Completed: {self.node}, {self.day} || {battery_profit} diff || {self.reward_hist[-1]} Reward")
        #print(f"Battery charge rate is {battery_rate} and current solar output is {solar_rate}")
        if (os.path.exists('stop.txt') or os.path.exists('print.txt')): # my way of asyncronously stopping :)
            #smoothed_action_hist = [sum(self.action_hist[i:i+14])/(14*3) - 2 for i in range(len(self.action_hist)-14)]

            #discharge = np.ma.masked_where(self.action_hist == 0, self.action_hist)-3
            #idle = np.ma.masked_where(self.action_hist == 1, self.action_hist)-3
            #charge = np.ma.masked_where(self.action_hist == 2, self.action_hist)-3
            #dotted_capacity_line = [capacity/max(self.charge_hist)+.00001]*20
            plt.plot(range(len(self.charge_hist)), np.array(self.charge_hist)/(self.battery_capacity), label="BCharge")
            #plt.plot(range(len(dotted_capacity_line)), dotted_capacity_line, label = "BCap")
            plt.plot(range(len(self.price_hist)), np.array(self.price_hist)/(sum(self.price_hist)/len(self.price_hist)), label="Price")
            plt.plot(range(len(self.action_hist)), [x/3-3 for x in self.action_hist], label="Action")
            #plt.plot(range(len(self.profit_hist)), self.profit_hist, label="ProfitDelta")
            plt.plot(range(len(self.reward_hist)), [x/max(np.abs(min(self.reward_hist)), max(self.reward_hist)) for x in self.reward_hist], label="Reward")
            #plt.plot(range(len(self.solar_hist)), self.solar_hist, label="Solar Output")
            plt.plot(range(len(self.dummy_hist)), [x-2 for x in self.dummy_hist], label="BatteryProfit")
            plt.legend()
            if os.path.exists('stop.txt'):
                plt.show()
            elif os.path.exists('print.txt'):
                plt.savefig(f"figure_alt_2.png")
            plt.clf()


        return None