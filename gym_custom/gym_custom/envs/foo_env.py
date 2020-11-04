import gym
from gym import spaces
import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt

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
        self.solar = Solar()
        self.solar_rate = lambda hour: self.solar_sqft * 15 * 1 / 1000 * self.solar.solar_output(hour)  # 15 w/sqft * sqft * kw/w = kW
        # TODO replace this with a function of solar outpuit

        self.action_space = spaces.Discrete(3)  # discharge, idle, store

        self.low = np.array(
            [-100000, 0, 10, 0, 5, 0, 0], dtype=np.float32
        )
        self.high = np.array(
            [100000, 1000, 10 * 100, 10 * 100, 5 * 100, 10000000, 289], dtype=np.float32
        )
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)
        # profit, price, capacity, charge, battery_rate, solar_rate, i

        self.charge_hist = []
        self.price_hist = []
        self.action_hist = []
        self.profit_hist = []

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

        if action == 0:  # discharge
            new_charge = max(0, charge - battery_rate * delta_t)
            discharge = charge - new_charge + curr_solar_rate * delta_t
            profit = discharge * price / 1000  # price is in $/MWh, discharge is in kWh
        elif action == 1:  # idle
            new_charge = charge

            discharge = curr_solar_rate * delta_t
            profit = discharge * price / 1000  # price is in $/MWh, discharge is in kWh
        elif action == 2:  # charge

            charge_rate = min(curr_solar_rate,  battery_rate)  # if battery can't charge as fast we need to use extra energy to sell
            new_charge = min(self.battery_capacity, charge + charge_rate * delta_t)

            extra_discharge_rate = max(0, curr_solar_rate - charge_rate)  # find extra energy
            discharge = extra_discharge_rate * delta_t

            profit = discharge * price / 1000  # price is in $/MWh, discharge is in kWh
        else:
            print("ACTION IS UNKNOWN, EXPECTED INT IN RANGE [0,2] FOUND ", action)
        new_price = self.day_data[i]  # +1 -1
        total_profit += profit
        self.profit_hist.append(profit)

        self.last_action = "Error"
        if action == 0:
            self.last_action = "discharging"
        if action == 1:
            self.last_action = "idling"
        if action == 2:
            self.last_action = "charging"


        self.state = (total_profit, new_price, capacity, new_charge, battery_rate, self.solar_rate((i+1) * 5 / 60), i + 1)

        done = bool(i == len(self.day_data)-1)
        if done:
            print("Rendering")
            self.render()

        action_penalty = 0
        if len(self.action_hist) < 6:
            action_penalty = 0
        else:
            action_changes = sum([1 if bool(self.action_hist[-6+i] is not self.action_hist[-5+i]) else 0 for i in range(6)])
            if action_changes > 2:
                action_penalty = total_profit*-1/(10-action_changes)

        reward = total_profit + action_penalty
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        days = pd.unique(self.data['Date'])
        day = np.random.choice(days)
        print(f"Initializing on {day}")
        self.day_data = self.data[self.data["Date"]==day].reset_index()['Value']
        start_interval = self.data[self.data["Date"]==day].reset_index()["Interval Number"][0]
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
        print(f"--------------{i}--------------")
        print(f"{capacity}kWh battery is at {charge / capacity}% after {self.last_action}")
        print(f"${total_profit} at current price of ${price}")
        print(f"Battery charge rate is {battery_rate} and current solar output is {solar_rate}")
        if input("View?"):
            smoothed_action_hist = [sum(self.action_hist[i:i+7])/21 - 1 for i in range(len(self.action_hist)-7)]
            # smoothed_action_hist += self.action_hist[-7:]
            plt.plot(range(len(self.charge_hist)), np.array(self.charge_hist)/max(self.charge_hist), label="BCharge")
            plt.plot(range(5), [capacity/max(self.charge_hist)]*5, label = "BCap")
            plt.plot(range(len(self.charge_hist)), np.array(self.price_hist)/(sum(self.price_hist)/len(self.price_hist)), label="Price")
            plt.plot(range(len(smoothed_action_hist)), smoothed_action_hist, label="Action")
            plt.plot(range(len(self.profit_hist)), self.profit_hist, label="ProfitDelta")
            plt.legend()
            plt.show()

        return None