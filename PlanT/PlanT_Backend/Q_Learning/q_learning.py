import gym
from gym import spaces
import numpy as np
import random

# Load POI Samples
pois = []

with open('locations.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row['id'] = int(row['id'])
        row['duration'] = int(row['duration'])
        row['latitude'] = float(row['latitude'])
        row['longitude'] = float(row['longitude'])
        row['tags'] = list(map(int, row['tags'].split('|'))
                           ) if row['tags'] else []
        pois.append(row)

# Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # radius of the earth (km)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Distance between POIs
distances = np.zeros((len(pois), len(pois)))
for i in range(len(pois)):
    for j in range(len(pois)):
        distances[i, j] = haversine(pois[i]['longitude'], pois[i]['latitude'],
                                    pois[j]['longitude'], pois[j]['latitude'])

def GetTravelTime(distance):
    speed_kmh = 50  # 50km/h
    speed_kpm = speed_kmh / 60  # distance traveled per minute (km)
    return distance / speed_kpm  # travel time (min)

def MinutesToTime(minutes):
    hours = int(minutes) // 60
    minutes = int(minutes) % 60
    return f"{hours:02d}:{minutes:02d}"

# Create Travel Environment
class CreateTravelEnv(gym.Env):
    def __init__(self, pois, distances, start_time=12*60, end_time=20*60):
        super(CreateTravelEnv, self).__init__()
        self.pois = pois
        self.distances = distances
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.visited = []
        self.current_location = random.choice([i for i in range(len(
            pois)) if pois[i]['category'] != 'accommodation'])  # start from a random non-accommodation POI
        self.restaurant_visits = 0
        self.action_space = spaces.Discrete(len(pois))
        self.observation_space = spaces.Box(
            low=0, high=len(pois)-1, shape=(1,), dtype=np.int32)
        self.last_reward = 0
        self.reward_reasons = []  # to explain the reasons of the rewards
        self.selected_tags = []

    def set_user_tags(self, selected_tags):
        self.selected_tags = selected_tags

    def reset(self):
        self.current_time = self.start_time
        self.visited = []
        self.current_location = random.choice([i for i in range(len(
            pois)) if pois[i]['category'] != 'accommodation'])  # start from a random non-accommodation POI
        self.visited.append(self.current_location)
        self.restaurant_visits = 0
        self.last_reward = 0
        self.reward_reasons = []
        return np.array([self.current_location])

    def step(self, action):
        done = False
        reward = 0
        reasons = []

        if self.pois[action]['category'] == 'accommodation' and self.current_time < self.end_time - 60:
            reasons.append("accommodation selected too early")
            reward = -10
            done = True
        elif action in self.visited or action >= len(pois):
            reasons.append("already visited or invalid action")
            reward = -10
            done = True
        elif self.pois[action]['category'] == 'restaurant' and self.restaurant_visits >= 3:
            reasons.append("too many restaurants visited")
            reward = -10
            done = True
        elif len(self.visited) > 0 and self.pois[action]['category'] == self.pois[self.visited[-1]]['category']:
            reasons.append("consecutive same category POI")
            reward = -10
            done = True
        else:
            travel_duration = GetTravelTime(
                self.distances[self.current_location, action])
            # hours to minutes
            visit_duration = self.pois[action]['duration'] * 60

            if self.current_time + travel_duration + visit_duration <= self.end_time:  # Check timeout
                self.current_time += travel_duration + visit_duration
                self.current_location = action
                self.visited.append(action)

                # Reward1. Every time visiting a new POI
                reward = 10
                reasons.append("POI Visit")

                # Reward2. Match the tags selected by the user
                if any(tag in self.selected_tags for tag in self.pois[action]['tags']):
                    reward += 20
                    reasons.append("Tag Match")

                if self.pois[action]['category'] == 'restaurant':  # check restaurants
                    self.restaurant_visits += 1

                # Reward3. Visiting Nearby POIs
                if len(self.visited) > 1:
                    prev_location = self.visited[-2]
                    if self.distances[prev_location, action] < 5:  # less than 5km
                        reward += 10
                        reasons.append("Nearby POI")

                # Reward4. Time efficiency (less travel time)
                if travel_duration < 10:
                    reward += 5
                    reasons.append("Efficient Travel Time")

                # Penalty for long travel times
                if travel_duration > 60:  # more than 1 hour
                    reward -= 15
                    reasons.append("Long Travel Time Penalty")

            else:
                reasons.append("time out")
                done = True

        # Ensure the final POI is an accommodation
        if done and self.pois[self.current_location]['category'] != 'accommodation' and self.current_time >= self.end_time - 60:
            accommodations = [i for i in range(
                len(pois)) if pois[i]['category'] == 'accommodation']
            closest_accommodation = min(
                accommodations, key=lambda acc: self.distances[self.current_location, acc])
            travel_duration = GetTravelTime(
                self.distances[self.current_location, closest_accommodation])
            if self.current_time + travel_duration <= self.end_time:
                self.current_time += travel_duration
                self.current_location = closest_accommodation
                self.visited.append(closest_accommodation)
                reward += 10
                reasons.append("Final Accommodation Visit")

        self.last_reward = reward  # update last reward
        self.reward_reasons = reasons  # update last reasons

        return np.array([self.current_location]), reward, done, {}

    def render(self):
        current_time_str = MinutesToTime(self.current_time)
        current_location_name = self.pois[self.current_location]['name']
        visited_names = [self.pois[i]['name'] for i in self.visited]

        print(
            f"Current Time: {current_time_str}, Reward: {self.last_reward} ({', '.join(self.reward_reasons)})")
        print(f"Visited POIs: {visited_names}")
        print(f"Current Location: {current_location_name}")

# Course Generation Function

def GenerateTravelCourse(selected_tags):
    env = CreateTravelEnv(pois, distances)
    env.set_user_tags(selected_tags)

    # Load the Q-table
    q_table = np.load('q_table.npy')

    total_reward = 0
    visited_pois = []

    while total_reward < 85:
        # print("************Travel Course************")
        state = env.reset()
        done = False
        total_reward = 0
        visited_pois = [pois[state[0]]['name']]

        while not done:
            # env.render()
            action = np.argmax(q_table[state[0]])
            next_state, reward, done, _ = env.step(action)
            if "time out" not in env.reward_reasons:
                if pois[action]['name'] not in visited_pois:
                    visited_pois.append(pois[action]['name'])
                total_reward += reward
            state = next_state

        # env.render()

        # Ensure the final POI is an accommodation
        if env.pois[state[0]]['category'] != 'accommodation':
            accommodations = [i for i in range(
                len(pois)) if pois[i]['category'] == 'accommodation']
            closest_accommodation = min(
                accommodations, key=lambda acc: env.distances[env.current_location, acc])
            # print('추가:', pois[closest_accommodation]['name'])
            next_state, reward, done, _ = env.step(closest_accommodation)
            if pois[closest_accommodation]['name'] not in visited_pois:
                visited_pois.append(pois[closest_accommodation]['name'])
            total_reward += reward
            state = next_state

        # print(f"Total Reward: {total_reward}")

    # Output the final current location and time
    final_location_name = env.pois[env.current_location]['name']
    final_time_str = MinutesToTime(env.current_time)

    if final_location_name not in visited_pois:
        visited_pois.append(final_location_name)

    return visited_pois


# User tags and result
user_selected_tags = [5]  # case 1: [1, 3], case 2: [5]
recommended_route = GenerateTravelCourse(user_selected_tags)
print("Recommended Travel Route:", recommended_route)
