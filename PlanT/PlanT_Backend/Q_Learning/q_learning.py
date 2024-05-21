import gym
from gym import spaces
import csv
import numpy as np
import random
from pathlib import Path

#ROOT
MODEL_ROOT = Path(__file__).parent

CSV_DIR = MODEL_ROOT/'locations.csv'
QTABLE_DIR = MODEL_ROOT/'q_table.npy'

# POI Sample
pois = []

with open(CSV_DIR, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row['id'] = int(row['id'])
        row['category'] = int(row['category'])
        row['duration'] = int(row['duration'])
        row['latitude'] = float(row['latitude'])
        row['longitude'] = float(row['longitude'])
        row['tags'] = list(map(int, row['tags'].split('|')))
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

class CreateTravelEnv(gym.Env):
    def __init__(self, pois, distances, start_time=10*60, end_time=20*60):
        super(CreateTravelEnv, self).__init__()
        self.pois = pois
        self.distances = distances
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.visited = []
        self.current_location = random.choice([i for i in range(len(pois)) if pois[i]['category'] != 3])  # start from a random non-accommodation POI
        self.restaurant_visits = 0
        self.action_space = spaces.Discrete(len(pois))
        self.observation_space = spaces.Box(low=0, high=len(pois)-1, shape=(1,), dtype=np.int32)
        self.last_reward = 0
        self.reward_reasons = []  # to explain the reasons of the rewards
        self.selected_tags = []

    def SetUserTags(self, selected_tags):
        self.selected_tags = selected_tags

    def reset(self):
        self.current_time = self.start_time
        self.visited = []
        self.current_location = random.choice([i for i in range(len(pois)) if pois[i]['category'] != 3])  # start from a random non-accommodation POI
        self.visited.append(self.current_location)
        self.restaurant_visits = 0
        self.last_reward = 0
        self.reward_reasons = []
        return np.array([self.current_location])

    def step(self, action):
        done = False
        reward = 0
        reasons = []

        # Prevent visiting accommodations except as the final action
        if self.pois[action]['category'] == 3 and self.current_time < self.end_time - 60:  # Accommodation only allowed as final action
            reasons.append("accommodation selected too early")
            reward = -10
            done = True
        elif action in self.visited or action >= len(pois):
            reasons.append("already visited or invalid action")
            reward = -10
            done = True
        elif self.pois[action]['category'] == 1 and self.restaurant_visits >= 3:
            reasons.append("too many restaurants visited")
            reward = -10
            done = True
        elif len(self.visited) > 0 and self.pois[action]['category'] == self.pois[self.visited[-1]]['category']:
            reasons.append("consecutive same category POI")
            reward = -10
            done = True
        else:
            travel_duration = GetTravelTime(self.distances[self.current_location, action])
            visit_duration = self.pois[action]['duration'] * 60  # hours to minutes

            if self.current_time + travel_duration + visit_duration <= self.end_time:  # Check timeout
                self.current_time += travel_duration + visit_duration
                self.current_location = action
                self.visited.append(action)

                # Reward1. Every time visiting a new POI
                reward = 10
                reasons.append("POI Visit")

                # Reward2. Match the tags selected by the user
                if any(tag in self.pois[action]['tags'] for tag in self.selected_tags):
                    reward += 20
                    reasons.append("Tag Match")

                if self.pois[action]['category'] == 1:  # check restaurants
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

        # final POI is an accommodation
        if done and self.pois[self.current_location]['category'] != 3 and self.current_time >= self.end_time - 60:
            accommodations = [i for i in range(len(pois)) if pois[i]['category'] == 3]
            closest_accommodation = min(accommodations, key=lambda acc: self.distances[self.current_location, acc])
            travel_duration = GetTravelTime(self.distances[self.current_location, closest_accommodation])
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

        print(f"Current Time: {current_time_str}, Reward: {self.last_reward} ({', '.join(self.reward_reasons)})")
        print(f"Visited POIs: {visited_names}")
        print(f"Current Location: {current_location_name}")


def GenerateTravelCourse(days, selected_tags):
    env = CreateTravelEnv(pois, distances)
    env.SetUserTags(selected_tags)

    # Load the Q-table
    q_table = np.load(QTABLE_DIR)

    itinerary = {}
    tag_scores = {tag: 0 for tag in selected_tags}

    for day in range(1, days + 1):
        daily_total_reward = 0
        while daily_total_reward < 85:
            # print("************Travel Course************")
            state = env.reset()
            done = False
            daily_route = [pois[state[0]]['id']]
            travel_times = [env.current_time+pois[state[0]]['duration']*60]
            itinerary_detail = [{
                'poi_id': pois[state[0]]['id'],
                'poi_name': pois[state[0]]['name'],
                'arrival_time': MinutesToTime(env.start_time),
                'departure_time': MinutesToTime(env.start_time + pois[state[0]]['duration'] * 60)
            }]
            daily_total_reward = 0
            env.current_time += pois[state[0]]['duration'] * 60

            while not done and env.current_time < env.end_time:
                action = np.argmax(q_table[state[0]])
                next_state, reward, done, _ = env.step(action)
                if "time out" not in env.reward_reasons:
                    poi_id = pois[action]['id']
                    poi_name = pois[action]['name']
                    arrival_time = MinutesToTime(env.current_time - pois[action]['duration'] * 60)
                    departure_time = MinutesToTime(env.current_time)
                    itinerary_detail.append({
                        'poi_id': poi_id,
                        'poi_name': poi_name,
                        'arrival_time': arrival_time,
                        'departure_time': departure_time
                    })
                    daily_route.append(poi_id)
                    travel_times.append(env.current_time - pois[action]['duration'] * 60)
                    travel_times.append(env.current_time)
                    daily_total_reward += reward
                state = next_state

            # final POI is an accommodation
            if env.pois[state[0]]['category'] != 3:
                # print("hotel")
                accommodations = [i for i in range(len(pois)) if pois[i]['category'] == 3]
                closest_accommodation = min(accommodations, key=lambda acc: env.distances[env.current_location, acc])
                next_state, reward, done, _ = env.step(closest_accommodation)

                end_duration = GetTravelTime(env.distances[env.current_location, closest_accommodation])
                env.current_time = travel_times[-1] + end_duration

                poi_id = pois[closest_accommodation]['id']
                poi_name = pois[closest_accommodation]['name']

                arrival_time = MinutesToTime(env.current_time)

                itinerary_detail.append({
                    'poi_id': poi_id,
                    'poi_name': poi_name,
                    'arrival_time': arrival_time,
                    'departure_time': None
                })
                daily_route.append(poi_id)
                travel_times.append(env.current_time)

                daily_total_reward += reward
            state = next_state

        for poi_id in daily_route:
            poi_tags = next(poi['tags'] for poi in pois if poi['id'] == poi_id)
            for tag in selected_tags:
                if tag in poi_tags:
                    tag_scores[tag] += 20

        formatted_travel_times = [MinutesToTime(time) for time in travel_times]
        itinerary[day] = [daily_route, formatted_travel_times, itinerary_detail, tag_scores]
    

    result = {day: itinerary[day] for day in range(1, days + 1)}

    return result

selected_tags = [1, 3] # [1, 3]은 호텔로 이동시간에 오류가 있고, [5]는 마지막 호텔 방문에서 오류가 있습니다
days = 1
recommended_itinerary = GenerateTravelCourse(days, selected_tags)
print(recommended_itinerary) # 시간 부분에서 혹시 전달 과정에서 오류가 있을까봐 시간 리스트와 태그 스코어 사이에 상세 정보를 추가했습니다!