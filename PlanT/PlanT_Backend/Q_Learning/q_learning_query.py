import csv
import numpy as np
import gym
from gym import spaces
import numpy as np
import random
import pandas as pd
from pathlib import Path
import os
import ast
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

#ROOT
MODEL_ROOT = Path(__file__).parent

CSV_DIR = MODEL_ROOT/'final_locations.csv'
QTABLE_DIR = MODEL_ROOT/'q_table.npy'
DIST_DIR = MODEL_ROOT/'poi_distances.npy'

# Load POIs
pois = []

with open('final_locations.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row['id'] = int(row['id'])
        row['category'] = int(row['category'])
        row['duration'] = int(row['duration'])
        row['latitude'] = float(row['latitude'])
        row['longitude'] = float(row['longitude'])
        row['tags'] = ast.literal_eval(row['tags'])  # 문자열을 리스트로 변환
        row['tags'] = [int(tag) for tag in row['tags']]
        pois.append(row)

# Load Distance between POIs
distances = np.load('poi_distances.npy')

# Search POIs
class FaissAsRetriever:
    def __init__(self, db_path, csv_path):
        self.db_path = db_path
        self.csv_path = csv_path
        # Add some infomation in metadata
        self.metadata_columns = ['id', 'category', 'duration']
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli', # 임시 임베딩 모델
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
            )
        self.allow_dangerous_deserialization = True
        self.vectorstore = None
        self.encoding = 'utf-8'
    
    def load_or_create_vectorstore(self):
        """vectorstore가 로컬에 존재하면 load하고, 존재하지 않으면 새로 생성합니다."""
        if os.path.exists(self.db_path):
            self._load_vectorstore()
        else:
            self._create_and_save_vectorstore()
    
    def _load_vectorstore(self):
        self.vectorstore = FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=self.allow_dangerous_deserialization)
        print("Loaded existing FAISS database from local storage.")
    
    def _create_and_save_vectorstore(self):
        loader = CSVLoader(self.csv_path, metadata_columns=self.metadata_columns, encoding=self.encoding)
        docs = loader.load()
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.vectorstore.save_local(self.db_path)
        print("Created and saved new FAISS database to local storage.")
    
    def search(self, query):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not loaded. Call load_or_create_vectorstore() first.")
        # Create custom retriever instance
        # retriever = self.vectorstore.as_retriever(search_type='mmr', search_kwargs= {'k': 30, 'fetch_k': 100, 'lambda_mult': 0.1})
        # retriever = self.vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs= {'k': 30, 'score_threshold': 0.5})
        trendy_pois = self.vectorstore.similarity_search_with_relevance_scores(query, k=30)
        # trendy_pois = retriever.invoke(query)

        return trendy_pois

    def faissRetriever(query):
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        # db_path = os.path.join(current_directory, 'vectorstore/faiss')
        # csv_path = os.path.join(current_directory, 'total_poi_vectordb.csv')
        db_path = 'vectorstore/faiss'
        csv_path = 'final_locations.csv'
        
        vectorstore_manager = FaissAsRetriever(db_path, csv_path)
        vectorstore_manager.load_or_create_vectorstore()
        
        results = vectorstore_manager.search(query)
        
        results_lst = []
        id_lst = []
        metadata_lst = []
        for i in results:
            # id_lst.append(i.metadata['id'])
            # metadata_lst.append(i.metadata)
            content = i[0].page_content
            metadata = i[0].metadata
            metadata['score'] = i[1]
            results_lst.append(content)
            id_lst.append(metadata['id'])
            metadata_lst.append(metadata)

        # poi_info = {
        #     "id": id_lst,
        #     "metadata": metadata_lst,
        # }

        return id_lst

def GetTravelTime(distance):
    speed_kmh = 30  # 30km/h (직선거리를 고려하여 이동 속도 느리게함)
    speed_kpm = speed_kmh / 60  # distance traveled per minute (km)
    return distance / speed_kpm  # travel time (min)

def MinutesToTime(minutes):
    hours = int(minutes) // 60
    minutes = int(minutes) % 60
    return f"{hours:02d}:{minutes:02d}"

class CreateTravelEnv(gym.Env):
    def __init__(self, pois, distances, start_time=11*60, end_time=22*60, poi_trend=set()):
        super(CreateTravelEnv, self).__init__()
        self.pois = pois
        self.distances = distances
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.visited = []
        self.restaurant_visits = 0
        self.poi_trend = poi_trend

        # 에이전트가 현재 위치한 poi의 인덱스를 저장하는 변수
        self.current_location = random.choice([i for i in range(len(pois)) if pois[i]['category'] != 3])  # 비숙박 장소 중 임의로 시작 위치 선택

        # 가능한 모든 행동(action)의 집합
        # spaces.Discrete(n): 이산적인 행동 공간으로 0부터 n-1 까지의 정수 값을 가질 수 있음
        # 에이전트가 선택할 수 있는 poi가 len(poi)개 존재
        # action은 에이전트가 특정 명소를 방문하는 것
        self.action_space = spaces.Discrete(len(pois))
        
        # 에이전트가 관찰할 수 있는 가능한 모든 상태(state)의 집합
        # spaces.Box(low, high, shape, dtype): 연속적인 값의 범위를 가지는 상태 공간을 정의
        # low: 관찰할 수 있는 값의 최솟값
        # high: 관찰할 수 있는 값의 최댓값
        # shape: 상태 공간의 형상(차원) -> 여기서는 (1,)로, 1차원 배열로 상태를 나타냄
        # dtype: 값의 데이터 타입 -> 여기서는 np.int32로 정수형 상태 값을 나타냄
        # state는 에이전트가 현재 위치한 poi의 인덱스
        
        # self.observation_space = spaces.Box(low=0, high=len(pois)-1, shape=(1,), dtype=np.int32) # 가능한 상태 공간 (POI 인덱스)
        self.observation_space = spaces.Discrete(len(pois))  # 가능한 상태 공간 (POI 인덱스)

        self.last_reward = 0
        self.reward_reasons = []
    
    
    def reset(self, visited_pois=None):
        self.current_time = self.start_time
        # self.visited = []
        self.visited = visited_pois if visited_pois else []
        self.current_location = random.choice([i for i in range(len(pois)) if pois[i]['category'] != 3 and i not in self.visited])  # start from a random non-accommodation POI
        self.visited.append(self.current_location)
        self.restaurant_visits = 0
        self.last_reward = 0
        self.reward_reasons = []
        return np.array([self.current_location])
    
    def step(self, action): # action=선택된 poi의 index
        done = False
        reward = 0
        reasons = []

        # Penalty 1: Only visit accommodation at the end
        if self.pois[action]['category'] == 3 and self.current_time < self.end_time - 30:  # 종료시간 30분 전
            reasons.append("accommodation selected too early")
            # print(reasons)
            reward = -10
            done = True # 숙소 가면 종료
        
        # Penalty 2: Only visit once
        elif action in self.visited or action >= len(pois):
            reasons.append("already visited or invalid action")
            # print(reasons)
            reward = -10
            done = True
        
        # Penalty 3: Visit restaurants at most 3 times
        elif self.pois[action]['category'] == 1 and self.restaurant_visits >= 3:
            reasons.append("too many restaurants visited")
            # print(reasons)
            reward = -10
            done = True
        
        # Penalty 4: Don't visit same category in a row
        elif len(self.visited) > 0 and self.pois[action]['category'] == self.pois[self.visited[-1]]['category']:
            reasons.append("consecutive same category POI")
            # print(reasons)
            reward = -10
            done = True
        
        else:
            travel_duration = GetTravelTime(self.distances[self.current_location, action])
            visit_duration = self.pois[action]['duration'] * 60  # hours to minutes

            if self.current_time + travel_duration + visit_duration <= self.end_time:  # Check timeout
                self.current_location = action # 도착 장소로 업데이트
                self.visited.append(action)
                # print('현재 상태 poi------>', self.current_location)

                self.current_time += travel_duration
                # print('Arrival time:',  MinutesToTime(self.current_time)) # 도착 시간 

                self.current_time += visit_duration
                # print('Departure time:',  MinutesToTime(self.current_time)) # 도착 시간 

                if self.pois[action]['category'] == 1:  # check restaurants
                    self.restaurant_visits += 1

                # Reward 1. Every time visiting a new POI
                reward = 10
                reasons.append("POI Visit")

                # Reward 2. Match the theme selected by the user
                if self.pois[action]['id'] in self.poi_trend:
                    reward += 50
                    reasons.append("Theme Match")
                
                # Reward 3. Visiting Nearby POIs
                if len(self.visited) > 1:
                    prev_location = self.visited[-2]
                    if self.distances[prev_location, action] < 5:  # less than 5km
                        reward += 15
                        reasons.append("Nearby POI")

                    elif self.distances[prev_location, action] > 15:  # more than 15km
                        reward -= 15
                        reasons.append("Long Distance POI")
            
            else:
                reasons.append("time out")
                # print(reasons)
                done = True
        
        # Ensure the final POI is an accommodation
        if done and self.pois[self.current_location]['category'] != 3:
            # print(self.current_location)
            accommodations = [i for i in range(len(pois)) if pois[i]['category'] == 3]
            closest_accommodation = min(accommodations, key=lambda acc: self.distances[self.current_location, acc])
            travel_duration = GetTravelTime(self.distances[self.current_location, closest_accommodation])
            self.current_time += travel_duration
            self.current_location = closest_accommodation
            # print('***', self.current_location)
            self.visited.append(closest_accommodation)
            reward += 10
            reasons.append("Final Accommodation Visit")
        
        self.last_reward = reward
        self.reward_reasons = reasons


        return np.array([self.current_location]), reward, done, {}

    def render(self):
        current_time_str = MinutesToTime(self.current_time)
        current_location_name = self.pois[self.current_location]['name']
        current_location_id = self.pois[self.current_location]['id']
        visited_names = [self.pois[i]['name'] for i in self.visited]

        print(f"Current Time: {current_time_str}, Reward: {self.last_reward} ({', '.join(self.reward_reasons)})")
        print(f"Visited POIs: {visited_names}")
        print(f"Current Location: {current_location_name} {current_location_id}")

def GenerateTravelCourse(days, poi_trend):
    env = CreateTravelEnv(pois, distances, poi_trend=poi_trend)

    q_table = np.load('q_table.npy')
    # print(q_table)

    itinerary = {} # travel plan
    visited_pois = set()

    for day in range(1, days + 1):
        # print("************New Day************")
        trend_score = 0
        daily_total_reward = 0

        threshold = 120
        cnt = 0

        while daily_total_reward < threshold:
            if cnt > 1000:
                threshold -= 10
                cnt = 0            
            cnt += 1
            # print("************Travel Course************")
            # print(visited_pois)
            daily_total_reward = 0
            travel_times = []
            done = False
            state = env.reset(visited_pois=list(visited_pois)) # initialize environment and choose first poi


            # print('현재 상태 poi------>', state)
            # env.render()

            daily_route = [pois[state[0]]['id']] # 시작 poi 넣기

            itinerary_detail = [{
                'poi_id': pois[state[0]]['id'],
                'poi_name': pois[state[0]]['name'],
                'arrival_time': MinutesToTime(env.start_time),
                'departure_time': MinutesToTime(env.start_time + pois[state[0]]['duration'] * 60)
            }]

            env.current_time += pois[state[0]]['duration'] * 60 # 체류시간 반영 후 업데이트
            travel_times.append(env.current_time) # 시작 poi 떠나는 시간

            # print('Departure time:', MinutesToTime(env.current_time))

            while not done and env.current_time < env.end_time: # 종료 조건 확인
                action = np.argmax(q_table[state[0]]) # 현재 state에서 가장 높은 Q 값(기대 장기적 보상)을 가지는 action 선택
                # print('try:', action)
                
                next_state, reward, done, _ = env.step(action) # 상태 update
                # print('current:', next_state[0], reward, done)
                # env.render()

                poi_id = pois[next_state[0]]['id']
                poi_name = pois[next_state[0]]['name']

                if not pois[next_state[0]]['category'] == 3:
                    arrival_time = MinutesToTime(env.current_time - pois[next_state[0]]['duration'] * 60)
                    departure_time = MinutesToTime(env.current_time)
                    travel_times.append(env.current_time - pois[next_state[0]]['duration'] * 60) # 도착시간
                    travel_times.append(env.current_time) # 떠나는 시간
                else:
                    arrival_time = MinutesToTime(env.current_time)
                    departure_time = None
                    travel_times.append(env.current_time)

                itinerary_detail.append({
                        'poi_id': poi_id,
                        'poi_name': poi_name,
                        'arrival_time': arrival_time,
                        'departure_time': departure_time
                    })
                
                daily_route.append(poi_id)

                daily_total_reward += reward
                # print('total reward:', daily_total_reward)
                state = next_state     
        
        formatted_travel_times = [MinutesToTime(time) for time in travel_times]

        for poi_id in daily_route:
            if poi_id in poi_trend:
                trend_score += 50
            visited_pois.add(poi_id)

        itinerary[day] = [daily_route, formatted_travel_times, itinerary_detail, trend_score]
    
    result = {day: itinerary[day] for day in range(1, days + 1)}

    return result


def RetrieveAndGenerate(days, method, content):
  poi_trend = []
  if method == 'tag':
     selected_tags = list(content)
     for poi in pois:
        if any(tag in selected_tags for tag in poi['tags']):
            poi_trend.append(poi['id'])
  
  elif method == 'query': # method == 'query'
    poi_result = FaissAsRetriever.faissRetriever(str(content)) # query 넣으면 해당되는 poi id 리스트가 반환됩니다.
    # print(poi_result)
    for id in poi_result:
      for row in pois:
          if row['id'] == int(id) and row['category'] == 3: # 숙소 제거
              break
      else:
          poi_trend.append(int(id))
  
  else:
    print("wrong method")

  recommended_route = GenerateTravelCourse(days, poi_trend)
  return recommended_route

# Output
days = 1
method = 'query'
content = "탄소 중립을 실천하는 친환경 여행" #query example
recommended_route = RetrieveAndGenerate(days, method, content)