import numpy as np
import random

# 다이아 5, 플레 2, 골드 1
dia = 5
platinum = 2
gold = 1

#공평한 점수
fair_score = 5

player_info = {
    '박준서' : ['다이아', '미드'],
    '김경민' : ['다이아', '서폿'],
    '김준엽' : ['플레', '정글'],
    '김용후' : ['골드', '원딜'],
    '김약갈' : ['골드', '미드'],
    '신현지' : ['다이아', '원딜']
}

Team_all = []
# 모든 구성원(player)들의 티어를 보고 점수로 바꿔줄거야.
for player_name in player_info.keys():
    Team_all.append(player_name)
    if player_info[player_name][0] == '다이아':
        player_info[player_name][0] = dia

    elif player_info[player_name][0] == '플레':
        player_info[player_name][0] = platinum
    else :
        player_info[player_name][0] = gold

print(f'Player list : {Team_all}')
print(f'Player info : {player_info}')

#무한루프
while True:
    random.shuffle(Team_all)
    Team_a, Team_b = Team_all[:int(len(Team_all)/2)], Team_all[int(len(Team_all)/2):]
    
    Team_a_value, Team_b_value = 0, 0 
    
    for a_player_name in Team_a:
        Team_a_value += player_info[a_player_name][0]
    for b_player_name in Team_b:
        Team_b_value += player_info[b_player_name][0]    

    #print(f'Team_a sum score : {Team_a_value}')
    #print(f'Team_b_sum score : {Team_b_value}')

    if np.abs(Team_a_value - Team_b_value) < fair_score:
        print(f'Final Team list : ')
        print(f'Team A : {Team_a}, Score : {Team_a_value}')
        print(f'Team B : {Team_b}, Score : {Team_b_value}')
        break
    
    