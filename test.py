from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from flappybird import Agent,Model,Model2,Model3,DQN,preprocess
import cv2
import time
game = FlappyBird()
env = PLE(game, fps=30, display_screen=True) 
env.init()
# env.reset_game()
actionset =  env.getActionSet()
state = env.getGameState()
action_dim = len(actionset)  
obs_shape = len(state)  
print(action_dim,obs_shape)
print(actionset,state)
# 根据parl框架构建agent
model = Model(act_dim=action_dim)
# model = Model2(act_dim=action_dim)
# model = Model3(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=0.99, lr=0.001)
agent = Agent(
    algorithm,
    obs_dim=obs_shape,
    act_dim=action_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=0)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
save_path = '.\model_dir\model_6700_2823.0.ckpt' #episode_reward: 1785.0
agent.restore(save_path)

obs = list(env.getGameState().values())
# #处理obs
# obs = preprocess(obs)
episode_reward = 0
while True:
    # 预测动作，只选最优动作
    action = agent.predict(obs)
    # 图像太快休眠
    # time.sleep(0.02)
    # # 新建窗口显示分数
    # observation = env.getScreenRGB()
    # score  = env.score()
    # # 格式转换
    # observation = cv2.cvtColor(observation,cv2.COLOR_RGB2BGR)
    # # 选择90度
    # observation = cv2.transpose(observation)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # observation = cv2.putText(observation, "score:"+str(int(score)), (0, 30), font, 0.6, (0, 0, 255), 2)
    # cv2.imshow("flappybird", observation)
    # cv2.waitKey(10)
    
    reward= env.act(actionset[action])
    obs = list(env.getGameState().values())
    # #处理obs
    # obs = preprocess(obs)
    done = env.game_over()
    episode_reward += reward
    if done:
        break
print("episode_reward:",episode_reward)
cv2.destroyAllWindows()
