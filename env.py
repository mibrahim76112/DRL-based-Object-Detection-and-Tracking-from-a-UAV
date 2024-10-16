import numpy as np 
from numpy.linalg import norm
import math

def reward(st_prev, st, at, w):
    s2 = np.sign(norm([st_prev[0], st_prev[1]]) - norm([st[0], st[1]])) * w[1]
    s4 = np.sign(st_prev[2] - st[2]) * w[3]
    r = 1 - w[0] * norm([st[0], st[1]]) + s2 * norm([st[3], st[4]]) - w[2] * abs(st[2]) + s4 * abs(st[5]) - w[4] * norm([at[0], at[1]])
    return r


def distance(image_center,bbox_center):
    cx,cy=bbox_center
    w,h=image_center

    dist=math.sqrt((cx-w)**2+(cy-h)**2)
    return dist

class Env():
    def __init__(self,max_eps_steps,s_dim):
        self.max_ep_steps=max_eps_steps
        self.s_dim=s_dim
        self.w=[1,1,1,1,1]
    
    def step(self,prev_state,state,action,frame_center,bbox_center,check_done):
        cx,cy=bbox_center
        w,h=frame_center

        ex=cx-w
        ey=cy-h

        ea=distance(frame_center,bbox_center)

        p_ex,p_ey,p_ea,_,_,_=prev_state
        c_ex,c_ey,c_ea,_,_,_=state

        n_ex=c_ex-p_ex
        n_ey=c_ey-p_ey
        n_ea=c_ea-p_ea

        next_state=[ex,ey,ea,n_ex,n_ey,n_ea]
        if check_done:
            done=True
            rew=-100
        else:
            rew=reward(prev_state,state,action,self.w)
            done=False
        
        if abs(ex)<=30 and abs(ey)<=30:
            rew=10000
            done=True
        
        return rew,next_state,done
    
    def reset(self,frame_center,bbox_center):
        cx,cy=bbox_center
        w,h=frame_center

        ex=cx-w
        ey=cy-h

        ea=distance(frame_center,bbox_center)

        return [ex,ey,ea,0,0,0]

        

