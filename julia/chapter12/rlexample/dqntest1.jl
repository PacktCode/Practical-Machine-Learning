# Practical Machine learning
# Reinforcement learning - Q learning example 
# Chapter 12

using DeepQLearning, NNGraph
# reload("DeepQLearning")
xs = linspace(0,360,100)
ys = round(sin(deg2rad(xs)),3)
deg2rad(xs)
# plot(x=xs,y=ys)

m = DQN(2,100,2)
alpha=0.0001; t_alpha =0.15
epsilon = 0.2; t_epsilon =0.45

init = [0. 0.]
s0 = NNMatrix(init'); a0 = 1; r0 = 0.
t = 0
for epoch = 1:1000 #0000
    t += 1
    avgReward = 0
    m.epsilon = epsilon * 1/t^t_epsilon
    m.alpha = alpha * 1/t^t_alpha
    for i = 2:length(xs)
        x, x2, y = xs[i],xs[i-1], ys[i]
        s = [x x2]
        s1 = NNMatrix(s')
        a1 = act(m,s1)
        r1 = (a1==1?-1:1) * sign(y)
        avgReward += r1
        if i > 2 learn(m,s0,a0,r1,s1) end
        s0 = s1; a0 = a1; r0 = r1
    end
    avgReward = avgReward / (length(xs)-2)
    if epoch % 100 == 0 println("$t $epoch avgReward = $(round(avgReward,3))   m.alpha=$(round(m.alpha,6))  m.epsilon=$(round(m.epsilon,6))") end
end
