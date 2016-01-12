# Practical Machine learning
# Reinforcement learning - Q learning example 
# Chapter 12

immutable Experience
    s0::NNMatrix
    a0::Int64
    r0::Float64
    s1::NNMatrix
end

type DQN
    matrices::Array{NNMatrix,1}
    numStates::Int64 # length of the state vector (input vector)
    numHidden::Int64 # number of hidden nodes
    numActions::Int64 # number of actions (output vector)
    w1::NNMatrix
    b1::NNMatrix
    w2::NNMatrix
    b2::NNMatrix
    gamma::Float64 # reward discount
    epsilon::Float64 # epsilon-greedy policy
    alpha::Float64 # learning rate
    errorClamp::Float64
    expSize::Int64
    expAddProb::Float64
    expLearn::Int64
    expCnt::Int64
    experiences::Vector{Experience}
    s0::NNMatrix
    a0::Int64
    r0::Float64
    s1::NNMatrix
    a1::Int64
    lastG::Graph
    solver::Solver
    function DQN(numStates::Int64, numHidden::Int64, numActions::Int64;
                 std=0.02, gamma=0.75, epsilon=0.1, alpha=0.00001, errorClamp=5.0,
                 expSize=5000, expAddProb=0.05, expLearn=5)

        matrices = Array(NNMatrix, 0) # reference to matrices used by solver
        w1 = randNNMat(numHidden,  numStates, std);  push!(matrices, w1)
        b1 = randNNMat(numHidden,          1, std);  push!(matrices, b1)
        w2 = randNNMat(numActions, numHidden, std);  push!(matrices, w2)
        b2 = randNNMat(numActions,         1, std);  push!(matrices, b2)

        new(matrices, numStates, numHidden, numActions, w1, b1, w2, b2,
            gamma, epsilon, alpha, errorClamp, expSize, expAddProb, expLearn,0, Array(Experience,0),
            NNMatrix(numStates, 1), 0, typemin(Float64), NNMatrix(numStates, 1), 0, Graph(),Solver())
    end
end

function forward(m::DQN, s::NNMatrix, doBP::Bool=false)
    g = Graph(doBP)
    h0 =  add(g, mul(g, m.w1, s), m.b1)
    hd = tanh(g, h0) # hidden state
    a = add(g, mul(g, m.w2, hd), m.b2) # action vector
    m.lastG = g
    return a
end

function act(m::DQN, s::NNMatrix)
    a = rand()<=m.epsilon? rand(1:m.numActions):indmax(forward(m, s, false).w)
    return a # return selected action
end

function learnFromTuple(m::DQN, s0::NNMatrix, a0::Int64, r0::Float64, s1::NNMatrix)
    tmat = forward(m,s1,false)
    qmax = r0 + m.gamma * tmat.w[indmax(tmat.w)]

    pred = forward(m, s0, true)
    tdError = pred.w[a0,1] - qmax
#     println("tmat=$(tmat.w) max=$(indmax(tmat.w)) q=$qmax r0=$(r0) tMax=$(tmat.w[indmax(tmat.w)]) a0=$a0 pred=$(pred.w) tdError=$tdError")

    tdErrorClamp = minimum([maximum([tdError,-m.errorClamp]),m.errorClamp]) # huber loss to robustify
    pred.dw[a0,1] = tdErrorClamp
    backprop(m.lastG)
    solverstats = step(m.solver, m.matrices, m.alpha, 1e-06, m.errorClamp)

#     for k = 1:length(m.matrices)
#         @inbounds mat = m.matrices[k] # mat ref
#         @inbounds for j = 1:mat.d, i = 1:mat.n
#             mat.w[i,j] += - m.alpha * mat.dw[i]
#             mat.dw[i,j] = 0
#         end
#     end
    return tdErrorClamp
end

function learn(m::DQN, s0::NNMatrix, a0::Int64, r0::Float64, s1::NNMatrix)

    tdError = learnFromTuple(m, s0, a0, r0, s1)

    if rand() <= m.expAddProb
        m.expCnt = m.expCnt >= m.expSize? 1 : m.expCnt + 1
        if length(m.experiences) < m.expSize
            push!(m.experiences,Experience(s0, a0, r0, s1))
        else
            m.experiences[m.expCnt] = Experience(s0, a0, r0, s1)
        end
    end

    #　memory replay
    len = length(m.experiences)
    for i = 1:min(len,m.expLearn)
        r = rand(1:len)
        exp = m.experiences[r]
        err = learnFromTuple(m, exp.s0, exp.a0, exp.r0, exp.s1)
    end

    return tdError
end
