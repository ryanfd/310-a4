import numpy as np
import math

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount

        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # set all of V = 0
        V = np.zeros(self.nStates)
        iterId = 0
        row_size = (len(self.T[0][0])-1)

        # converge on value
        while True:
            print("\nStep", iterId)
            print("--------------------------")

            v_temp = V.copy()
            epsilon = 0
            print("V_Temp:", v_temp)

            for (s, state) in enumerate(initialV):
                if s == len(V)-1: # at end of V
                    break
                print("State:", s)

                for a in range(len(self.T)):
                    print("Action:", a)

                    print("p = [", end="")
                    for i in range(len(self.T[a][s])):
                        print("", self.T[a][s][i], end=" ")
                        if (i + 1) % row_size == 0:
                            print("", end="")
                    print("]\n", end="\n")

                    # find neighbouring state based on current state and action
                    next_s = a+1
                    print("Next State:", next_s)

                    # P(s'|s, a)
                    p = self.T[a][s][next_s]

                    # initial reward
                    print("Immediate Reward =", self.R[a][s])

                    # bellman function
                    w = p*(self.R[a][s] + self.discount * v_temp[next_s])
                    sum_tuple = list()
                    sum_tuple.append(w)
                    sum_tuple = tuple(sum_tuple)
                    print("sum_tuple:", sum_tuple)
                # end of for loop

                V[s] = sum(sum_tuple)
                print("V =", V)

                epsilon = max(epsilon, abs(V[s] - v_temp[s]))
                print("Epsilon =", epsilon, "\n")
            # end of for loop

            iterId += 1
            if epsilon < tolerance or iterId >= nIterations:
                break
        # end of while loop
        
        return [V,iterId,epsilon]
    # end of ValueIteration()

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        policy = np.zeros(self.nStates)
        print("POLICY EXTRACTION")
        print("------------------")
        print("V:", V)
        print("Policy:", policy)

        for s in range(len(self.T)):
            curr_policy = np.zeros(len(self.T))

            curr_policy = self.R[s] + self.discount * np.dot(self.T[s][:][:], V)
            print("Curr_Policy:", curr_policy)

            # argmax policy
            curr_policy = np.argmax(curr_policy)

            policy[s] = curr_policy
        # end of for loop

        # policy = np.argmax(self.R + self.discount * np.dot(self.T, V), axis=0)
        print("Policy:", policy)

        return policy 
    # end of Extract Policy

    """
    BASED ON CODE FROM: https://gist.github.com/shivamkalra/cd0726fd37bb7c9f5e1bfd8fe45b26f1
    """
    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        print("\nEVALUATE POLICY")
        print("----------------")

        policy_copy = policy.copy()
        int_policy = policy_copy.astype(int) # convert policy_copy to ints for indexing

        # out of bounds indexing issue
        for i in range(len(int_policy)):
            if int_policy[i] > 1:
                int_policy[i] = 1
        
        V = np.zeros(self.nStates)
        Tpi = self.T[int_policy, np.arange(self.nStates), :]
        Rpi = self.R[int_policy, np.arange(self.nStates)]
        part_eval = np.identity(self.nStates) - self.discount * Tpi
        V = np.linalg.solve(part_eval, Rpi)

        print("T[pi]:\n", Tpi)
        print("R[pi]:\n", Rpi)
        print("V:", V)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy
        V = self.evaluatePolicy(policy)
        iterId = 0

        print("POLICY ITERATION")
        print("-----------------")

        while iterId < nIterations:
            print("\nStep", iterId)

            new_policy = self.extractPolicy(V)
            print("New Policy:", new_policy)

            if (policy == new_policy).all():
                break

            V = self.evaluatePolicy(new_policy)
            policy = new_policy

            iterId += 1
        # end of while loop

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [policy,V,iterId,epsilon]
        