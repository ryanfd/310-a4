import numpy as np

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
        while iterId < nIterations:
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
            if epsilon < tolerance:
                break
        # end of while loop

        # print("Value Iteration Result:", [V,iterId,epsilon])
        
        return [V,iterId,epsilon]
    # end of valueIteration()

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

            # np.dot() -> product of 2 arrays
            curr_policy = self.R[s] + self.discount * np.dot(self.T[s][:][:], V)
            print("Curr_Policy:", curr_policy)

            # argmax policy
            curr_policy = np.argmax(curr_policy)

            policy[s] = curr_policy
        # end of for loop

        # policy = np.argmax(self.R + self.discount * np.dot(self.T, V), axis=0)
        print("Policy:", policy)

        print("Extract Policy Result:", policy)

        return policy 
    # end of extractPolicy()

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

        print("Evaluate Policy Result:", V)

        return V
    # end of evaluatePolicy()
        
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

            if (policy == new_policy).all(): # if old and new policy are equal, exit while loop
                break

            V = self.evaluatePolicy(new_policy)
            policy = new_policy # compare new policies to previous ones

            iterId += 1
        # end of while loop

        # print("Policy Iteration Result:", [policy,V,iterId])

        return [policy,V,iterId]
    # end of policyIteration()
            
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

        V = initialV
        policy_copy = policy.copy()
        int_policy = policy_copy.astype(int) # convert policy_copy to ints for indexing

        # out of bounds indexing issue
        for i in range(len(int_policy)):
            if int_policy[i] > 1:
                int_policy[i] = 1

        Tpi = self.T[int_policy, np.arange(self.nStates), :]
        Rpi = self.R[int_policy, np.arange(self.nStates)]
        iterId = 0
        epsilon = 0

        print("\nPARTIAL POLICY EVALUATIONS")
        print("---------------------------")
        print("initialV:", initialV)

        while iterId < nIterations:
            print("TPI:",Tpi)
            print("V:",V)

            new_V = Rpi + self.discount * np.dot(Tpi, V)
            print("new_V:", new_V)

            # calculate epsilon
            for i in range(len(new_V)):
                epsilon = abs(V[i] - new_V[i])
            print("epsilon:", epsilon)

            V = new_V

            iterId += 1
            if epsilon < tolerance:
                break
        # end of while loop

        print("Partial Result:", [V,iterId,epsilon])

        return [V,iterId,epsilon]
    # end of evaluatePolicyPartially()

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

        policy = initialPolicy
        V = initialV
        iterId = 0
        epsilon = 0

        while iterId < nIterations:
            temp_V = V
            [V, _, _] = self.evaluatePolicyPartially(policy, temp_V, nEvalIterations) # partial eval
            new_policy = self.extractPolicy(V) # policy improvement
            [new_V, _, _] = self.valueIteration(V, 1)

            print("V:", temp_V)
            print("Policy:", new_policy)
            print("new_V:", new_V)

            for i in range(len(new_V)):
                epsilon = abs(temp_V[i] - new_V[i])
            print("new epsilon:", epsilon)

            policy = new_policy
            V = new_V

            iterId += 1
            if epsilon < tolerance:
                break
        # end of while loop

        print("Modified Result:", [policy,V,iterId,epsilon])
        
        return [policy,V,iterId,epsilon]
    # end of modifiedPolicyIteration()
        