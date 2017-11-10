# initiate Q value of different states
# state = [target_state, dis_0t, dis_1t, dis_1exit, steps, action]
# action = [move_to_tA, exit, wait_tA, wait_exit]

import numpy as np

Max_step = 25.0
MOVEP = [0.0445, 0.4168]

if __name__ == '__main__':

    Q = np.zeros((3,10,10,8,25,4)) # target_state, dis_0t, dis_1t, dis_1exit, steps, action
    Q = np.array(Q) - Max_step
    
    # target_state = 0
    for a0t in range(10):
        for a1t in range(10):
            for a1e in range(8):
                for steps in range(25):
                    catch = Max_step - steps - max(a0t, a1t) -1 if steps + max(a0t, a1t) + 1 <= Max_step else 0 - Max_step
                    ex = 5 - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    pig_ex = Max_step - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    no_move = [(1-MOVEP[0])**(Max_step-steps), (1-MOVEP[0])**(Max_step-steps-1)*(1-MOVEP[1])]
                    Q[0][a0t][a1t][a1e][steps][0] = no_move[0]*(0 - Max_step) + (1 - no_move[0])*catch
                    Q[0][a0t][a1t][a1e][steps][1] = no_move[0]*ex + (1 - no_move[0])*(0.25**a1e)*pig_ex
                    Q[0][a0t][a1t][a1e][steps][2] = catch
                    Q[0][a0t][a1t][a1e][steps][3] = no_move[1]*ex + (1 - no_move[1])*(0.25**a1e)*pig_ex
	
    # target_state = 1
    for a0t in range(10):
        for a1t in range(10):
            for a1e in range(8):
                for steps in range(25):
                    catch = Max_step - steps - a1t if steps + a1t <= Max_step else 0 - Max_step
                    ex = 5 - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    pig_ex = Max_step - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    no_move = [(1-MOVEP[0])**(Max_step-steps), (1-MOVEP[0])**(Max_step-steps-1)*(1-MOVEP[1])]
                    if a1t==0:                    
                        Q[1][a0t][a1t][a1e][steps][0] = no_move[0]*(0 - Max_step) + (1 - no_move[0])*catch
                        Q[1][a0t][a1t][a1e][steps][1] = no_move[0]*ex + (1 - no_move[0])*(0.25**a1e)*pig_ex
                        Q[1][a0t][a1t][a1e][steps][2] = catch 
                        Q[1][a0t][a1t][a1e][steps][3] = no_move[1]*ex + (1 - no_move[1])*(0.25**a1e)*pig_ex
                    else:
                        Q[1][a0t][a1t][a1e][steps][0] = catch 
                        Q[1][a0t][a1t][a1e][steps][1] = no_move[0]*ex + (1 - no_move[0])*(0.25**a1e)*pig_ex
                        Q[1][a0t][a1t][a1e][steps][2] = no_move[1]*catch + (1 - no_move[1])*(0 - Max_step)
                        Q[1][a0t][a1t][a1e][steps][3] = no_move[1]*ex + (1 - no_move[1])*(0.25**a1e)*pig_ex

    # target_state = 2
    for a0t in range(10):
        for a1t in range(10):
            for a1e in range(8):                
                for steps in range(25): 
                    catch = Max_step - steps - a1t if steps + a1t <= Max_step else 0 - Max_step
                    ex = 5 - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    pig_ex = Max_step - steps - a1e if steps + a1e <= Max_step else 0 - Max_step
                    no_move = [(1-MOVEP[0])**(Max_step-steps), (1-MOVEP[0])**(Max_step-steps-1)*(1-MOVEP[1])]                   
                    Q[2][a0t][a1t][a1e][steps][0] = catch
                    Q[2][a0t][a1t][a1e][steps][1] = no_move[0]*ex + (1 - no_move[0])*(0.25**a1e)*pig_ex
                    Q[2][a0t][a1t][a1e][steps][2] = no_move[1]*catch + (1 - no_move[1])*(0 - Max_step)
                    Q[2][a0t][a1t][a1e][steps][3] = no_move[1]*ex + (1 - no_move[1])*(0.25**a1e)*pig_ex					

    try:
        with open('astarnew.txt','w') as f:

            for i in range(3):
                for j in range(10):
                    for k in range(10):
                        for l in range(8):
                            np.savetxt(f, Q[i][j][k][l], fmt='%-7.2f')
            
    except Exception as e:
        print('Unable to save the results: %s' % e)


        
    new_Q = np.loadtxt('astar.txt')

    # Note that this returned a 2D array!
    print(new_Q.shape)

##    new_Q = new_Q.reshape((3,10,10,8,25,4))
##    assert np.all(new_Q == Q)

