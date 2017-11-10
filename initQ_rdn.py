# initiate Q value of different states
# state = [target_state, dis_1t, dis_1exit, steps, action]
# action = [move_to_tA, exit, wait_tA, wait_exit]

import numpy as np

Max_step = 25.0

if __name__ == '__main__':

    Q = np.zeros((3,10,10,8,25,4)) # target_state, dis_1t, dis_1exit, steps, action
    Q = np.array(Q) - Max_step
    
    # target_state = 0 or 2
    for a0t in range(10):
        for a1t in range(10):
            for a1e in range(8):
                for steps in range(25):
                    Q[0][a0t][a1t][a1e][steps][0] = 0 - Max_step
                    Q[0][a0t][a1t][a1e][steps][1] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                    Q[0][a0t][a1t][a1e][steps][2] = 0 - Max_step
                    Q[0][a0t][a1t][a1e][steps][3] = Max_step - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                    
                    Q[2][a0t][a1t][a1e][steps][0] = 0 - Max_step
                    Q[2][a0t][a1t][a1e][steps][1] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                    Q[2][a0t][a1t][a1e][steps][2] = 0 - Max_step
                    Q[2][a0t][a1t][a1e][steps][3] = Max_step - steps - a1e if steps + a1e < Max_step else 0 - Max_step
	
    # target_state = 1
    for a0t in range(10):
        for a1t in range(10):
            for a1e in range(8):
                for steps in range(25):
                    if a1t==0:                    
                        Q[1][a0t][a1t][a1e][steps][0] = 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][1] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][2] = Max_step - steps - a1t if steps + 1 < Max_step else 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][3] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                    else:
                        Q[1][a0t][a1t][a1e][steps][0] = Max_step - steps - a1t if steps + a1t < Max_step else 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][1] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][2] = 0 - Max_step
                        Q[1][a0t][a1t][a1e][steps][3] = 5 - steps - a1e if steps + a1e < Max_step else 0 - Max_step 
                                

					

    try:
        with open('rdn.txt','w') as f:

            for i in range(3):
                for j in range(10):
                    for k in range(10):
                        for l in range(8):
                            np.savetxt(f, Q[i][j][k][l], fmt='%-7.2f')
            
    except Exception as e:
        print('Unable to save the results: %s' % e)


        
    new_Q = np.loadtxt('rdn.txt')

    # Note that this returned a 2D array!
    print(new_Q.shape)

    new_Q = new_Q.reshape((3,10,10,8,25,4))
    assert np.all(new_Q == Q)

