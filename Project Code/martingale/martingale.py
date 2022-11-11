""""""  		  	   		  	  			  		 			     			  	 
"""Assess a betting strategy.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Student Name: Rimon Mikheal 		  	   		  	  			  		 			     			  	 
GT User ID: rmikhael3	  	   		  	  			  		 			     			  	 
GT ID: 903737444 		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def author():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
    :rtype: str  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return "rmikhael3"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def gtid():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT ID of the student  		  	   		  	  			  		 			     			  	 
    :rtype: int  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return 903737444  # replace with your GT ID number
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			     			  	 
    :type win_prob: float  		  	   		  	  			  		 			     			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			     			  	 
    :rtype: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    result = False  		  	   		  	  			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			     			  	 
        result = True  		  	   		  	  			  		 			     			  	 
    return result  		  	   		  	  			  		 			     			  	 


def Stratgey(prob, goal, episodes):
    spins = 1000
    winnings_amount = np.full((episodes, spins), 80, dtype=np.int)
    for episode in range(episodes):
        cur_spin_num = 0
        episode_winnings = 0
        bet_amount = 1
        while (episode_winnings < goal) and (cur_spin_num < spins):
            won = False
            won = get_spin_result(prob)
            if won:
                episode_winnings = episode_winnings + bet_amount
                bet_amount = 1
                cur_spin_num += 1
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
                cur_spin_num += 1
            winnings_amount[episode][cur_spin_num-1] = episode_winnings
    return winnings_amount.T

def StratgeyWithLimit(prob, goal, episodes):
    loss_counter = 0
    spins = 1000
    winnings_amount = np.full((episodes, spins), goal, dtype=np.int)
    for episode in range(episodes):
        cur_spin_num = 0
        episode_winnings = 0
        bet_amount = 1
        while (episode_winnings < goal) and (cur_spin_num < spins) and (episode_winnings > -256):
            won = False
            won = get_spin_result(prob)
            if won:
                episode_winnings = episode_winnings + bet_amount
                bet_amount = 1
                cur_spin_num += 1
            else:
                episode_winnings = episode_winnings - bet_amount
                if (episode_winnings + 256) < (bet_amount*2):
                    bet_amount = episode_winnings + 256
                else:
                    bet_amount = bet_amount * 2
                cur_spin_num += 1
            winnings_amount[episode][cur_spin_num-1] = episode_winnings
        if episode_winnings == -256:
            winnings_amount[episode][cur_spin_num:] = -256
            loss_counter += 1
            print(loss_counter)

    return winnings_amount.T


def experiments(prob):
    #EXPERIMENT-1
    #Figure 1 --> All Winnings
    #Transposed because i dont want to show all 1000 plot only the episode
    #Added alpha per https://edstem.org/us/courses/16631/discussion/1019604?answer=2322184
    fig1 = Stratgey(prob, 80, 10)
    df1 = pd.DataFrame(fig1)
    ax1 = df1.plot(title="Winnings of 10 episodes", alpha=0.2)
    #plt.show()
    ax1.set_xlim(0, 300)
    ax1.set_ylim(-256, 100)
    ax1.set_ylabel("Expected winnings")
    ax1.set_xlabel("Spin number")
    #plt.show()
    plt.savefig("images/figure1.png")
    plt.clf()

    #Figure 2 --> Mean
    fig2 = Stratgey(prob, 80, 1000)
    fig2Mean = np.mean(fig2, axis=1)
    fig2Std = np.std(fig2, axis=1)
    df2 = pd.DataFrame(np.array([
        fig2Mean,
        (fig2Mean + fig2Std),
        (fig2Mean - fig2Std)
    ]))
    ax2 = df2.ix[0].plot(title="Winnings of 1000 episodes", color="black", style="-")
    df2.ix[1].plot(color="blue")
    df2.ix[2].plot(color="blue")
    ax2.set_xlim(0, 300)
    ax2.set_ylim(-256, 100)
    ax2.set_ylabel("Expected winnings")
    ax2.set_xlabel("Spin number")
    plt.legend(["Mean", "Mean/Standard deviation"])
    #plt.show()
    plt.savefig("images/figure2.png")
    plt.clf()


    #Figure 3 --> Median
    fig3Median = np.median(fig2, axis=1)
    df3 = pd.DataFrame(np.array([
        fig3Median,
        (fig3Median + fig2Std),
        (fig3Median - fig2Std)
    ]))
    ax3 = df3.ix[0].plot(title="Winnings of 1000 episodes", color="black", style="-")
    df3.ix[1].plot(color="blue")
    df3.ix[2].plot(color="blue")
    ax3.set_xlim(0, 300)
    ax3.set_ylim(-256, 100)
    ax3.set_ylabel("Expected winnings")
    ax3.set_xlabel("Spin number")
    plt.legend(["Median", "Median/Standard deviation"])
    #plt.show()
    plt.savefig("images/figure3.png")
    plt.clf()


    #EXPERIMENT-2
    #Figure 4 --> Mean/Bank
    fig4 = StratgeyWithLimit(prob, 80, 1000)
    fig4Mean = np.mean(fig4, axis=1)
    fig4Std = np.std(fig4, axis=1)
    df4 = pd.DataFrame(np.array([
        fig4Mean,
        (fig4Mean + fig4Std),
        (fig4Mean - fig4Std)
    ]))
    ax4 = df4.ix[0].plot(title="Winnings of 1000 episodes With Bank", color="black", style="-")
    df4.ix[1].plot(color="blue")
    df4.ix[2].plot(color="blue")
    ax4.set_xlim(0, 300)
    ax4.set_ylim(-256, 100)
    ax4.set_ylabel("Expected winnings")
    ax4.set_xlabel("Spin number")
    plt.legend(["Mean", "Mean/Standard deviation with Bank"])
    #plt.show()
    plt.savefig("images/figure4.png")
    plt.clf()

    #Figure 5 --> Median/Bank
    fig5Median = np.median(fig4, axis=1)
    df5 = pd.DataFrame(np.array([
        fig5Median,
        fig5Median + fig4Std,
        fig5Median - fig4Std
    ]))
    ax5 = df5.ix[0].plot(title="Winnings of 1000 episodes With Bank", color="black", style="-")
    df5.ix[1].plot(color="blue")
    df5.ix[2].plot(color="blue")
    ax5.set_xlim(0, 300)
    ax5.set_ylim(-256, 100)
    ax5.set_ylabel("Expected winnings")
    ax5.set_xlabel("Spin number")
    plt.legend(["Median", "Median/Standard deviation with Bank"])
    #plt.show()
    plt.savefig("images/figure5.png")
    plt.clf()



def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Method to test your code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    win_prob = (18 / 38)  # Black is 18 numbers of the total 38
    np.random.seed(gtid())  # do this only once  		  	   		  	  			  		 			     			  	 
    print(get_spin_result(win_prob))  # test the roulette spin


    # add your code here to implement the experiments
    experiments(win_prob)


if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
