# operations-research
for I in csvfiles:
    try:
        import pandas as pd
        from datetime import datetime
        import functools



        import pandas as pd
        import numpy as np
        t='/content/'+I
        a= pd.read_csv(t)
        n_trucks = round(sum(a['Weight'])//45000) + 1
        weight= a['Weight'].to_list()
        capacity = []
        for i in range(2*n_trucks):
          capacity.append(0)
        for i in weight:
          capacity.append(i)
        cap_datframe = pd.DataFrame(capacity,columns=["Capacity"])
        time_capacity= a['time_limit'].to_list()
        time_cap = []
        for i in range(2*n_trucks):
          time_cap.append(0)
        for i in time_capacity:
          time_cap.append(i)
        time_cap_dataframe  = pd.DataFrame(time_cap,columns=["Time_Capacity"])
        city = a['Origin City'].to_list()
        city_1=[]
        for i in range(2*n_trucks):
          city_1.append("North Bergen")
        city_1= city_1+city
        city_df = pd.DataFrame(city_1)
        new={'Origin City':'North Bergen','Origin State':'NJ'}
        a = a.append(new, ignore_index=True)
        Order_Line_Data_Werner = a

        locations = pd.read_csv('Locations_Werner.csv', index_col=0)

        locations = locations.drop_duplicates(keep='first')

        locations.rename(columns={'ID': 'ADDR_SRC_ID'}, inplace=True)

        Travel_Matrix = pd.read_csv('Travel_Matrix_Werner.csv', index_col=0)

        Order_Line_Data_Werner = Order_Line_Data_Werner.merge(locations, how='left',

                                                                      left_on=['Origin City', 'Origin State'],

                                                                      right_on=['city', 'state'])

        Order_Line_Data_Werner.drop(columns=['city', 'state', 'locations'], inplace=True)

        Order_Line_Data_Werner.rename(columns={'ADDR_SRC_ID': 'Pick_Up_Location_Id'}, inplace=True)

        Order_Line_Data_Werner = Order_Line_Data_Werner.merge(locations, how='left',

                                                                      left_on=['Destination City', 'Destination State'],

                                                                      right_on=['city', 'state'])

        Order_Line_Data_Werner.drop(columns=['city', 'state', 'locations'], inplace=True)

        Order_Line_Data_Werner.rename(columns={'ADDR_SRC_ID': 'Delivery_Location_Id'}, inplace=True)


        Order_Line_Data_Werner['Order_Number'] = list(range(1, (len(Order_Line_Data_Werner) + 1)))
        Order_Line_Data_Werner.set_index(['Order_Number'], drop = True, inplace = True)


        pick_up_loc = Order_Line_Data_Werner['Pick_Up_Location_Id'].tolist()

        From,To,Dist=[],[],[]

        for  i  in pick_up_loc:
          for j in pick_up_loc:
            l=str(j)
            From.append(i)
            To.append(l)
            Dist.append(Travel_Matrix.loc[i,l])

        distance_1D = pd.DataFrame(From,  columns=['From'])

        distance_1D['To'] = To

        distance_1D['Distance']=Dist

        len(From)

        len(set(From))

        t,u=[],[]

        for  i  in range(len(pick_up_loc)):
          for j in range(len(pick_up_loc)):
            
            t.append(i)
            u.append(j)

        distance_1D['From']=t
        distance_1D['To']=u

        pist = distance_1D.pivot(index='From', columns='To', values='Distance')






        terminal = pist[len(pick_up_loc)-1].tolist()

        terminal

        terminal_distance = terminal

        df2 = pd.DataFrame(terminal)

        df2  = df2[:-1]

        df2 = df2.T

        df2

        terminals=[]

        for i in range(2*n_trucks):
          terminals.append(0)
        for i in terminal:
          terminals.append(i)



        terminals = terminals[:-1]

        terminal_distance = pd.DataFrame(terminals)

        terminal_df = terminal_distance[(2*n_trucks):]

        terminal_df=terminal_df[np.repeat(terminal_df.columns.values,(2*n_trucks))]

        terminal_df

        terminal_cols = []

        for i in range(2*n_trucks):
          terminal_cols.append(i)

        terminal_df.columns=terminal_cols

        terminal_df

        t_distance = terminal_distance.T

        t_distance

        t_distance =t_distance.loc[t_distance.index.repeat(2*n_trucks)]

        t_distance_index = []

        for i in range(2*n_trucks):
          t_distance_index.append(i)

        t_distance.index = t_distance_index

        t_distance = t_distance.append(terminal_df)

        cities= pist [:-1]
        del cities[len(pick_up_loc)-1]

        cities_columns=[]

        for i in range(2*n_trucks,len(terminals)):
          cities_columns.append(i)

        cities.columns = cities_columns

        cities.index = cities_columns


        t_distance[t_distance.isnull()] = cities


        time = round(t_distance/60)
        import pandas as pd
        import numpy as np

        import matplotlib.pyplot as plt



        ### VARIABLES ###
        ### VARIABLES ###
        p_c = 1 # Probability of crossover
        p_m = 0.3 # Probability of mutation
        K = 3 # For Tournament selection
        pop = 100 # Population per generation
        gen = 60 # Number of generations
        ### VARIABLES ###
        ### VARIABLES ###
        import random as rd
        Cap_Dataframe = cap_datframe
        Dist_Dataframe = t_distance
        time_cap_Dataframe = time
        time_limit_Dataframe = time_cap_dataframe
        X0=[]
        for i in range(len(Cap_Dataframe)):
              X0.append(i)

        Solution = ["1","11", "12", "10","2"]
        n_list = np.empty((0,len(X0)))

        for i in range(int(pop)): # Shuffles the elements in the vector n times and stores them
            rnd_sol_1 = rd.sample(X0,len(X0))
            n_list = np.vstack((n_list,rnd_sol_1))
        def Complete_Distance_Not_Random(Dist_array):
              t = 0
              r = 1
              Every_Dist_Truck = []
              for i in Dist_array:
                  if r < len(Dist_array):
                      # For each location and the one next to it, find it on the distance dataframe
                      Dist = Dist_Dataframe.loc[Dist_array[t],Dist_array[r]]
                      Every_Dist_Truck = np.append(Every_Dist_Truck,Dist)
                      t = t+1
                      r = r+1
              return sum(Every_Dist_Truck)

        def Complete_time_Not_Random(Dist_array):
            t = 0
            r = 1
            Every_Dist_Truck = []
            for i in Dist_array:
                if r < len(Dist_array):
                    # For each location and the one next to it, find it on the distance dataframe
                    Dist = time_cap_Dataframe.loc[Dist_array[t],Dist_array[r]]
                    Every_Dist_Truck = np.append(Every_Dist_Truck,Dist)
                    t = t+1
                    r = r+1
            return sum(Every_Dist_Truck)
        def Penalty_1(Array,Penalty_Value):
            Truck_Cap_1_M_1,Truck_Cap_2_M_1 = [],[]     
            for j in Array: # Only the values between 1 and 10 (warehouses)
                if 0 <= j <(2*n_trucks):
                    Truck_Cap_1_M_1 = np.append(Truck_Cap_1_M_1,j) # The warehouses
            for i in Truck_Cap_1_M_1: # The warehouses' indexes
                value_index = np.where(Array == i) # Index of the warehouse
                Truck_Cap_2_M_1 = np.append(Truck_Cap_2_M_1,value_index) # Take the indecies of all the warehouses
            t = 0
            Caps_Sum_One_By_One  = []
            for k in range(len(Truck_Cap_2_M_1)):
                if t < (2*n_trucks-1):
                    Ind_1 = int(Truck_Cap_2_M_1[t])+1 # Index + 1
                    Ind_2 = int(Truck_Cap_2_M_1[t+1]) # Index of second element
                    if Ind_1 in range(2*n_trucks) and Ind_2 in range(2*n_trucks): # If two warehouses were next to each other
                        Truck_Cap_3_M_1_1 = 0 # There would be no truck capacity
                    Truck_Cap_3_M_1_1 = Array[Ind_1:Ind_2] # Else, the truck would be from Ind_1 to Ind_2
                    u = 0
                    Caps_One_By_One = []
                    for o in Truck_Cap_3_M_1_1: # For each item (location) in the truck
                        Caps_Sum = 0
                        Capacity_1 = Truck_Cap_3_M_1_1[u]
                        Capacity_2 = Cap_Dataframe.loc[Capacity_1,'Capacity'] # How much the capacity is
                        Caps_One_By_One = np.append(Caps_One_By_One,Capacity_2) # Append it
                        u = u+1
                    Caps_Sum = sum(Caps_One_By_One) # Add them up
                    Caps_Sum_One_By_One = np.append(Caps_Sum_One_By_One,Caps_Sum) # Append all the sums   
                t = t+1
            Diff_Cap = []
            for i in Caps_Sum_One_By_One: # Capacity of each truck 
                if i > 45000:
                    Diff = i - 45000 # Find the difference
                    Diff_Cap = np.append(Diff_Cap,Diff)
                else:
                    Diff = 0
                    Diff_Cap = np.append(Diff_Cap,Diff)
            How_Much_Extra = sum(Diff_Cap)
            Penalty_M_1 = How_Much_Extra * Penalty_Value
            return Penalty_M_1
        def Penalty_2(Array,Penalty_Value_1,Penalty_Value_2):
              Add_Penalties = []
              if Array[0] not in range(2*n_trucks -1): # If first element in array was not a warehouse
                  
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value_1)
              if Array[-1] not in range(2*n_trucks -1): # If last element in array was not a warehouse
                  
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value_1)
              if Array[1] in range(2*n_trucks -1): # If second elemenet was warehouse
                  
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value_1)
              if Array[-2] in range(2*n_trucks -1): # If second to last was warehouse
                  
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value_1)
              t = 0
              for i in Array:
                  if t < len(Array)-2: # If three cons. elements were warehouses
                      A1 = Array[t]
                      A2 = Array[t+1]
                      A3 = Array[t+2]
                      if A1 in range(2*n_trucks +1) and A2 in range(2*n_trucks +1) and A3 in range(2*n_trucks +1):
                          
                          Add_Penalties = np.append(Add_Penalties,Penalty_Value_2)
                  t = t+1
              Sum_Add_Penalties = sum(Add_Penalties)
              return Sum_Add_Penalties



        def Penalty_3(Array,Penalty_Value):
              Add_Penalties = []
              if Array[0] not in range(2*n_trucks -1): # If first element in array was not a warehouse
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value)
              if Array[-1] not in range(2*n_trucks -1): # If last element in array was not a warehouse
                  Add_Penalties = np.append(Add_Penalties,Penalty_Value)
              Sum_Add_Penalties = sum(Add_Penalties)
              return Sum_Add_Penalties

        def Penalty_4(Array,Penalty_Value_4):
              Truck_Cap_1_M_1,Truck_Cap_2_M_1,ott = [],[],[]    
              for j in Array: # Only the values between 1 and 10 (warehouses)
                  if 0 <= j <(2*n_trucks)-1:
                      Truck_Cap_1_M_1 = np.append(Truck_Cap_1_M_1,j)
              for i in Truck_Cap_1_M_1: # The warehouses' indexes
                  value_index = np.where(Array == i)
                  Truck_Cap_2_M_1 = np.append(Truck_Cap_2_M_1,value_index)
              t = 0
              time_Caps_Sum_One_By_One,caps_max =[],[]
              for k in range(len(Truck_Cap_2_M_1)):
                  if t < (2*n_trucks -1):
                      Ind_1 = int(Truck_Cap_2_M_1[t])+1
                      Ind_2 = int(Truck_Cap_2_M_1[t])
                      if Ind_1 in range(11) and Ind_2 in range(11):
                          Truck_Cap_3_M_1_1 = 0
                      Truck_Cap_3_M_1_1 = Array[Ind_1:Ind_2]
                      u = 0
                        
                      #print()
                      #print("Cities between Two WH:",Truck_Cap_3_M_1_1)
                      truck_4 = np.append(Truck_Cap_3_M_1_1,1)
                        
                      time_capacity=[]
                      if len(Truck_Cap_3_M_1_1) != 0:
                          for o in Truck_Cap_3_M_1_1:
                                
                                Capacity_1 = Truck_Cap_3_M_1_1[u]
                                Capacity_2 = time_limit_Dataframe.loc[Capacity_1,'Time_Capacity']
                                time_capacity = np.append(time_capacity,Capacity_2)
                                caps_max=max(time_capacity)
                                truck_4 = np.append(Truck_Cap_3_M_1_1,1)
                                tt = Complete_time_Not_Random(truck_4)
                          time_Caps_Sum_One_By_One = np.append(time_Caps_Sum_One_By_One,caps_max)

                          ott=np.append(ott,tt)
                  t=t+1
              Diff_time = []
              for i in ott:
                for j in time_Caps_Sum_One_By_One: # Capacity of each truck 
                  if i > j:
                      Diff = i - j # Find the difference
                      Diff_time = np.append(Diff_time,Diff)
                  else:
                      Diff = 0
                      Diff_time = np.append(Diff_time,Diff)
              How_Much_Extra_1 = sum(Diff_time)
              Penalty_M_4 = How_Much_Extra_1 * Penalty_Value_4
              return Penalty_M_4
        def fitness(Array):

            Cost = Complete_Distance_Not_Random(Array) # The distance function
            
            a1 = Penalty_1(Array,10000000)
            b1 = Penalty_2(Array,20000000,30000000) # The penalty of warehouses' locations
            c1 = Penalty_3(Array,40000000) # The penalty of warehouses' locations
            d1 = Penalty_4(Array,3000)
            Total_Distance_N_i = Cost+ a1 +b1+c1
            return(Total_Distance_N_i)     
        Final_Best_in_Generation_X = []
        Worst_Best_in_Generation_X = []

        For_Plotting_the_Best = np.empty((0,len(X0)+1))

        One_Final_Guy = np.empty((0,len(X0)+2))
        One_Final_Guy_Final = []

        Min_for_all_Generations_for_Mut_1 = np.empty((0,len(X0)+1))
        Min_for_all_Generations_for_Mut_2 = np.empty((0,len(X0)+1))

        Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(X0)+2))
        Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(X0)+2))

        Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(X0)+2))
        Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(X0)+2))

        Generation = 1 


        for i in range(gen):
            
            
            New_Population = np.empty((0,len(X0))) # Saving the new generation
            
            All_in_Generation_X_1 = np.empty((0,len(X0)+1))
            All_in_Generation_X_2 = np.empty((0,len(X0)+1))
            
            Min_in_Generation_X_1 = []
            Min_in_Generation_X_2 = []
            
            Save_Best_in_Generation_X = np.empty((0,len(X0)+1))
            Final_Best_in_Generation_X = []
            Worst_Best_in_Generation_X = []
            
            
            #print()
            #print("--> GENERATION: #",Generation)
            
            Family = 1
            
            for j in range(int(pop/2)): # range(int(pop/2))
                
                #print()
                #print("--> FAMILY: #",Family)
                  
                
                # Tournament Selection
                # Tournament Selection
                # Tournament Selection
                
                Parents = np.empty((0,len(X0)))
                
                for i in range(2):
                    
                    Battle_Troops = []
                    
                    Warrior_1_index = np.random.randint(0,len(n_list))
                    Warrior_2_index = np.random.randint(0,len(n_list))
                    Warrior_3_index = np.random.randint(0,len(n_list))
                    
                    while Warrior_1_index == Warrior_2_index:
                        Warrior_1_index = np.random.randint(0,len(n_list))
                    while Warrior_2_index == Warrior_3_index:
                            Warrior_3_index = np.random.randint(0,len(n_list))
                    while Warrior_1_index == Warrior_3_index:
                            Warrior_3_index = np.random.randint(0,len(n_list))
                    
                    Warrior_1 = n_list[Warrior_1_index,:]
                    Warrior_2 = n_list[Warrior_2_index,:]
                    Warrior_3 = n_list[Warrior_3_index,:]
                    
                    Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
                    
                    
                    # For Warrior #1
                    Prize_Warrior_1 = fitness(Warrior_1) 
                    
                    
                    # For Warrior #2
                    Prize_Warrior_2 = fitness(Warrior_2)
                    
                    
                    # For Warrior #3
                    Prize_Warrior_3 = fitness(Warrior_3)
                    
                    
                    
                    
                    if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                        Winner = Warrior_1
                    elif Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                        Winner = Warrior_2
                    else:
                        Winner = Warrior_3
                    
                
                    Parents = np.vstack((Parents,Winner))
                
                
                
                Parent_1 = Parents[0]
                Parent_2 = Parents[1]
                
                
                Child_1 = np.empty((0,len(X0)))
                Child_2 = np.empty((0,len(X0)))
                
                
                # Where to crossover
                
                Ran_CO_1 = np.random.rand()
                
                if Ran_CO_1 < p_c:
                    
                    # Choose two random numbers to crossover with their locations
                    Cr_1 = np.random.randint(0,len(X0))
                    Cr_2 = np.random.randint(0,len(X0))
                    
                    while Cr_1 == Cr_2:
                        Cr_2 = np.random.randint(0,len(X0))
                    
                    if Cr_1 < Cr_2:
                    
                        Cr_2 = Cr_2 + 1
                        
                        
                        New_Dep_1 = Parent_1[Cr_1:Cr_2] # Mid seg. of parent #1
                        
                        New_Dep_2 = Parent_2[Cr_1:Cr_2] # Mid seg. of parent #2
                        
                        First_Seg_1 = Parent_1[:Cr_1] # First seg. of parent #1
                        
                        First_Seg_2 = Parent_2[:Cr_1] # First seg. of parent #2
                
                        Temp_First_Seg_1_1 = [] # Temporay for first seg.
                        Temp_Second_Seg_2_2 = [] # Temporay for second seg.
                        
                        Temp_First_Seg_3_3 = [] # Temporay for first seg.
                        Temp_Second_Seg_4_4 = [] # Temporay for second seg.
                        
                        
                        
                        for i in First_Seg_2: # For i in all the elements of the first segment of parent #2
                            if i not in New_Dep_1: # If they aren't in seg. parent #1
                                Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1,i) # Append them
                        
                        Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1)) # Add it next to the mid seg.
                        
                        for i in Parent_2: # For parent #2
                            if i not in Temp_New_Vector_1: # If not in what is made so far ^^
                                Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2,i) # Append it
                        
                        Child_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1,Temp_Second_Seg_2_2)) # Now you can make the child from the elements
                        
                        for i in First_Seg_1: # Do the same for child #2
                            if i not in New_Dep_2:
                                Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3,i)
                        
                        Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2))
                
                        for i in Parent_1:
                            if i not in Temp_New_Vector_2:
                                Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4,i)
                        
                        Child_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2,Temp_Second_Seg_4_4))
            
                    else: # The same in reverse of Cr_1 and Cr_2
                    
                        Cr_1 = Cr_1 + 1
                        
                        New_Dep_1 = Parent_1[Cr_2:Cr_1]
                        
                        New_Dep_2 = Parent_2[Cr_2:Cr_1]
                        
                        First_Seg_1 = Parent_1[:Cr_2]
                        
                        First_Seg_2 = Parent_2[:Cr_2]
                
                        Temp_First_Seg_1_1 = []
                        Temp_Second_Seg_2_2 = []
                        
                        Temp_First_Seg_3_3 = []
                        Temp_Second_Seg_4_4 = []
                        
                        for i in First_Seg_2:
                            if i not in New_Dep_1:
                                Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1,i)
                        
                        Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1))
                        
                        for i in Parent_2:
                            if i not in Temp_New_Vector_1:
                                Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2,i)
                        
                        Child_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1,Temp_Second_Seg_2_2))
                        
                        for i in First_Seg_1:
                            if i not in New_Dep_2:
                                Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3,i)
                        
                        Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2))
                
                        for i in Parent_1:
                            if i not in Temp_New_Vector_2:
                                Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4,i)
                        
                        Child_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2,Temp_Second_Seg_4_4))
                        
                
                else: # If random number was above p_c
                    
                    Child_1 = Parent_1
                    Child_2 = Parent_2
                    
            
                # Mutation Child #1
                # Mutation Child #1
                # Mutation Child #1
                
                Mutated_Child_1 = []

                
                Ran_Mut_1 = np.random.rand() # Probablity to Mutate
                Ran_Mut_2 = np.random.randint(0,len(X0))
                Ran_Mut_3 = np.random.randint(0,len(X0))
                
                A1 = Ran_Mut_2
                A2 = Ran_Mut_3
                
                while A1 == A2:
                    A2 = np.random.randint(0,len(X0))
                
                if Ran_Mut_1 < p_m: # If probablity to mutate is less than p_m, then mutate
                    if A1 < A2:
                        M_Child_1_Pos_1 = Child_1[A1] # Take the index
                        M_Child_1_Pos_2 = Child_1[A2] # Take the index
                        A2 = A2+1
                        Rev_1 = Child_1[:] # copy of child #1
                        Rev_2 = list(reversed(Child_1[A1:A2])) # reverse the order
                        t = 0
                        for i in range(A1,A2):
                            Rev_1[i] = Rev_2[t] # The reversed will become instead of the original
                            t = t+1
                        
                        Mutated_Child_1 = Rev_1
                    
                    else:
                        M_Child_1_Pos_1 = Child_1[A2]
                        M_Child_1_Pos_2 = Child_1[A1]
                        A1 = A1+1
                        Rev_1 = Child_1[:]
                        Rev_2 = list(reversed(Child_1[A2:A1]))
                        t = 0
                        for i in range(A2,A1):
                            Rev_1[i] = Rev_2[t]
                            t = t+1
                        
                        Mutated_Child_1 = Rev_1
                else:
                    Mutated_Child_1 = Child_1
                
                
                
                Mutated_Child_2 = []

                
                Ran_Mut_1 = np.random.rand() # Probablity to Mutate
                Ran_Mut_2 = np.random.randint(0,len(X0))
                Ran_Mut_3 = np.random.randint(0,len(X0))
                
                A1 = Ran_Mut_2
                A2 = Ran_Mut_3
                
                while A1 == A2:
                    A2 = np.random.randint(0,len(X0))
                
                if Ran_Mut_1 < p_m: # If probablity to mutate is less than p_m, then mutate
                    if A1 < A2:
                        M_Child_1_Pos_1 = Child_2[A1]
                        M_Child_1_Pos_2 = Child_2[A2]
                        A2 = A2+1
                        Rev_1 = Child_2[:]
                        Rev_2 = list(reversed(Child_2[A1:A2]))
                        t = 0
                        for i in range(A1,A2):
                            Rev_1[i] = Rev_2[t]
                            t = t+1
                        
                        Mutated_Child_2 = Rev_1
                    
                    else:
                        M_Child_1_Pos_1 = Child_2[A2]
                        M_Child_1_Pos_2 = Child_2[A1]
                        A1 = A1+1
                        Rev_1 = Child_2[:]
                        Rev_2 = list(reversed(Child_2[A2:A1]))
                        t = 0
                        for i in range(A2,A1):
                            Rev_1[i] = Rev_2[t]
                            t = t+1
                        
                        Mutated_Child_2 = Rev_1
                else:
                    Mutated_Child_2 = Child_2
                
                
                
                
                Total_Cost_Mut_1 = fitness(Mutated_Child_1) 
                
                Total_Cost_Mut_2 = fitness(Mutated_Child_2) 
                
                
                #print()
                #print("FV at Mutated Child #1 at Gen #",Generation,":", Total_Cost_Mut_1)
                #print("FV at Mutated Child #2 at Gen #",Generation,":", Total_Cost_Mut_2)
                
                
                
                All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
                All_in_Generation_X_1_1 = np.column_stack((Total_Cost_Mut_1, All_in_Generation_X_1_1_temp))
                
                All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
                All_in_Generation_X_2_1 = np.column_stack((Total_Cost_Mut_2, All_in_Generation_X_2_1_temp))
                
                All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
                All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
                
                
                Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
                
                
                New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
                
                t = 0
                
                R_1 = []
                for i in All_in_Generation_X_1:
                    
                    if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                        R_1 = All_in_Generation_X_1[t,:]
                    t = t+1
                      
                Min_in_Generation_X_1 = R_1[np.newaxis]
                
                
                t = 0
                R_2 = []
                for i in All_in_Generation_X_2:
                    
                    if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                        R_2 = All_in_Generation_X_2[t,:]
                    t = t+1
                        
                Min_in_Generation_X_2 = R_2[np.newaxis]
                
                
                Family = Family+1
            
            t = 0
            R_Final = []
            
            for i in Save_Best_in_Generation_X:
                
                if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
                    R_Final = Save_Best_in_Generation_X[t,:]
                t = t+1
            
            Final_Best_in_Generation_X = R_Final[np.newaxis]
            
            
            
            For_Plotting_the_Best = np.vstack((For_Plotting_the_Best,Final_Best_in_Generation_X))
            
            t = 0
            R_22_Final = []
            
            for i in Save_Best_in_Generation_X:
                
                if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
                    R_22_Final = Save_Best_in_Generation_X[t,:]
                t = t+1
            
            Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
            
            
            
            
            # Elitism, the best in the generation lives
            # Elitism, the best in the generation lives
            # Elitism, the best in the generation lives
            
            Darwin_Guy = Final_Best_in_Generation_X[:]
            Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
            
            Darwin_Guy = Darwin_Guy[0:,1:].tolist()
            Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
            
            
            Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
            Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
            
            
            New_Population[Worst_1] = Darwin_Guy
            
            
            n_list = New_Population
            

            Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
            Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
            
            Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
            Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
            
            Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
            Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
            
            Generation = Generation+1
            



        One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))
            
        t = 0
        Final_Here = []
        for i in One_Final_Guy:
            
            if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
                Final_2 = []
                Final_2 = [One_Final_Guy[t,1]]
                Final_Here = One_Final_Guy[t,:]
            t = t+1
                
        A_2_Final = min(One_Final_Guy[:,1])

        One_Final_Guy_Final = Final_Here[np.newaxis]
        Final_sol = One_Final_Guy_Final[:,2:]

        Final_sol = Final_sol.tolist()

        Final_sol = Final_sol[0]
        To_Save_Capacities = Final_sol
        Truck_Cap_1_M_1,Truck_Cap_2_M_1 = [],[]    
        for j in To_Save_Capacities: # Only the values between 1 and 10 (warehouses)
            if 1 <= j <= 2*n_trucks:
                Truck_Cap_1_M_1 = np.append(Truck_Cap_1_M_1,j)
        for i in Truck_Cap_1_M_1: # The warehouses' indexes
            value_index = np.where(To_Save_Capacities == i)
            Truck_Cap_2_M_1 = np.append(Truck_Cap_2_M_1,value_index)
        t = 0
        Caps_All, Caps_Sum_One_By_One  = [],[]
        for k in range(len(Truck_Cap_2_M_1)):
            if t < (2*n_trucks-1):
                Ind_1 = int(Truck_Cap_2_M_1[t])+1
                Ind_2 = int(Truck_Cap_2_M_1[t])
                if Ind_1 in range(2*n_trucks+1) and Ind_2 in range(2*n_trucks+1):
                    Truck_Cap_3_M_1_1 = 0
                Truck_Cap_3_M_1_1 = To_Save_Capacities[Ind_1:Ind_2]
                u = 0
                '''
                print()
                print("Cities between Two WH:",Truck_Cap_3_M_1_1)
                '''
                Caps_One_By_One = []
                if len(Truck_Cap_3_M_1_1) != 0:
                    for o in Truck_Cap_3_M_1_1:
                        Caps_Sum = 0
                        Capacity_1 = Truck_Cap_3_M_1_1[u]
                        Capacity_2 = Cap_Dataframe.loc[Capacity_1,"Capacity"]
                        Caps_One_By_One = np.append(Caps_One_By_One,Capacity_2)
                        Caps_All = np.append(Caps_All,Capacity_2)
                        u = u+1
                        Caps_Sum = sum(Caps_One_By_One)
                    Caps_Sum_One_By_One = np.append(Caps_Sum_One_By_One,Caps_Sum)
            t = t+1

        #print("Trucks' Capacities:",Caps_Sum_One_By_One)
        city_df1=city_df.reindex(index=Final_sol)
        city_df1.to_csv("plan_{}.csv".format(I))

    except KeyError:
      pass
